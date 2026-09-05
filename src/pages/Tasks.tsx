import React, { useCallback, useEffect, useState } from 'react';
import { RecoveryControls } from '../components/run/RecoveryControls';
import {
  AlertTriangle,
  ArrowLeft,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Loader2,
  Play,
  RotateCcw,
  ShieldCheck,
} from 'lucide-react';
import {
  agentService,
  type ArtifactReport,
  type TaskRecord,
  type TaskSnapshot,
  type RunState,
  type TaskSummary,
} from '../services/agent.service';
import {
  compactionWarning,
  describeCompaction,
  explainLedger,
  fitted,
  ledgerRows,
} from '../components/run/context-ledger';
import { useToast } from '../hooks/useToast';
import { isBusy } from '../components/run/useRun';
import { useActiveRun } from '../contexts/ActiveRunContext';
import styles from './Tasks.module.css';

/**
 * Every task this machine has run, and what each one rests on.
 *
 * PS 26117 asks for exactly this: *"every task you have run, with its plan, the
 * models it chose and why, the evidence it retrieved, and the artifacts it
 * produced."* The list is deliberately not filtered to successes — the run
 * somebody comes here to look at is usually the one that went wrong, and a
 * screen showing only good news would be worse than no screen.
 *
 * ## Why the artifacts are checked again on opening
 *
 * The saved record says what the check found when the run ended. This page also
 * asks the backend to re-open the files now, because a deliverable can be
 * moved, replaced or truncated long after the run that made it. The two
 * disagreeing is information, so both are shown rather than the newer quietly
 * replacing the older.
 */

/** How each ending is drawn. Anything unmapped falls back to the ready/review
 *  pair, so a status from a newer backend still renders. */
const STATUS_STYLE: Partial<Record<RunState, string>> = {
  created: styles.rowLive,
  classified: styles.rowLive,
  routed: styles.rowLive,
  planned: styles.rowLive,
  running: styles.rowLive,
  awaiting_approval: styles.rowLive,
  executing_tool: styles.rowLive,
  tool_result_recorded: styles.rowLive,
  verifying: styles.rowLive,
  // Not faults. Drawn apart from `failed` so a list of them does not train
  // people to skip the row that actually broke.
  degraded_needs_human: styles.rowInterrupted,
  cancelled: styles.rowInterrupted,
  stopped_by_budget: styles.rowInterrupted,
  stopped_by_policy: styles.rowInterrupted,
};

const KIND_ICONS = {
  document: FileText,
  workbook: FileSpreadsheet,
  text: FileText,
} as const;

/**
 * What a row's tag says.
 *
 * Distinguishing the endings matters more here than anywhere else in the
 * product: this is the screen somebody opens to find out what happened, and a
 * run that was stopped on purpose, one that hit its time budget, and one the
 * application closed on top of are three different stories that all used to
 * read as "failed".
 */
function statusLabel(state: RunState, ready: boolean): string {
  switch (state) {
    case 'paused': return 'paused';
    case 'recovering': return 'ready for recovery';
    case 'compacting': return 'compacting context';
    case 'waiting_for_external_event': return 'waiting';
    case 'stopped_by_length': return 'output limit';
    case 'created':
    case 'classified':
    case 'routed':
    case 'planned':
      return 'starting';
    case 'running':
    case 'tool_result_recorded':
      return 'running';
    case 'awaiting_approval':
      return 'needs approval';
    case 'executing_tool':
      return 'running a tool';
    case 'verifying':
      return 'verifying';
    case 'cancelled':
      return 'stopped';
    case 'stopped_by_budget':
      return 'reached its limit';
    case 'stopped_by_policy':
      return 'not permitted';
    case 'degraded_needs_human':
      return 'interrupted';
    case 'failed':
      return 'failed';
    case 'completed':
    default:
      return ready ? 'ready' : 'needs review';
  }
}

function when(iso: string): string {
  const at = new Date(iso);
  return Number.isNaN(at.getTime())
    ? iso
    : at.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

function howLong(seconds: number): string {
  if (seconds < 60) return `${seconds}s`;
  return `${Math.floor(seconds / 60)}m ${seconds % 60}s`;
}

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function ArtifactRow({
  artifact,
  now,
  onReveal,
}: {
  artifact: ArtifactReport;
  /** The same file as it is on disk today. Absent until the re-check lands. */
  now: ArtifactReport | undefined;
  onReveal: (name: string) => void;
}) {
  const Icon = KIND_ICONS[artifact.kind];
  // Only worth calling out when it *changed*. Printing "still sound" on every
  // row would train people to skip the line that matters.
  const changed = now && now.sound !== artifact.sound;

  return (
    <li className={styles.artifact}>
      <Icon size={16} className={styles.artifactIcon} />
      <div className={styles.artifactBody}>
        <div className={styles.artifactName}>
          <strong>{artifact.name}</strong>
          <span className={styles.dim}>{size(artifact.bytes)}</span>
          <span className={artifact.sound ? styles.tagSound : styles.tagUnsound}>
            {artifact.sound ? 'checked out when produced' : 'did not pass its check'}
          </span>
        </div>
        <p className={styles.artifactDetail}>{artifact.detail}</p>
        {changed && now && (
          <p className={styles.changed}>
            <AlertTriangle size={13} />
            <span>On disk now: {now.detail}</span>
          </p>
        )}
        {artifact.problems.length > 0 && (
          <ul className={styles.problems}>
            {artifact.problems.map((text, i) => (
              <li key={i}>{text}</li>
            ))}
          </ul>
        )}
      </div>
      <button
        className={styles.revealBtn}
        onClick={() => onReveal(artifact.name)}
        aria-label={`Show ${artifact.name} in the file manager`}
      >
        <FolderOpen size={14} />
      </button>
    </li>
  );
}

function Detail({ runId, onBack }: { runId: string; onBack: () => void }) {
  const [record, setRecord] = useState<TaskRecord | null>(null);
  const [onDisk, setOnDisk] = useState<ArtifactReport[]>([]);
  const [error, setError] = useState<string | null>(null);
  /** What is known about a run that has no record: one still going, or one the
   *  process took down with it. Both are exactly the rows somebody clicks. */
  const [known, setKnown] = useState<TaskSnapshot | null>(null);

  useEffect(() => {
    let live = true;
    void (async () => {
      try {
        const loaded = await agentService.task(runId);
        if (!live) return;
        setRecord(loaded);
        // Best-effort, and after the record: a re-check that fails should not
        // stop the task being readable.
        try {
          const fresh = await agentService.taskArtifacts(runId);
          if (live) setOnDisk(fresh);
        } catch {
          // Leaves the rows showing what the run found, which is still a true
          // statement about the moment it ran.
        }
      } catch (e) {
        // A run writes its record when it ends, so a run that has not ended —
        // or was never allowed to — has none. That is not an error to show
        // somebody who has just clicked the row: the durable history knows
        // what it managed to do, and showing that is the whole point of
        // keeping it.
        const snapshot = await agentService.snapshot(runId).catch(() => null);
        if (!live) return;
        if (snapshot) setKnown(snapshot);
        else setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      live = false;
    };
  }, [runId]);

  const reveal = useCallback(
    (name: string) => {
      void agentService.revealArtifact(runId, name).catch(e => {
        setError(e instanceof Error ? e.message : String(e));
      });
    },
    [runId],
  );

  const back = (
    <button className={styles.back} onClick={onBack}>
      <ArrowLeft size={14} />
      <span>All tasks</span>
    </button>
  );

  if (error) {
    return (
      <div className={styles.page}>
        {back}
        <p className={styles.failure} role="alert">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </p>
      </div>
    );
  }

  // No record, but a history. Everything below it needs a `TaskRecord` — an
  // answer, evidence, a verification — and this run has none of those, so what
  // is shown is what actually happened rather than a page of empty sections.
  if (!record && known) {
    return (
      <div className={styles.page}>
        {back}
        <h1 className={styles.prompt}>{known.prompt}</h1>
        <RecoveryControls key={runId} runId={runId} />
        <p className={styles.meta}>
          {statusLabel(known.state, false)} &middot;{' '}
          {known.modelName || 'not routed yet'} &middot; started {when(known.startedAt)}
        </p>
        <p className={styles.failure} role="status">
          <AlertTriangle size={15} />
          <span>
            {known.failure ?? 'This task has not finished yet.'} It has no full record — a task writes
            one when it ends — so what follows is what was written down as it happened.
            {known.unreadableEvents.length > 0 &&
              ` ${known.unreadableEvents.length} step(s) of that history could not be read, and are missing below.`}
          </span>
        </p>

        <section className={styles.card}>
          <h2 className={styles.cardTitle}>What it did</h2>
          {known.activity.length === 0 ? (
            <p className={styles.dim}>
              {known.state === 'running'
                ? 'Nothing has been called yet.'
                : 'It stopped before calling anything.'}
            </p>
          ) : (
            <ol className={styles.steps}>
              {known.activity.map(item => (
                <li key={item.toolCallId} className={styles.step}>
                  <span>{item.tool}</span>
                  <span className={styles.dim}>{item.status}</span>
                </li>
              ))}
            </ol>
          )}
        </section>

        {known.artifacts.length > 0 && (
          <section className={styles.card}>
            <h2 className={styles.cardTitle}>Files it had produced</h2>
            <ul className={styles.problems}>
              {known.artifacts.map(name => (
                <li key={name}>{name}</li>
              ))}
            </ul>
            <p className={styles.dim}>
              These were written before it stopped, and were never checked — a file is re-opened
              and verified when a task finishes, and this one did not.
            </p>
          </section>
        )}
      </div>
    );
  }

  if (!record) {
    return (
      <div className={styles.page}>
        {back}
        <p className={styles.dim}>Reading this task&rsquo;s record&hellip;</p>
      </div>
    );
  }

  const verification = record.verification;
  const standing = verification?.standing;

  return (
    <div className={styles.page}>
      {back}

      <header className={styles.detailHead}>
        <h1 className={styles.prompt}>{record.prompt}</h1>
        <RecoveryControls key={runId} runId={runId} />
        <p className={styles.meta}>
          {when(record.startedAt)} &middot; {howLong(record.durationSeconds)} &middot; {record.turns}{' '}
          turn(s)
        </p>
      </header>

      {record.failure && (
        <p className={styles.failure} role="alert">
          <AlertTriangle size={15} />
          <span>{record.failure}</span>
        </p>
      )}

      <section className={styles.card}>
        <h2 className={styles.cardTitle}>Model</h2>
        <p className={styles.body}>
          <strong>{record.routing.modelName}</strong> took this as{' '}
          {record.routing.intent.toLowerCase()} work
          {record.routing.usedFallback ? ', after the first choice did not fit' : ''}.
        </p>
        {/* In the order they applied, and not summarised: these are the answer
          * to "why this model", and a paraphrase is not that answer. */}
        <ul className={styles.reasons}>
          {record.routing.reasons.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
        <p className={styles.dim}>
          {record.routing.gpuPlanSummary} &middot;{' '}
          {record.endpoint.runtime === 'llamaCpp' ? 'llama.cpp' : 'Python sidecar'} on{' '}
          {record.endpoint.baseUrl}
        </p>
      </section>

      <section className={styles.card}>
        <header className={styles.cardHead}>
          <h2 className={styles.cardTitle}>Plan</h2>
          <span className={styles.dim}>
            {record.plan.stepsTaken} of {record.plan.maxSteps} tool calls
          </span>
        </header>
        {/* What the run set out to do, with no per-step tick: one planned step
          * can take several tool calls, so reaching the end of the loop does
          * not establish that each step was carried out. The produced files
          * and the check are the evidence for that. */}
        <ol className={styles.steps}>
          {record.plan.steps.map(step => (
            <li key={step.ordinal} className={styles.step}>
              <span className={styles.stepMark} aria-hidden>
                {step.ordinal}
              </span>
              <span>{step.intent}</span>
            </li>
          ))}
        </ol>
        <p className={styles.dim}>{record.plan.stoppedBecause}</p>
      </section>

      {record.artifacts.length > 0 && (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Produced</h2>
          <ul className={styles.artifacts}>
            {record.artifacts.map(artifact => (
              <ArtifactRow
                key={artifact.path}
                artifact={artifact}
                now={onDisk.find(item => item.path === artifact.path)}
                onReveal={reveal}
              />
            ))}
          </ul>
        </section>
      )}

      {verification && standing && (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Checked</h2>
          <p
            className={standing.standing === 'ready' ? styles.verdictReady : styles.verdictReview}
          >
            {standing.standing === 'ready' ? (
              <ShieldCheck size={15} />
            ) : (
              <AlertTriangle size={15} />
            )}
            <span>
              {standing.standing === 'ready'
                ? 'Every claim resolved to a passage this task retrieved, and its figures matched the recorded calculations.'
                : `${standing.blocking} thing(s) needed checking before this was relied on, and ${standing.advisory} were worth a look.`}
            </span>
          </p>
          {verification.findings.length > 0 && (
            <ul className={styles.findings}>
              {verification.findings.map((finding, i) => (
                <li
                  key={i}
                  className={
                    finding.severity === 'blocking'
                      ? `${styles.finding} ${styles.findingBlocking}`
                      : styles.finding
                  }
                >
                  {finding.detail}
                </li>
              ))}
            </ul>
          )}
        </section>
      )}

      {record.calculations.length > 0 && (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Working</h2>
          {/* The engine's record, not the model's account of it. Every step is
            * here because a figure nobody can retrace is one nobody can check. */}
          <ul className={styles.calculations}>
            {record.calculations.map((calculation, i) => (
              <li key={i} className={styles.calculation}>
                <code className={styles.expression}>{calculation.expression}</code>
                <ol className={styles.calcSteps}>
                  {calculation.steps.map((step, j) => (
                    <li key={j}>
                      {step.description} &rarr; {step.result}
                    </li>
                  ))}
                </ol>
                <p className={styles.result}>
                  = {calculation.formatted}{' '}
                  <span className={styles.dim}>({calculation.rounding})</span>
                </p>
              </li>
            ))}
          </ul>
        </section>
      )}

      {record.evidence.length > 0 && (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Evidence</h2>
          <ul className={styles.evidence}>
            {record.evidence.map(passage => (
              <li key={passage.marker} className={styles.passage}>
                <span className={styles.marker}>[E{passage.marker}]</span>
                <div className={styles.passageBody}>
                  <p className={styles.citation}>{passage.citation}</p>
                  <p className={styles.excerpt}>{passage.excerpt}</p>
                </div>
              </li>
            ))}
          </ul>
        </section>
      )}

      {(record.contextLedger || (record.compactions?.length ?? 0) > 0) && (
        <section className={styles.card}>
          <h2 className={styles.cardTitle}>Context</h2>
          {/* The question this section answers is not "how many tokens" but
              "why did this run keep losing its history". The sentence comes
              first for that reason; the table is for whoever wants to check
              it. */}
          {record.contextLedger && (
            <>
              {explainLedger(record.contextLedger) && (
                <p className={styles.body}>{explainLedger(record.contextLedger)}</p>
              )}
              <ul className={styles.ledger}>
                {ledgerRows(record.contextLedger).map(row => (
                  <li
                    key={row.section}
                    className={`${styles.ledgerRow}${
                      row.committedNotOccupied ? ` ${styles.ledgerReserve}` : ''
                    }`}
                  >
                    <span>{row.label}</span>
                    <span className={styles.ledgerBar}>
                      <span
                        className={styles.ledgerFill}
                        style={{ width: `${Math.round(row.share * 100)}%` }}
                      />
                    </span>
                    <span className={styles.ledgerTokens}>
                      {row.tokens.toLocaleString()}
                    </span>
                  </li>
                ))}
              </ul>
              {/* Said only when it is known. An unrecorded window is not
                  evidence that anything did not fit. */}
              {fitted(record.contextLedger) === false && (
                <p className={styles.dim}>
                  The next turn would not have fitted in this model&rsquo;s window.
                </p>
              )}
            </>
          )}

          {(record.compactions?.length ?? 0) > 0 && (
            <>
              {compactionWarning(record.compactions ?? []) && (
                <p className={styles.body}>{compactionWarning(record.compactions ?? [])}</p>
              )}
              <ul className={styles.compactions}>
                {(record.compactions ?? []).map(pass => (
                  <li key={`${pass.ordinal}-${pass.at}`} className={styles.compaction}>
                    {describeCompaction(pass)}
                  </li>
                ))}
              </ul>
            </>
          )}

          {/* What a resumption of this run would read. Shown because a person
              deciding whether to re-run a failed task needs to know it will not
              redo the document it already produced. */}
          {record.workingNotes && record.workingNotes.completed.length > 0 && (
            <p className={styles.dim}>
              A resumption of this task would already know it had done:{' '}
              {record.workingNotes.completed
                .map(effect => `${effect.tool} → ${effect.target}`)
                .join(', ')}
              .
            </p>
          )}
        </section>
      )}

      <section className={styles.card}>
        <h2 className={styles.cardTitle}>Answer</h2>
        {record.answer.trim() ? (
          <div className={styles.answer}>{record.answer}</div>
        ) : (
          <p className={styles.dim}>This task ended without an answer.</p>
        )}
      </section>
    </div>
  );
}

/**
 * Two affordances on every history row.
 *
 * "Open" reads the saved record — the right thing when the operator wants to
 * audit what the run actually did. "Replay" starts a fresh run with the same
 * prompt — the right thing when they want to see the answer again, or try a
 * different model on the same input. The two are deliberately separate
 * actions: a click that secretly started a long run would be worse than
 * confusing.
 *
 * Replay is blocked while another run is in flight anywhere in the app, and
 * it is blocked on a row whose run is still live — replaying a running task
 * is the same as starting a duplicate.
 */
function TaskRowActions({
  task,
  onOpen,
}: {
  task: TaskSummary;
  onOpen: (runId: string) => void;
}) {
  // Shared, so a replayed task is the run every other surface follows.
  const { state, start } = useActiveRun();
  const { addToast } = useToast();
  const busy = isBusy(state.phase);
  const disabled = busy || task.live;
  const onReplay = useCallback(async () => {
    if (disabled) return;
    try {
      const summary = await start(task.prompt, undefined, {
        correlationId: `replay-${task.runId}-${Date.now()}`,
      });
      if (summary) onOpen(summary.runId);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      addToast('error', `Replay failed: ${message}`);
    }
  }, [disabled, start, task.prompt, task.runId, onOpen, addToast]);

  return (
    <div className={styles.rowActions}>
      <button
        type="button"
        className={styles.replayBtn}
        onClick={() => void onReplay()}
        disabled={disabled}
        title={
          task.live
            ? 'This task is still running; wait for it to finish.'
            : busy
              ? 'Another run is in progress.'
              : 'Run the same prompt again'
        }
        aria-label="Replay this task"
      >
        {disabled && task.live ? (
          <Loader2 size={13} className={styles.spin} />
        ) : (
          <RotateCcw size={13} />
        )}
        <span>Replay</span>
      </button>
      <button
        type="button"
        className={styles.openBtn}
        onClick={() => onOpen(task.runId)}
        title="Open this task's record"
        aria-label="Open this task's record"
      >
        <Play size={13} />
        <span>Open</span>
      </button>
    </div>
  );
}

export const Tasks = () => {
  const [tasks, setTasks] = useState<TaskSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
  const { addToast } = useToast();

  const load = useCallback(async () => {
    try {
      setTasks(await agentService.history());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setTasks([]);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (open) {
    return (
      <Detail
        runId={open}
        onBack={() => {
          setOpen(null);
          // Re-read on the way back: a run may have finished while the detail
          // was open, and a list that silently omits it is one nobody trusts.
          void load();
        }}
      />
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.head}>
        <h1 className={styles.title}>Tasks</h1>
        <p className={styles.subtitle}>
          Every task run on this machine, with the plan it was held to, the model it chose and why,
          the evidence it retrieved and the files it produced.
        </p>
      </header>

      {error && (
        <p className={styles.failure} role="alert">
          <AlertTriangle size={15} />
          <span>{error}</span>
        </p>
      )}

      {tasks === null ? (
        <p className={styles.dim}>Reading the task records&hellip;</p>
      ) : tasks.length === 0 ? (
        <p className={styles.empty}>
          Nothing has been run yet. Ask something in the workbench and it will appear here with
          everything it rested on.
        </p>
      ) : (
        <ul className={styles.list}>
          {tasks.map(task => (
            <li key={task.runId} className={styles.row}>
              <button
                className={styles.rowOpen}
                onClick={() => setOpen(task.runId)}
                aria-label={`Open task ${task.prompt}`}
              >
                <span className={styles.rowPrompt}>{task.prompt}</span>
                <span className={styles.rowMeta}>
                  {task.live ? 'started ' + when(task.startedAt) : when(task.finishedAt)} &middot;{' '}
                  {task.modelName || 'routing'} &middot;{' '}
                  {task.live ? 'still running' : howLong(task.durationSeconds)}
                  {task.artifactCount > 0 && ` · ${task.artifactCount} file(s)`}
                  {task.evidenceCount > 0 && ` · ${task.evidenceCount} passage(s)`}
                </span>
                {/* "Ready" means it finished, its claims resolved and its files
                  * opened — not merely that it did not crash. The three
                  * endings that are not failures are named rather than folded
                  * into one: a run somebody stopped, a run that ran out of
                  * time, and a run the application closed on top of all read
                  * as "failed" otherwise, and only one of those is a fault. */}
                <span className={STATUS_STYLE[task.state] ?? (task.ready ? styles.rowReady : styles.rowReview)}>
                  {statusLabel(task.state, task.ready)}
                </span>
              </button>
              <TaskRowActions task={task} onOpen={runId => setOpen(runId)} />
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};
