import React, { useState } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  CircleSlash,
  Eye,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Loader2,
  ShieldCheck,
  X,
  XCircle,
} from 'lucide-react';
import {
  agentService,
  type ArtifactPreview,
  type ArtifactReport,
  type PlanRecord,
  type VerificationReport,
} from '../../services/agent.service';
import { labelFor, type RunViewState } from './useRun';
import type { Activity } from './recovery';
import { MilestoneGate } from './MilestoneGate';
import { RecoveryControls } from './RecoveryControls';
import styles from './RunView.module.css';

/**
 * One run, as it happens and afterwards.
 *
 * The order of the sections is the order somebody checks work in: what it was
 * asked, what it planned, what it did, what it produced, whether that holds up,
 * and only then the answer. Putting the answer last is deliberate — an answer
 * read before its provenance is an answer taken on trust, and the point of this
 * screen is that it need not be.
 *
 * Nothing here decides anything. Every judgement it shows — whether a file is
 * sound, whether a claim resolves to a source, why the run stopped — was made
 * in Rust against the file or the passage itself, not in the browser against
 * the model's description of them.
 */

const KIND_ICONS = {
  document: FileText,
  workbook: FileSpreadsheet,
  text: FileText,
} as const;

/** Bytes as somebody would say them. */
function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** A span of milliseconds the way a person would say it. */
function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms} ms`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(ms < 10_000 ? 1 : 0)} s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/** True when an activity row has any extra detail worth expanding into. */
function hasDetail(item: {
  inputSummary?: string;
  outputSummary?: string;
  errorMessage?: string;
  artifactPath?: string;
}): boolean {
  return Boolean(
    item.inputSummary || item.outputSummary || item.errorMessage || item.artifactPath,
  );
}

function Plan({ plan, stopped }: { plan: PlanRecord; stopped: string | null }) {
  const minutes = Math.round(plan.maxDurationSeconds / 60);
  return (
    <section className={styles.section}>
      <header className={styles.sectionHead}>
        <h2 className={styles.sectionTitle}>Plan</h2>
        <span className={styles.budget}>
          {plan.stepsTaken} of {plan.maxSteps} tool calls &middot; {minutes} min limit
        </span>
      </header>

      {/* Shown as what the run set out to do, with no per-step tick. One
        * planned step can take several tool calls, so nothing here knows that
        * a given step is finished — the artifacts and the check below are the
        * evidence for what was actually achieved, and a checklist ticking
        * itself off on call count would contradict them. */}
      <ol className={styles.steps}>
        {plan.steps.map(step => (
          <li key={step.ordinal} className={styles.step}>
            <span className={styles.stepMark} aria-hidden>
              {step.ordinal}
            </span>
            <span>{step.intent}</span>
          </li>
        ))}
      </ol>

      <p className={styles.tools}>Allowed to use: {plan.permittedTools.join(', ')}.</p>

      {/* Shown while it is still the freshest thing known — the final stop
        * reason arrives with the summary and is shown under the answer. */}
      {stopped && (
        <p className={styles.stopped} role="status">
          <CircleSlash size={14} />
          <span>{stopped}</span>
        </p>
      )}
    </section>
  );
}

function Verification({ report }: { report: VerificationReport }) {
  const standing = report.standing;
  const ready = standing.standing === 'ready';

  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>Checked</h2>

      <p className={ready ? styles.verdictReady : styles.verdictReview}>
        {ready ? <ShieldCheck size={15} /> : <AlertTriangle size={15} />}
        <span>
          {standing.standing === 'ready'
            ? 'Every claim in this answer resolves to a passage the task retrieved, and its figures match the recorded calculations.'
            : `This is a draft, not a finished answer. ${standing.blocking} thing(s) need checking before it is relied on, and ${standing.advisory} are worth a look.`}
        </span>
      </p>

      <p className={styles.counts}>
        {report.citationsResolved} citation(s) resolved &middot; {report.figuresChecked} figure(s)
        matched to a calculation
      </p>

      {report.findings.length > 0 && (
        <ul className={styles.findings}>
          {report.findings.map((finding, i) => (
            <li
              key={i}
              className={
                finding.severity === 'blocking'
                  ? `${styles.finding} ${styles.findingBlocking}`
                  : styles.finding
              }
            >
              <span className={styles.findingBadge}>
                {finding.severity === 'blocking' ? 'Needs checking' : 'Worth a look'}
              </span>
              <span className={styles.findingText}>{finding.detail}</span>
              {finding.excerpt && <code className={styles.excerpt}>{finding.excerpt}</code>}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function Artifacts({ artifacts, runId }: { artifacts: ArtifactReport[]; runId: string | null }) {
  const [problem, setProblem] = useState<string | null>(null);
  // Per-artifact preview state. Inline-rendering every preview eagerly would
  // re-render the whole list on every click; tracking the open one by name
  // keeps the panel scoped to the artifact the user is actually looking at.
  const [openName, setOpenName] = useState<string | null>(null);
  const [previews, setPreviews] = useState<Record<string, ArtifactPreview | 'loading' | 'error'>>({});

  const reveal = async (name: string) => {
    if (!runId) return;
    try {
      setProblem(null);
      await agentService.revealArtifact(runId, name);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    }
  };

  const togglePreview = async (name: string) => {
    if (!runId) return;
    if (openName === name) {
      setOpenName(null);
      return;
    }
    setOpenName(name);
    if (previews[name] && previews[name] !== 'error') {
      return;
    }
    setPreviews(prev => ({ ...prev, [name]: 'loading' }));
    try {
      const preview = await agentService.previewArtifact(runId, name);
      setPreviews(prev => ({ ...prev, [name]: preview }));
    } catch (error) {
      setPreviews(prev => ({ ...prev, [name]: 'error' }));
      setProblem(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <section className={styles.section}>
      <h2 className={styles.sectionTitle}>Produced</h2>
      <ul className={styles.artifacts}>
        {artifacts.map(artifact => {
          const Icon = KIND_ICONS[artifact.kind];
          const isOpen = openName === artifact.name;
          const preview = previews[artifact.name];
          return (
            <li key={artifact.path} className={styles.artifact}>
              <Icon size={17} className={styles.artifactIcon} />
              <div className={styles.artifactBody}>
                <div className={styles.artifactName}>
                  <strong>{artifact.name}</strong>
                  <span className={styles.artifactSize}>{size(artifact.bytes)}</span>
                  {/* Re-opened and checked by the backend, not inferred from
                    * the fact that a write returned without error. */}
                  <span className={artifact.sound ? styles.tagSound : styles.tagUnsound}>
                    {artifact.sound ? 'opens and checks out' : 'did not pass its check'}
                  </span>
                </div>
                <p className={styles.artifactDetail}>{artifact.detail}</p>
                {artifact.problems.length > 0 && (
                  <ul className={styles.problems}>
                    {artifact.problems.map((text, i) => (
                      <li key={i}>{text}</li>
                    ))}
                  </ul>
                )}
                {isOpen && (
                  <ArtifactPreviewPane preview={preview} name={artifact.name} />
                )}
              </div>
              <div className={styles.artifactActions}>
                <button
                  className={styles.revealBtn}
                  onClick={() => void togglePreview(artifact.name)}
                  aria-label={isOpen ? `Hide preview of ${artifact.name}` : `Preview ${artifact.name}`}
                  aria-expanded={isOpen}
                >
                  {isOpen ? <ChevronDown size={15} /> : <Eye size={15} />}
                </button>
                <button
                  className={styles.revealBtn}
                  onClick={() => void reveal(artifact.name)}
                  aria-label={`Show ${artifact.name} in the file manager`}
                >
                  <FolderOpen size={15} />
                </button>
              </div>
            </li>
          );
        })}
      </ul>
      {problem && (
        <p className={styles.stopped} role="alert">
          <AlertTriangle size={14} />
          <span>{problem}</span>
        </p>
      )}
    </section>
  );
}

/**
 * Inline preview pane for a single artifact.
 *
 * Lives inside the artifact's <li> so opening one does not reflow the rest of
 * the list. Renders whatever the backend returned: a code block for text,
 * markdown and sheet bodies; an <img> for raster formats; a "not previewable"
 * message when the format is something we deliberately don't try to render.
 */
function ArtifactPreviewPane({
  preview,
  name,
}: {
  preview: ArtifactPreview | 'loading' | 'error' | undefined;
  name: string;
}) {
  if (preview === undefined || preview === 'loading') {
    return (
      <div className={styles.previewPane} aria-busy="true">
        <Loader2 size={14} className={styles.spin} />
        <span>Reading {name}…</span>
      </div>
    );
  }
  if (preview === 'error') {
    return (
      <div className={styles.previewPane}>
        <AlertTriangle size={14} />
        <span>Could not load a preview of {name}.</span>
      </div>
    );
  }
  if (preview.kind === 'unsupported') {
    return (
      <div className={styles.previewPane}>
        <CircleSlash size={14} />
        <span>
          Preview not available for this format ({preview.mime || 'unknown'}). Use
          the folder button to open it in the file manager.
        </span>
      </div>
    );
  }
  if (preview.kind === 'image') {
    return (
      <div className={styles.previewPane}>
        <img
          className={styles.previewImage}
          src={preview.dataUrl}
          alt={`Preview of ${name}`}
        />
        {preview.truncated && (
          <p className={styles.previewNote}>Preview is truncated to fit.</p>
        )}
      </div>
    );
  }
  // text, markdown, docxBody, xlsxFirstSheet, pptxSlideList — all string bodies
  // with optional truncation.
  const mono =
    preview.kind === 'docxBody' ||
    preview.kind === 'xlsxFirstSheet' ||
    preview.kind === 'text';
  return (
    <div className={styles.previewPane}>
      <pre
        className={mono ? styles.previewPre : styles.previewMarkdown}
        data-truncated={preview.truncated || undefined}
      >
        {preview.content}
      </pre>
      {preview.truncated && (
        <p className={styles.previewNote}>
          Preview is truncated. Use the folder button for the full file.
        </p>
      )}
    </div>
  );
}

/**
 * One row in the Work section.
 *
 * Three things matter at a glance: what was called, whether it took, and how
 * long it took. Below that, the things that *might* matter on a closer look —
 * the redacted input, the redacted output, the file it produced, the reason
 * it failed. Open them on demand so the list does not turn into a wall of
 * text the moment a run gets long.
 */
function ActivityRow({ item }: { item: Activity }) {
  // Default open for failed rows: the operator's eye is already there, and
  // making them click to read the error is the wrong way to save space.
  const [open, setOpen] = useState(item.status === 'failed');
  const expandable = hasDetail(item);

  const StatusIcon = STATUS_ICONS[item.status];
  const duration =
    item.startedAt && item.endedAt
      ? formatDuration(Math.max(0, item.endedAt - item.startedAt))
      : item.status === 'running'
        ? 'in progress'
        : null;

  return (
    <li className={styles.activityItem}>
      <div className={styles.activityRow}>
        <StatusIcon
          size={14}
          className={styles[`statusIcon_${item.status}`]}
          aria-hidden
        />
        <span className={styles.activityLabel}>{labelFor(item.tool)}</span>
        <span className={styles.activityStatus}>{STATUS_TEXT[item.status]}</span>
        {duration && <span className={styles.activityDuration}>{duration}</span>}
        {expandable && (
          <button
            type="button"
            className={styles.activityExpand}
            onClick={() => setOpen(o => !o)}
            aria-expanded={open}
            aria-label={open ? 'Hide details' : 'Show details'}
          >
            {open ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
          </button>
        )}
      </div>
      {open && expandable && (
        <dl className={styles.activityDetail}>
          {item.inputSummary && (
            <>
              <dt>Asked</dt>
              <dd>{item.inputSummary}</dd>
            </>
          )}
          {item.outputSummary && (
            <>
              <dt>Returned</dt>
              <dd>{item.outputSummary}</dd>
            </>
          )}
          {item.artifactPath && (
            <>
              <dt>Produced</dt>
              <dd className={styles.activityPath}>{item.artifactPath}</dd>
            </>
          )}
          {item.errorMessage && (
            <>
              <dt>Failed because</dt>
              <dd className={styles.activityError}>{item.errorMessage}</dd>
            </>
          )}
        </dl>
      )}
    </li>
  );
}

const STATUS_TEXT: Record<Activity['status'], string> = {
  running: 'running',
  done: 'done',
  failed: 'failed',
  // Not a failure of the tool: the policy or a person said no, and the model
  // reads that and carries on.
  refused: 'not permitted',
  // The side effect had already happened, so it was not done a second time.
  // Worth showing: a reader counting document writes should not count this
  // one twice.
  replayed: 'already done — not repeated',
  // In flight when the process went away. Nobody can say whether it took
  // effect, so it was not tried again.
  unknown: 'interrupted — needs checking',
};

const STATUS_ICONS: Record<
  Activity['status'],
  React.ComponentType<{ size?: number; className?: string }>
> = {
  running: Loader2,
  done: CheckCircle2,
  failed: XCircle,
  refused: CircleSlash,
  replayed: Check,
  unknown: AlertTriangle,
};

interface Props {
  state: RunViewState;
  onAbort: () => void;
  /**
   * True once a stop has been sent and the run has not yet acknowledged it.
   *
   * The button needs this because sending an abort and the run actually
   * ending are two different moments, and only the second one is worth
   * telling somebody about. A Stop that goes back to reading "Stop" the
   * instant the IPC resolves reports that the request was delivered while
   * looking exactly like it reports that the run has ended.
   */
  stopping?: boolean;
  onNewTask: () => void;
  /**
   * Re-run the same prompt. Rendered as a "Rerun" button when the run
   * has finished. Optional — the workbench does not need it because its
   * "New task" button opens the composer; the demo page uses it so a
   * judge can re-trigger the same scenario with one click.
   */
  onRerun?: () => void;
}

export const RunView = ({ state, onAbort, onNewTask, onRerun, stopping = false }: Props) => {
  const running = state.phase === 'starting' || state.phase === 'running';
  const summary = state.summary;

  return (
    <article className={styles.run}>
      <header className={styles.head}>
        <p className={styles.prompt}>{state.prompt}</p>
        <div className={styles.headActions}>
          {!running && onRerun && (
            <button
              className={styles.newBtn}
              onClick={onRerun}
              aria-label="Run the same prompt again"
            >
              Rerun
            </button>
          )}
          {running ? (
            <button
              className={styles.stopBtn}
              onClick={onAbort}
              disabled={stopping}
              aria-busy={stopping || undefined}
            >
              <X size={14} />
              <span>{stopping ? 'Stopping…' : 'Stop'}</span>
            </button>
          ) : (
            <button className={styles.newBtn} onClick={onNewTask}>
              New task
            </button>
          )}
        </div>
      </header>

      {/* Said plainly rather than left for somebody to infer from a trace that
        * looks thin. A recovered run is a reconstruction from what was written
        * down as it happened, and it is missing whatever the live stream would
        * have shown in between — which is exactly the sort of gap a person
        * reading a trace will otherwise read as "the run did nothing". */}
      {state.recovered && (
        <p className={styles.note} role="status">
          {running
            ? 'Reattached to a task that was already running. What is shown below was read back from its record.'
            : 'Read back from this task’s record after the window reopened.'}
          {state.historyIncomplete &&
            ' Part of the record could not be read, so some steps may be missing from the list below.'}
        </p>
      )}

      {/* Which model took it and why. The routing decision is shown with the
        * work rather than buried in the audit log, because "why this model"
        * is the question asked when an answer looks wrong. */}
      {summary && (
        <p className={styles.routing}>
          <strong>{summary.routing.modelName}</strong> took this as{' '}
          {summary.routing.intent.toLowerCase()} work
          {summary.routing.usedFallback ? ', after the first choice did not fit' : ''} &middot;{' '}
          {summary.routing.reasons[0]} &middot;{' '}
          {summary.endpoint.runtime === 'llamaCpp' ? 'llama.cpp' : 'Python sidecar'} on{' '}
          {summary.endpoint.baseUrl}
        </p>
      )}

      {state.plan && <Plan plan={state.plan} stopped={state.stopped} />}
      {state.runId && <RecoveryControls key={state.runId} runId={state.runId} />}

      {/* The milestone gate: if the run paused at a checkpoint,
        * the gate renders here, before any further work, so the
        * reader sees the decision point at the top of the work
        * section rather than buried below it. */}
      {state.phase === 'awaiting_milestone' && state.milestone && state.runId && (
        <MilestoneGate
          runId={state.runId}
          gate={state.milestone}
          onAcknowledged={(_gate, _decision) => {
            // The live stream will emit `milestone_acknowledged`
            // and clear the gate through the reducer. Nothing
            // to do here: the button is busy until then, and a
            // failure surfaces as the gate's own error.
          }}
        />
      )}

      <section className={styles.section}>
        <header className={styles.sectionHead}>
          <h2 className={styles.sectionTitle}>Work</h2>
          {running && (
            <span className={styles.live}>
              <Loader2 size={13} className={styles.spin} />
              <span>{state.turns > 0 ? `turn ${state.turns + 1}` : 'starting'}</span>
            </span>
          )}
        </header>

        {state.activity.length === 0 ? (
          <p className={styles.quiet}>
            {running
              ? 'Reading the task. Nothing has been called yet.'
              : state.phase === 'failed'
                ? // Distinguished from a run that simply needed no tools: a run
                  // that never started did not answer anything, and saying it
                  // did would be the misleading half of an already bad outcome.
                  'This task stopped before it could call anything.'
                : 'This task was answered without calling a tool.'}
          </p>
        ) : (
          <ol className={styles.activity}>
            {state.activity.map(item => (
              <ActivityRow key={item.id} item={item} />
            ))}
          </ol>
        )}

        {state.compactions > 0 && (
          <p className={styles.note}>
            Earlier turns were replaced by a summary {state.compactions} time(s) so the task could
            continue. Anything answered after that point rests on the summary rather than on the
            turns themselves.
          </p>
        )}
      </section>

      {summary && summary.artifacts.length > 0 && (
        <Artifacts artifacts={summary.artifacts} runId={state.runId} />
      )}

      {summary?.verification && <Verification report={summary.verification} />}

      {summary && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Answer</h2>
          {summary.text.trim() ? (
            <div className={styles.answer}>{summary.text}</div>
          ) : (
            <p className={styles.quiet}>This task ended without an answer.</p>
          )}
          <p className={styles.note}>
            {summary.plan.stoppedBecause} {summary.turns} turn(s).
          </p>
        </section>
      )}

      {/* The one thing on this screen that asks a person to go and do
        * something in the world. A side effect was in flight when the process
        * went away, so the file it names may or may not exist — and repeating
        * it could do it twice. Named individually because "something was
        * interrupted" is not actionable and "note.docx may not have been
        * written" is. */}
      {state.unknownEffects.length > 0 && (
        <section className={styles.section}>
          <h2 className={styles.sectionTitle}>Needs checking</h2>
          <ul className={styles.problems}>
            {state.unknownEffects.map(effect => (
              <li key={effect.idempotencyKey}>
                <strong>{effect.target || effect.tool}</strong> — this action was interrupted
                while it was happening, so nobody can say whether it took effect. It has not
                been attempted again.
              </li>
            ))}
          </ul>
        </section>
      )}

      {state.phase === 'failed' && state.error && (
        <p className={styles.failure} role="alert">
          <AlertTriangle size={15} />
          <span>{state.error}</span>
        </p>
      )}
    </article>
  );
};
