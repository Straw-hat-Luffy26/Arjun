import React, { useState, useMemo } from 'react';
import {
  AlertTriangle,
  Check,
  CheckCircle2,
  CircleDashed,
  ChevronDown,
  CircleSlash,
  Copy,
  Download,
  Eye,
  FileSpreadsheet,
  FileText,
  FolderOpen,
  Loader2,
  Presentation,
  RotateCcw,
  X,
} from 'lucide-react';
import { save } from '@tauri-apps/plugin-dialog';
import { PdfView } from './PdfView';
import {
  agentService,
  messageStatus,
  MESSAGE_STATUS_LABELS,
  type MessageStatusKind,
  type ArtifactPreview,
  type ArtifactReport,
  type ChatMessage,
  type RunSummary,
} from '../../services/agent.service';
import { formatDuration, formatTokens } from './format';
import { collapseForDisplay } from '../../contexts/ConversationContext';
import { iconForTool, labelForTool } from '../../services/toolNames';
import { artifactPresentation, type ArtifactGlyph } from '../../services/artifactKind';
import { previewDisplay } from '../../services/artifactPreview';
import { InlineErrorBoundary } from '../ui';
import { ChatOrb } from './ChatOrb';
import { ThinkingTree, type ThinkingNode } from './ThinkingTree';
import { parseThinking } from './parseThinking';
import { RunProgressPanel } from './RunProgressPanel';
import type { ProgressStep } from './runProgress';
import { useTokenMetrics, type TokenMetrics } from './useTokenMetrics';
import { Markdown } from './Markdown';
import { ReasoningStream } from './ReasoningStream';
import type { LiveReasoning } from '../../contexts/ConversationContext';
import styles from './ChatSurface.module.css';

function size(bytes: number): string {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) return `${Math.round(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

/** One row per line of reasoning, for the collapsed timeline. */
function thinkingNodes(reasoning: string): ThinkingNode[] {
  if (!reasoning) return [];
  return reasoning
    .split('\n')
    .filter((line) => line.trim())
    .map((line, idx) => ({
      id: `r-${idx}`,
      label: line.trim().replace(/^[-·]\s*/, ''),
      status: 'done' as const,
      icon: 'none' as const,
    }));
}

/**
 * The full reading, for the hover the pill cannot fit.
 *
 * The pill shows output tokens and the rate because those are what change
 * while an answer is being written. The prompt size and whether the numbers
 * were reported or estimated matter too, but only to someone who has stopped
 * to look — so they live here rather than widening the line.
 */
function tokenTitle(metrics: TokenMetrics): string {
  const parts: string[] = [];
  if (metrics.tokensIn > 0) parts.push(`${metrics.tokensIn} tokens in`);
  parts.push(
    metrics.approx
      ? `about ${metrics.tokensOut} tokens out, estimated from the text — this model reported no usage`
      : `${metrics.tokensOut} tokens out`,
  );
  if (metrics.speed > 0) parts.push(`${metrics.speed} tokens/second`);
  return parts.join(' · ');
}

/**
 * The icon for one status.
 *
 * Shared by the status pill and the run footer so the two cannot drift. They
 * already had: the pill was rewritten to derive its state from what was
 * recorded, and the footer was left rendering a hard-coded shield and the word
 * "verified" for every turn that had a run id — including failures, runs a
 * person stopped, and runs the verifier never looked at. One component, two
 * rules, and only one of them true.
 */
function StatusIcon({ state, size }: { state: MessageStatusKind; size: number }) {
  if (state === 'thinking' || state === 'usingTool' || state === 'composing') {
    return <Loader2 size={size} className={styles.spin} />;
  }
  // The tick is reserved for the one state that earned it: the verifier ran
  // and every claim resolved.
  if (state === 'verified') return <CheckCircle2 size={size} />;
  if (state === 'failed') return <X size={size} />;
  // Finished, but not certified: needs review, unverified, stopped, or
  // completed with nothing to check. None of those is a failure and none of
  // them is a pass.
  return <CircleDashed size={size} />;
}

function StatusPill({
  state,
  elapsedText,
  runningTools,
  metrics,
}: {
  state: MessageStatusKind;
  elapsedText: string | null;
  runningTools: number;
  metrics: TokenMetrics;
}) {
  const labels: Record<MessageStatusKind, string> = {
    ...MESSAGE_STATUS_LABELS,
    // The only label that depends on something outside the status itself.
    usingTool: runningTools > 1 ? `Using ${runningTools} tools…` : 'Using a tool…',
  };

  return (
    <span className={styles.statusPill} data-state={state}>
      <StatusIcon state={state} size={11} />
      <span>{labels[state]}</span>
      {elapsedText && <span className={styles.statusPillSub}>· {elapsedText}</span>}
      {(state === 'composing' ||
        state === 'verified' ||
        state === 'needsReview' ||
        state === 'unverified') &&
        metrics.tokensOut > 0 && (
        <span className={styles.statusPillSub} title={tokenTitle(metrics)}>
          · {metrics.approx ? '~' : ''}
          {formatTokens(metrics.tokensOut)} tok
          {metrics.speed > 0 && ` · ${metrics.speed} tok/s`}
        </span>
      )}
    </span>
  );
}

interface AssistantMessageCellProps {
  message: ChatMessage;
  liveContent?: string;
  isLive?: boolean;
  activity?: {
    id: string;
    tool: string;
    status: 'running' | 'done' | 'failed' | 'refused' | 'replayed' | 'unknown';
    startedAt?: number;
    endedAt?: number;
    inputSummary?: string;
    outputSummary?: string;
    artifactPath?: string;
    errorMessage?: string;
  }[];
  runSummary?: RunSummary | null;
  /**
   * The files this message's run produced.
   *
   * Separate from `runSummary` deliberately: that arrives only while this
   * message's inspector is open, and a deliverable must not depend on the
   * inspector being open to be visible.
   */
  artifacts?: ArtifactReport[];
  /**
   * What this turn has been doing, newest last.
   *
   * Keyed to this message by the reducer, never matched by position in the
   * log: a list found by index is how one turn's progress ends up under
   * another turn's answer.
   */
  progress?: ProgressStep[];
  /**
   * The reasoning this turn has produced so far.
   *
   * Live only, and absent on every finished message: it is held in the
   * reducer for the life of the run and never persisted, so a reopened
   * conversation shows the answer with no thinking behind it.
   */
  reasoning?: LiveReasoning;
  /** Only the newest assistant cell draws the orb. */
  showAvatar?: boolean;
  onOpenInspector?: (runId: string) => void;
  onRetry?: () => void;
  /**
   * Raised when a widget in this message calls `sendPrompt`.
   *
   * Absent while a turn is in flight, which is what stops a widget from
   * queueing turns of its own: the capability exists only when the person
   * could have typed the same thing themselves.
   */
  onPrompt?: (text: string) => void;
  composerDisabled?: boolean;
}

export function AssistantMessageCell({
  message,
  liveContent,
  isLive,
  activity,
  runSummary,
  artifacts,
  progress,
  // Renamed on the way in. `reasoning` is already spoken for in this
  // component: `parseThinking` returns the inline <think> block under that
  // name, and the two are different things — one was streamed on its own
  // channel, the other was dug out of the answer after the fact.
  reasoning: liveReasoning,
  showAvatar,
  onOpenInspector,
  onRetry,
  onPrompt,
}: AssistantMessageCellProps) {
  // What was stored: the model's exact words, in the order it produced them.
  //
  // Repetition is collapsed for *display* only, and only when the answer has no
  // fenced code in it. It used to be collapsed inside the streaming reducer,
  // which meant the edited text was what got persisted, sent as `finalContent`,
  // resolved against by the verifier and written into the audit record — a
  // display convenience editing the evidence. `stored` is what everything else
  // in the product sees; `content` is what this cell draws.
  const stored = isLive ? (liveContent ?? message.content) : message.content;
  const display = useMemo(() => collapseForDisplay(stored), [stored]);
  const content = display.text;
  const isStreaming = isLive === true || message.status === 'streaming';
  const isFailed = message.status === 'failed';
  const isDone = !isStreaming && !isFailed;

  const elapsedMs = message.elapsedMs ?? null;
  const elapsedText = elapsedMs !== null ? formatDuration(elapsedMs) : null;
  const modelName = message.modelName ?? runSummary?.routing.modelName ?? 'model';
  const modelRole = message.modelRole ?? runSummary?.routing.role ?? null;
  const usedFallback = message.usedFallback ?? runSummary?.routing.usedFallback ?? false;

  const runningTools = activity?.filter(a => a.status === 'running').length ?? 0;
  const toolsTotal = activity?.length ?? 0;

  // What this turn actually is, from what was actually recorded.
  //
  // This used to read `isFailed ? 'failed' : isDone ? 'verified' : ...`, so
  // "not streaming and not failed" was rendered as **Verified**, with a green
  // tick, for every turn — including ones the verifier never looked at, ones
  // it found blocking problems in, and ones a person stopped part way through.
  // The strongest claim the product can make was the one it made by default.
  const status = messageStatus({
    isStreaming,
    contentLength: content.length,
    runningTools,
    // Persisted per message, so a cell rendered from disk long after the run's
    // events have gone still knows how it ended and what checked it.
    outcome: message.outcome ?? (isFailed ? 'failed' : null),
    verification:
      message.verification ??
      (runSummary?.verification
        ? runSummary.verification.standing.standing === 'ready'
          ? 'ready'
          : 'needsReview'
        : null),
  });

  const [copied, setCopied] = useState(false);

  const metrics = useTokenMetrics(
    isStreaming,
    content.length,
    message.tokensIn,
    message.tokensOut,
    elapsedMs,
  );

  // The model's reasoning, and the answer with the reasoning taken out.
  //
  // `parseThinking` reports an *open* block as well as a closed one, which is
  // what makes a reasoning pass visible while it happens instead of only after
  // it ends. See the note in that file: matching only closed pairs is what
  // made a reasoning model look like it had frozen.
  const parsed = useMemo(() => parseThinking(content), [content]);
  const { reasoning, answer } = parsed;
  const nodes = useMemo(() => thinkingNodes(reasoning), [reasoning]);
  // Keyed on whether a tag was seen, not on whether the thought is non-empty:
  // the instant `<think>` arrives and before a single character follows it,
  // the raw buffer already holds a tag that Markdown would swallow whole.
  const displayContent = parsed.sawTag ? answer : content;

  /**
   * The reasoning to show, from whichever channel carried it.
   *
   * Two channels exist because two kinds of server exist. One splits the
   * reasoning into its own field, which arrives live on `model_thinking` and
   * is what `liveReasoning` holds. The other leaves it inline in the content
   * inside `<think>` tags, which `parseThinking` recovers from the answer.
   *
   * They are merged here rather than rendered in two places. The inline block
   * used to become rows in the tool timeline, so the same product had three
   * surfaces that could be called thinking: this panel, that timeline, and the
   * stage list titled "Thinking". One concept, one surface.
   *
   * `undefined` when there is neither, which is the correct state for a model
   * with no reasoning switch — the panel then renders nothing at all rather
   * than an empty box implying a thought nobody can see.
   */
  const thinking: LiveReasoning | undefined = useMemo(() => {
    if (liveReasoning && liveReasoning.text.trim() !== '') return liveReasoning;
    if (reasoning.trim() !== '') return { text: reasoning, trimmed: false };
    return undefined;
  }, [liveReasoning, reasoning]);

  // The tool record for this turn, and only that.
  //
  // The model's own reasoning steps used to be prepended here under a
  // `reasoning` group key. They now go to the Thinking panel, which shows the
  // prose rather than a bulleted summary of it.
  const timelineNodes: ThinkingNode[] = useMemo(
    () => (activity ?? []).map(toolNode),
    [activity],
  );

  return (
    <div className={styles.assistantRow}>
      {/* Left column: the orb on the newest cell only, and an empty
        * slot of the same width on the rest so every message in the log
        * stays on one left edge. */}
      <div className={styles.assistantAvatar}>
        {showAvatar && <ChatOrb active={isStreaming} size={36} />}
      </div>

      {/* RIGHT: Meta + Content */}
      <div className={styles.assistantCol}>
        <div className={styles.assistantMeta}>
          <StatusPill
            state={status}
            elapsedText={elapsedText}
            runningTools={runningTools}
            metrics={metrics}
          />
          <span className={styles.assistantMetaSep}>·</span>
          <span className={styles.assistantModel}>
            {modelName}
            {modelRole && <span className={styles.assistantModelRole}> · {modelRole}</span>}
            {usedFallback && (
              <span className={styles.assistantFallback}>fallback</span>
            )}
          </span>
        </div>

        <div className={styles.assistantBody}>
          {isFailed ? (
            <p className={styles.errorLine}>
              <AlertTriangle size={13} />
              <span>{message.error ?? 'The run did not finish cleanly.'}</span>
            </p>
          ) : (
            <>
              {/* What the turn is doing, above the answer. Rendered before
                * the tool timeline because it covers the earlier interval:
                * the reading, routing and loading that happen before the
                * model is asked anything. */}
              {progress && progress.length > 0 && (
                <RunProgressPanel
                  steps={progress}
                  isLive={isStreaming}
                  hasAnswer={displayContent.length > 0}
                />
              )}

              {/* The model thinking, while it thinks. Above the timeline
                * and the answer because it covers the interval between
                * them: the request has gone, the answer has not started,
                * and this is the only thing happening. It closes itself
                * when the answer begins. */}
              <ReasoningStream
                reasoning={thinking}
                isLive={isStreaming}
                hasAnswer={displayContent.length > 0}
              />

              {/* The record of the turn. It stays after the run ends:
                * this is what happened, not a progress spinner. */}
              {timelineNodes.length > 0 && (
                <ThinkingTree
                  nodes={timelineNodes}
                  isLive={isStreaming}
                  summary={elapsedText}
                />
              )}

              {/* FIX 2 & 6: Token-by-token streaming with markdown rendering. */}
              <div
                className={styles.assistantText}
                aria-live={isStreaming ? 'polite' : undefined}
                aria-busy={isStreaming || undefined}
              >
                {displayContent ? (
                  <Markdown content={displayContent} onPrompt={onPrompt} />
                ) : isStreaming ? (
                  <span className={styles.assistantPlaceholder} aria-hidden="true" />
                ) : null}
                {isStreaming && displayContent && (
                  <span className={styles.caret} aria-hidden="true" />
                )}
              </div>

              {/* Per-message actions sit after the text in the DOM so a
                * screen reader reaches the answer before the controls. */}
              {isDone && displayContent.length > 0 && (
                <div className={styles.messageActions}>
                  <button
                    type="button"
                    className={styles.messageAction}
                    onClick={() => {
                      void navigator.clipboard
                        .writeText(displayContent)
                        .then(() => {
                          setCopied(true);
                          window.setTimeout(() => setCopied(false), 1500);
                        })
                        .catch(() => setCopied(false));
                    }}
                    aria-label={copied ? 'Copied to clipboard' : 'Copy this answer'}
                  >
                    {copied ? <Check size={12} /> : <Copy size={12} />}
                    <span>{copied ? 'Copied' : 'Copy'}</span>
                  </button>
                </div>
              )}
            </>
          )}

          {isFailed && onRetry && (
            <button className={styles.retryBtn} onClick={onRetry} type="button">
              <RotateCcw size={12} /> Retry
            </button>
          )}

          {/*
            Drawn from `artifacts`, which `ChatSurface` fetches for every run,
            rather than from `runSummary`, which it supplies only while this
            message's inspector is open. Conditioned on the summary, a produced
            file appeared only after a click nothing prompted — the model would
            say "saved as sum-of-2-numbers.pdf" and the chat would show no file.

            The id comes from `message.runId` — the same field the "View
            details" button below uses — for the same reason: the summary is
            usually null here, so reading the id off it would leave the list
            undrawable exactly when it is needed.
          */}
          {message.runId && artifacts && artifacts.length > 0 && (
            <ArtifactList runId={message.runId} artifacts={artifacts} />
          )}

          {message.runId && onOpenInspector && (
            <div className={styles.assistantFooter}>
              <button
                className={styles.detailsBtn}
                onClick={() => onOpenInspector(message.runId!)}
                type="button"
              >
                View details ?
              </button>
              <span className={styles.assistantFooterMeta}>
                {toolsTotal > 0 && <>{toolsTotal} tool{toolsTotal === 1 ? '' : 's'} · </>}
                {runSummary && <>{runSummary.plan.steps.length} step{runSummary.plan.steps.length === 1 ? '' : 's'} · </>}
                <StatusIcon state={status} size={10} />
                <span>{MESSAGE_STATUS_LABELS[status].toLowerCase()}</span>
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

interface ActivityEntry {
  id: string;
  tool: string;
  status: 'running' | 'done' | 'failed' | 'refused' | 'replayed' | 'unknown';
  startedAt?: number;
  endedAt?: number;
  inputSummary?: string;
  outputSummary?: string;
  artifactPath?: string;
  errorMessage?: string;
}

const STATUS_LABEL: Record<ActivityEntry['status'], string> = {
  running: 'running',
  done: 'done',
  failed: 'failed',
  refused: 'not permitted',
  replayed: 'already done',
  unknown: 'interrupted',
};

// The labels and the icon rule both live in `services/toolNames.ts`. They used
// to be a second copy of the table in `useRun.ts`, keyed on the pre-namespace
// spelling, so a live event carrying the current name fell through to the raw
// string. See that module.
function toolLabel(tool: string) {
  return labelForTool(tool);
}

function toolIcon(tool: string): ThinkingNode['icon'] {
  return iconForTool(tool);
}

/**
 * What the chevron reveals for a tool row: what the tool was asked,
 * what came back, what it produced, why it failed. Anything the run
 * did not record is simply left out rather than shown as an empty
 * field.
 */
function toolDetail(item: ActivityEntry): string | undefined {
  const parts: string[] = [];
  if (item.inputSummary) parts.push(`Asked: ${item.inputSummary}`);
  if (item.outputSummary) parts.push(`Returned: ${item.outputSummary}`);
  if (item.artifactPath) parts.push(`Produced: ${item.artifactPath}`);
  if (item.errorMessage) parts.push(`Failed because: ${item.errorMessage}`);
  if (parts.length === 0 && item.status !== 'done' && item.status !== 'running') {
    parts.push(STATUS_LABEL[item.status]);
  }
  return parts.length > 0 ? parts.join('\n') : undefined;
}

/** One backend activity record as a timeline row. */
function toolNode(item: ActivityEntry): ThinkingNode {
  const duration =
    item.endedAt && item.startedAt
      ? formatDuration(Math.max(0, item.endedAt - item.startedAt))
      : undefined;
  return {
    id: item.id,
    label: toolLabel(item.tool),
    group: 'tools',
    icon: toolIcon(item.tool),
    status:
      item.status === 'running'
        ? 'running'
        : item.status === 'done' || item.status === 'replayed'
          ? 'done'
          : 'failed',
    meta: item.status === 'running' ? 'in progress' : duration,
    detail: toolDetail(item),
  };
}

/**
 * One icon per glyph. Keyed on [`ArtifactGlyph`] — a set this file controls —
 * rather than on the artifact kind, which arrives from Rust as JSON and once
 * carried a fourth value this table had no entry for. `artifactPresentation`
 * maps any kind onto a glyph, so this lookup cannot miss.
 */
const GLYPH_ICONS: Record<ArtifactGlyph, typeof FileText> = {
  document: FileText,
  workbook: FileSpreadsheet,
  deck: Presentation,
  file: FileText,
};

function ArtifactList({ runId, artifacts }: { runId: string; artifacts: ArtifactReport[] }) {
  return (
    <ul className={styles.artifactList}>
      {artifacts.map(artifact => (
        // Per row, so one file the surface cannot draw costs that row and not
        // the conversation around it.
        <InlineErrorBoundary key={artifact.path} label={artifact.name}>
          <ArtifactRow runId={runId} artifact={artifact} />
        </InlineErrorBoundary>
      ))}
    </ul>
  );
}

function ArtifactRow({ runId, artifact }: { runId: string; artifact: ArtifactReport }) {
  const [open, setOpen] = useState(false);
  const [preview, setPreview] = useState<ArtifactPreview | 'loading' | 'error' | undefined>(undefined);
  const [problem, setProblem] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState<string | null>(null);
  const presentation = artifactPresentation(artifact.kind);
  const Icon = GLYPH_ICONS[presentation.glyph];

  const reveal = async () => {
    try {
      await agentService.revealArtifact(runId, artifact.name);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    }
  };

  /**
   * Saves a copy where the person chooses.
   *
   * The platform's own dialog decides the destination. This application never
   * enumerates the filesystem to offer a list, and never picks a folder on the
   * person's behalf — a produced file belongs to them, and where it goes is
   * their decision to make in the picker their operating system already gives
   * them for exactly this.
   *
   * A cancelled dialog returns null, which is not a failure and says nothing.
   */
  const saveAs = async () => {
    setProblem(null);
    try {
      const destination = await save({
        defaultPath: artifact.name,
        title: `Save ${artifact.name}`,
      });
      if (!destination) return;

      setSaving(true);
      const bytes = await agentService.exportArtifact(runId, artifact.name, destination);
      // Named rather than a bare tick: "saved" with no destination leaves the
      // person hunting for a file they were just told exists somewhere.
      setSaved(`Saved ${size(bytes)} to ${destination}`);
    } catch (error) {
      setProblem(error instanceof Error ? error.message : String(error));
    } finally {
      setSaving(false);
    }
  };

  const togglePreview = async () => {
    if (open) {
      setOpen(false);
      return;
    }
    setOpen(true);
    if (preview && preview !== 'error') return;
    setPreview('loading');
    try {
      const p = await agentService.previewArtifact(runId, artifact.name);
      setPreview(p);
    } catch {
      setPreview('error');
    }
  };

  return (
    <li className={styles.artifactRow}>
      <div className={styles.artifactRowMain}>
        <Icon size={13} className={styles.artifactIcon} aria-hidden="true" />
        <button type="button" className={styles.artifactNameBtn} onClick={togglePreview} title={`${presentation.label} — ${open ? 'hide preview' : 'preview'}`}>
          <span className={styles.artifactName}>{artifact.name}</span>
        </button>
        <span className={styles.artifactSize}>{size(artifact.bytes)}</span>
        <span className={artifact.sound ? styles.tagSound : styles.tagUnsound}>
          {artifact.sound ? 'opens and checks out' : 'did not pass its check'}
        </span>
        <div className={styles.artifactActions}>
          <button type="button" className={styles.iconBtn} onClick={togglePreview} aria-label={open ? 'Hide preview' : 'Preview'} title={open ? 'Hide preview' : 'Preview'}>
            {open ? <ChevronDown size={12} /> : <Eye size={12} />}
          </button>
          <button
            type="button"
            className={styles.iconBtn}
            onClick={saveAs}
            disabled={saving}
            aria-label={`Save ${artifact.name} as…`}
            title="Save as…"
          >
            {saving ? <Loader2 size={12} className={styles.spin} /> : <Download size={12} />}
          </button>
          <button type="button" className={styles.iconBtn} onClick={reveal} aria-label="Show in file manager" title="Show in file manager">
            <FolderOpen size={12} />
          </button>
        </div>
      </div>
      {open && (
        <ArtifactPreviewPane
          preview={preview}
          name={artifact.name}
          runId={runId}
        />
      )}
      {saved && (
        <p className={styles.savedLine} role="status">
          <Check size={12} />
          <span>{saved}</span>
        </p>
      )}
      {problem && (
        <p className={styles.errorLine} role="alert">
          <AlertTriangle size={12} />
          <span>{problem}</span>
        </p>
      )}
    </li>
  );
}

/**
 * A PDF, fetched as bytes and rendered.
 *
 * Its own component because it is the one preview whose payload does not come
 * from `artifact_preview`. That command answers a PDF with no body on purpose
 * — there is no PDF reader in the Rust process — so the pane asks for the file
 * itself instead, and `PdfView` draws it.
 *
 * The fetch is deferred to the moment the pane opens rather than done with the
 * preview, so a message listing four PDFs does not pull four documents over
 * IPC for previews nobody expanded.
 */
function PdfPane({ runId, name }: { runId: string; name: string }) {
  const [state, setState] = useState<
    { status: 'loading' } | { status: 'ready'; base64: string } | { status: 'failed'; reason: string }
  >({ status: 'loading' });

  React.useEffect(() => {
    let live = true;
    setState({ status: 'loading' });

    agentService
      .artifactBytes(runId, name)
      .then(bytes => {
        if (live) setState({ status: 'ready', base64: bytes.base64 });
      })
      .catch(error => {
        // The message carries the real reason — a file over the transfer cap
        // reports its actual size and what to do instead — so it is shown
        // rather than replaced with a generic line.
        if (live) {
          setState({
            status: 'failed',
            reason: error instanceof Error ? error.message : String(error),
          });
        }
      });

    return () => {
      live = false;
    };
  }, [runId, name]);

  if (state.status === 'loading') {
    return (
      <div className={styles.previewPane} aria-busy="true">
        <Loader2 size={12} className={styles.spin} />
        <span>Reading {name}…</span>
      </div>
    );
  }
  if (state.status === 'failed') {
    return (
      <div className={styles.previewPane}>
        <CircleSlash size={12} />
        <span>{state.reason}</span>
      </div>
    );
  }
  return <PdfView base64={state.base64} name={name} />;
}

function ArtifactPreviewPane({
  preview,
  name,
  runId,
}: {
  preview: ArtifactPreview | 'loading' | 'error' | undefined;
  name: string;
  runId: string;
}) {
  if (preview === undefined || preview === 'loading') {
    return (
      <div className={styles.previewPane} aria-busy="true">
        <Loader2 size={12} className={styles.spin} />
        <span>Reading {name}…</span>
      </div>
    );
  }
  if (preview === 'error') {
    return (
      <div className={styles.previewPane}>
        <AlertTriangle size={12} />
        <span>Could not load a preview of {name}.</span>
      </div>
    );
  }

  // A PDF is the exception to the line below: `previewDisplay` can only decide
  // what to do with the payload it is given, and Rust gives a PDF none. The
  // file itself is fetched instead and rendered by pdf.js.
  if (preview.kind === 'pdf') {
    return <PdfPane runId={runId} name={name} />;
  }

  // What to draw is decided in `artifactPreview.ts`, against the shape Rust
  // actually sends. This function only draws it.
  const display = previewDisplay(preview);

  if (display.layout === 'notice') {
    return (
      <div className={styles.previewPane}>
        <CircleSlash size={12} />
        <span>{display.message}</span>
      </div>
    );
  }

  if (display.layout === 'image') {
    return (
      <div className={styles.previewPane}>
        <img className={styles.previewImage} src={display.src} alt={`Preview of ${name}`} />
        {display.note && <p className={styles.previewNote}>{display.note}</p>}
      </div>
    );
  }

  return (
    <div className={styles.previewPane}>
      <pre
        className={display.mono ? styles.previewPre : styles.previewMarkdown}
        data-truncated={preview.truncated || undefined}
      >
        {display.body}
      </pre>
      {display.note && <p className={styles.previewNote}>{display.note}</p>}
    </div>
  );
}
