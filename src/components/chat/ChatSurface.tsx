import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { ArrowDown } from 'lucide-react';
import { useConversation } from '../run/useConversation';
import { useActiveRun } from '../../contexts/ActiveRunContext';
import { ChatComposer } from './ChatComposer';
import {
  agentService,
  isTerminal,
  listenAttachmentProgress,
  describeAttachmentProgress,
  describeAttachmentKind,
  type AttachmentProgress,
  type ComposerAttachment,
} from '../../services/agent.service';

/**
 * How long a stop waits for the run to say it ended.
 *
 * Long enough to cover a run finishing the tool call it was in the middle of,
 * short enough that a person is not left watching a disabled button. When it
 * expires nothing is assumed either way — the message says the stop was sent
 * and that this screen cannot confirm the run ended.
 */
const STOP_ACKNOWLEDGEMENT_TIMEOUT_MS = 15_000;
import {
  applyAttachmentOcrEvent,
  listenAttachmentOcr,
  type OcrPageRead,
} from '../../services/ocr.service';
import { OcrReadout } from './OcrReadout';
import { useOcrPreference } from './useOcrPreference';
import { AssistantMessageCell } from './AssistantMessageCell';
import type { ProgressStep } from './runProgress';
import type { LiveReasoning } from '../../contexts/ConversationContext';
import { RunView } from '../run/RunView';
import {
  useAdoptedRun,
  useContextLedger,
  useConversationActivity,
  useTaskRecord,
} from '../run/runAdopt';
import { ChatHeader } from './ChatHeader';
import { TaskPanel } from './TaskPanel';
import {
  type ChatMessage,
  type Classification,
  type RunSummary,
} from '../../services/agent.service';
import type { Activity, RunViewState } from '../run/recovery';
import styles from './ChatSurface.module.css';

/**
 * The chat surface: message log + sticky composer.
 *
 * Owns no business state — all data flows through `useConversation`. The
 * per-run state used by the inspector and by the assistant cell's
 * tool-activity list is read through `useAdoptedRun`, a thin hook over
 * the same snapshot/event-replay machinery the workbench uses.
 *
 * Conversation history lives in the ARJUN dropdown in the application
 * shell, not in a sidebar inside this surface, so a chat is focused on
 * the current conversation.
 */
/** How close to the bottom still counts as "following the stream". */
const NEAR_BOTTOM_PX = 100;

/** A turn waiting for the current run to finish, with its own attachments. */
interface QueuedTurn {
  text: string;
  attachments: ComposerAttachment[];
}

export interface ChatSurfaceProps {
  /** Optional: a system prompt to prepend to every user message. */
  scenarioInstructions?: string;
  /** Optional: a custom classification to apply to every user message. */
  classification?: Classification;
  /** Whether to show the conversation sidebar. Reserved for future
   *  layouts; the current UI does not render a sidebar. */
  showSidebar?: boolean;
}

/**
 * Publishes the chat's run to the one active-run source.
 *
 * The chat issues its own run id — it has to, because it reserves the assistant
 * cell before the backend replies — so the shared provider does not learn about
 * a chat-launched run on its own. Handing it over here is what makes the SIH
 * dashboard's routing, plan, activity, verification and security panes follow
 * the run the person is actually watching, rather than sitting idle for the
 * whole of it.
 */
function useShareActiveRun(activeRunId: string | null): void {
  const { follow } = useActiveRun();
  const shared = useRef<string | null>(null);
  useEffect(() => {
    if (!activeRunId || shared.current === activeRunId) return;
    shared.current = activeRunId;
    void follow(activeRunId).catch(() => {
      // The run may not have a durable record yet -- it was started a moment
      // ago. The dashboard shows what it has; it does not invent the rest.
      shared.current = null;
    });
  }, [activeRunId, follow]);
}

export function ChatSurface({
  classification,
  showSidebar: _showSidebar = false,
}: ChatSurfaceProps) {
  const {
    conversation,
    isStreaming,
    activeMessageId,
    activeRunId,
    streamingContents,
    progressByMessage,
    reasoningByMessage,
    send,
    replay,
  } = useConversation();

  // Every surface follows the run the chat started, not one of its own.
  useShareActiveRun(activeRunId);

  const [inspectorRunId, setInspectorRunId] = useState<string | null>(null);

  // Messages typed while a run is in flight. They are held here and
  // sent one at a time, in order, as soon as the surface goes idle —
  // the user never has to wait for a turn to finish before asking the
  // next thing. `flushing` covers the gap between calling `send` and
  // `isStreaming` catching up.
  // A queued turn carries its own attachments. Holding only the text would
  // let a turn be sent later with whatever was attached at flush time, which
  // is how one message ends up answering about another's document.
  const [queued, setQueued] = useState<QueuedTurn[]>([]);
  // What the backend is doing with an attachment right now. Cleared when the
  // run ends, so the line disappears rather than lingering as stale status.
  const [reading, setReading] = useState<AttachmentProgress | null>(null);
  // The OCR model's own output for this turn's attachments. Cleared when the
  // next turn starts rather than when this one ends, so the evidence for the
  // answer on screen stays on screen next to it.
  const [ocrPages, setOcrPages] = useState<OcrPageRead[]>([]);
  const ocrPreference = useOcrPreference();

  useEffect(() => {
    const sub = listenAttachmentProgress(p => {
      // Scoped to the turn that asked for the read. The channel is
      // application-wide, and a second window reading a document would
      // otherwise put its page counter under this window's composer.
      if (p.messageId && activeMessageId && p.messageId !== activeMessageId) {
        return;
      }
      setReading(p.phase === 'done' ? null : p);
    });
    return () => {
      void sub.then(un => un());
    };
  }, [activeMessageId]);

  useEffect(() => {
    const sub = listenAttachmentOcr(event =>
      setOcrPages(prev => applyAttachmentOcrEvent(prev, event)),
    );
    return () => {
      void sub.then(un => un());
    };
  }, []);

  useEffect(() => {
    if (!isStreaming) setReading(null);
  }, [isStreaming]);
  const [flushing, setFlushing] = useState(false);
  /**
   * The run a stop has been sent for, until that run acknowledges it.
   *
   * Held as an id rather than a boolean so switching the inspector to a
   * different run cannot leave a second run's button reading "Stopping".
   */
  const [stoppingRunId, setStoppingRunId] = useState<string | null>(null);
  /** Said out loud when a stop was sent and nothing came back. */
  const [stopProblem, setStopProblem] = useState<string | null>(null);

  const adopted = useAdoptedRun(inspectorRunId);
  const taskSummary = useTaskRecord(inspectorRunId);

  /**
   * Stop the run the inspector is showing.
   *
   * This used to be `() => setInspectorRunId(null)` — the panel closed, the
   * button had said Stop, and the run carried on using the machine with
   * nothing on screen to say so. Closing a window is not stopping work, and
   * the two had become the same gesture.
   *
   * Sending the abort is only half of it. `agent_abort_run` resolving means
   * the backend accepted the request, not that the run has ended: a run
   * mid-tool-call finishes what it is doing first. So the button stays in
   * its stopping state until the run's *own* events say it reached a
   * terminal state — see the effect below, which is the acknowledgement.
   */
  const stopInspectedRun = useCallback(async () => {
    const runId = inspectorRunId;
    if (!runId || stoppingRunId) return;
    setStopProblem(null);
    setStoppingRunId(runId);
    try {
      const outcome = await agentService.abort(runId);
      if (!outcome.requested) {
        // The run finished between the render and the click. An ordinary
        // race, and the record already says how it ended, so there is
        // nothing to report and nothing left to wait for.
        setStoppingRunId(current => (current === runId ? null : current));
      }
    } catch (error) {
      setStopProblem(
        `The stop could not be sent: ${
          error instanceof Error ? error.message : String(error)
        }. The run is still going.`,
      );
      setStoppingRunId(current => (current === runId ? null : current));
    }
  }, [inspectorRunId, stoppingRunId]);

  /**
   * The acknowledgement: the run itself reporting that it has ended.
   *
   * Keyed on the run's terminal *state* rather than on its phase. A run
   * paused at a milestone gate is not "running" either, and treating that as
   * a stop would report the work as over while it waits for a person.
   */
  const stoppedState = adopted?.view.state ?? null;
  useEffect(() => {
    if (!stoppingRunId) return;
    if (stoppedState && isTerminal(stoppedState)) setStoppingRunId(null);
  }, [stoppingRunId, stoppedState]);

  /**
   * A stop that is never acknowledged is reported, not waited on forever.
   *
   * The button would otherwise sit disabled and reading "Stopping" with no
   * way back, which tells a person less than saying plainly that the request
   * went out and nothing came back.
   */
  useEffect(() => {
    if (!stoppingRunId) return;
    const timer = window.setTimeout(() => {
      setStopProblem(
        'Stop was sent, but the run has not reported that it ended. It may still be ' +
          'finishing the step it was on. Nothing here can confirm it has stopped.',
      );
      setStoppingRunId(null);
    }, STOP_ACKNOWLEDGEMENT_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [stoppingRunId]);

  // Clear a stale message when the inspector moves to a different run.
  useEffect(() => {
    setStopProblem(null);
    setStoppingRunId(null);
  }, [inspectorRunId]);

  // runsByMessageId: a stable map of `messageId → runId` for the
  // current conversation, used to find which run a message belongs to
  // when the user clicks "View details" or the retry button.
  const runsByMessageId = useMemo(() => {
    const map = new Map<string, string>();
    if (conversation) {
      for (const r of conversation.runs) {
        map.set(r.messageId, r.runId);
      }
    }
    return map;
  }, [conversation]);

  // The assistant cell that is currently streaming: the one matching
  // the active message id, or the run that is still live.
  const streamingMessageId = useMemo(() => {
    if (!conversation) return null;
    if (activeMessageId) return activeMessageId;
    for (const r of conversation.runs) {
      if (r.live) return r.messageId;
    }
    return null;
  }, [conversation, activeMessageId]);

  // The most recent run in the conversation. Used to feed the chat
  // header's context chip; the inspector, if open, takes precedence.
  const latestRunId = useMemo(() => {
    if (!conversation || conversation.runs.length === 0) return null;
    return conversation.runs[conversation.runs.length - 1].runId;
  }, [conversation]);
  const headerContext = useContextLedger(
    inspectorRunId ?? latestRunId,
  );

  // Tool activity for the whole conversation. The live run streams;
  // finished runs are read once from their snapshots.
  const runIds = useMemo(
    () => (conversation ? conversation.runs.map(r => r.runId) : []),
    [conversation],
  );
  const liveRunId = useMemo(() => {
    if (activeRunId) return activeRunId;
    if (!conversation) return null;
    for (const r of conversation.runs) {
      if (r.live) return r.runId;
    }
    return null;
  }, [activeRunId, conversation]);
  const activityByRun = useConversationActivity(runIds, liveRunId);

  // The newest assistant message is the only one that carries the orb.
  // One avatar per screen reads as the assistant speaking; one per cell
  // turned a long conversation into a column of spinning circles.
  const orbMessageId = useMemo(() => {
    const msgs = conversation?.messages ?? [];
    for (let i = msgs.length - 1; i >= 0; i--) {
      if (msgs[i].role === 'assistant') return msgs[i].id;
    }
    return null;
  }, [conversation]);

  // Auto-scroll to the bottom on new content, but only if the user has
  // not scrolled upward. The "↓ N new messages" pill lets them catch
  // up without losing their place.
  const scrollRef = useRef<HTMLDivElement>(null);
  const [unseenCount, setUnseenCount] = useState(0);
  const [stuckAtBottom, setStuckAtBottom] = useState(true);

  const messages = conversation?.messages ?? [];

  useEffect(() => {
    if (!stuckAtBottom) {
      setUnseenCount(c => c + 1);
      return;
    }
    requestAnimationFrame(() => {
      const el = scrollRef.current;
      if (el) el.scrollTop = el.scrollHeight;
    });
  }, [messages.length, streamingContents, stuckAtBottom]);

  const onScroll = () => {
    const el = scrollRef.current;
    if (!el) return;
    const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
    const atBottom = distance < NEAR_BOTTOM_PX;
    setStuckAtBottom(atBottom);
    if (atBottom) setUnseenCount(0);
  };

  const scrollToBottom = () => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setUnseenCount(0);
    setStuckAtBottom(true);
  };

  const handleSubmit = useCallback(
    async (text: string, attachments: ComposerAttachment[]) => {
      if (isStreaming || flushing) {
        setQueued(q => [...q, { text, attachments }]);
        return;
      }
      // The previous turn's read belongs to the previous turn. Clearing it
      // here rather than on completion is what keeps the evidence beside the
      // answer it produced.
      setOcrPages([]);
      await send(text, classification, {
        attachments,
        ocrDetent: ocrPreference.detent,
      });
    },
    [send, classification, isStreaming, flushing, ocrPreference.detent],
  );

  // Drain the queue a message at a time. `setFlushing(false)` in the
  // `finally` is what re-runs this effect for the next one.
  useEffect(() => {
    if (isStreaming || flushing || queued.length === 0) return;
    const next = queued[0];
    setFlushing(true);
    setQueued(q => q.slice(1));
    setOcrPages([]);
    void send(next.text, classification, {
      attachments: next.attachments,
      ocrDetent: ocrPreference.detent,
    }).finally(() => setFlushing(false));
  }, [isStreaming, flushing, queued, send, classification, ocrPreference.detent]);

  // The task panel shows up only when the latest run had actual
  // orchestration (tools, multiple turns, plan). For a one-line
  // answer it would be a banner that says "I said one thing", so
  // we hide it.
  const showTaskPanel = useMemo(() => {
    if (!conversation) return false;
    const last = conversation.runs[conversation.runs.length - 1];
    if (!last) return false;
    if (last.live) return true;
    const totalMessages = conversation.messages.length;
    if (totalMessages >= 4) return true;
    return false;
  }, [conversation, isStreaming]);

  return (
    <div className={styles.chatRoot}>
      <div className={styles.chatMain}>
        <ChatHeader
          conversation={conversation}
          ledger={headerContext.ledger}
          compactions={headerContext.compactions.length}
          lastCompaction={
            headerContext.compactions.length > 0
              ? headerContext.compactions[headerContext.compactions.length - 1]
              : null
          }
        />
        <div className={styles.chatScrollWrap}>
        <div
          className={styles.chatScroll}
          ref={scrollRef}
          onScroll={onScroll}
        >
          {messages.length === 0 && (
            <p className={styles.emptyState}>
              Ask Arjun anything. The conversation persists across turns.
            </p>
          )}

          {messages.map(m => (
            <MessageRow
              key={m.id}
              message={m}
              isLive={m.status === 'streaming'}
              liveContent={
                m.status === 'streaming'
                  ? streamingContents.get(m.id) ?? ''
                  : undefined
              }
              runId={runsByMessageId.get(m.id) ?? null}
              activity={activityByRun.get(runsByMessageId.get(m.id) ?? '')}
              progress={progressByMessage.get(m.id)}
              reasoning={reasoningByMessage.get(m.id)}
              showAvatar={m.id === orbMessageId}
              runSummary={
                inspectorRunId && runsByMessageId.get(m.id) === inspectorRunId
                  ? taskSummary ?? null
                  : null
              }
              onOpenInspector={runId => setInspectorRunId(runId)}
              /*
               * A widget asking for a turn. Routed through the same `send` the
               * composer uses, so a click inside a model-written page is an
               * ordinary message with an ordinary classification — not a
               * privileged path into the agent.
               */
              onPrompt={text => {
                if (isStreaming) return;
                void send(text);
              }}
              onRetry={() => {
                if (m.role === 'user') {
                  void replay(m);
                } else {
                  // The retry on a failed assistant cell re-sends the
                  // previous user message. Find the user message just
                  // before this assistant message.
                  const idx = messages.findIndex(x => x.id === m.id);
                  for (let i = idx - 1; i >= 0; i--) {
                    if (messages[i].role === 'user') {
                      void replay(messages[i]);
                      return;
                    }
                  }
                }
              }}
              composerDisabled={isStreaming}
            />
          ))}
        </div>

        {!stuckAtBottom && messages.length > 0 && (
          <button
            type="button"
            className={styles.scrollDownBtn}
            onClick={scrollToBottom}
            data-unseen={unseenCount > 0 || undefined}
            aria-label={
              unseenCount > 0
                ? `Scroll to the newest message (${unseenCount} new)`
                : 'Scroll to the newest message'
            }
            title="Scroll to the newest message"
          >
            <ArrowDown size={16} />
          </button>
        )}
        </div>

        <div className={styles.composerWrap}>
          <OcrReadout pages={ocrPages} live={isStreaming || flushing} />
          {reading && (
            <div className={styles.readingStatus} role="status" aria-live="polite">
              <span className={styles.readingPulse} aria-hidden="true" />
              <span>{describeAttachmentProgress(reading)}</span>
              <span className={styles.readingFile}>
                {'\u{1F4C4}'} {reading.name}
                {describeAttachmentKind(reading)
                  ? ' · ' + describeAttachmentKind(reading)
                  : ''}
              </span>
            </div>
          )}
          <ChatComposer
            streaming={isStreaming || flushing}
            activeRunId={activeRunId}
            queued={queued.map(q => q.text)}
            onCancelQueued={i => setQueued(q => q.filter((_, j) => j !== i))}
            onSubmit={handleSubmit}
            ocrPreference={ocrPreference}
          />
        </div>
      </div>

      {inspectorRunId && adopted && (
        <div className={styles.inspectorOverlay} role="dialog" aria-modal="true">
          <div
            className={styles.inspectorBackdrop}
            onClick={() => setInspectorRunId(null)}
          />
          <div className={styles.inspectorPanel}>
            <div className={styles.inspectorHeader}>
              <h3 className={styles.inspectorTitle}>Run details</h3>
              <button
                type="button"
                className={styles.inspectorClose}
                onClick={() => setInspectorRunId(null)}
              >
                Close
              </button>
            </div>
            {stopProblem && (
              <p className={styles.inspectorNotice} role="status">
                {stopProblem}
              </p>
            )}
            <div className={styles.inspectorBody}>
              <RunView
                state={adopted.view}
                onAbort={() => void stopInspectedRun()}
                onNewTask={() => setInspectorRunId(null)}
                stopping={stoppingRunId === inspectorRunId}
              />
            </div>
          </div>
        </div>
      )}

      {showTaskPanel && !inspectorRunId && (
        <TaskPanel
          runId={
            inspectorRunId ??
            (conversation && conversation.runs.length > 0
              ? conversation.runs[conversation.runs.length - 1].runId
              : null)
          }
        />
      )}
    </div>
  );
}

interface MessageRowProps {
  message: ChatMessage;
  isLive?: boolean;
  liveContent?: string;
  /** This turn's own step list, found by message id and never by position. */
  progress?: ProgressStep[];
  /** This turn's reasoning so far. Live only; absent once the run ends. */
  reasoning?: LiveReasoning;
  runId: string | null;
  activity?: Activity[];
  runSummary?: RunSummary | null;
  /** Only the newest assistant cell draws the orb. */
  showAvatar?: boolean;
  onOpenInspector: (runId: string) => void;
  onRetry: () => void;
  /** Raised when a widget in this message asks for a turn. */
  onPrompt?: (text: string) => void;
  composerDisabled?: boolean;
}

function MessageRow({
  message,
  isLive,
  liveContent,
  progress,
  reasoning,
  runId,
  activity,
  runSummary,
  showAvatar,
  onOpenInspector,
  onRetry,
  onPrompt,
  composerDisabled,
}: MessageRowProps) {
  if (message.role === 'user') {
    return (
      <div className={styles.userRow}>
        <div className={styles.userBubble}>
          <p className={styles.userText}>{message.content}</p>
        </div>
      </div>
    );
  }
  if (message.role === 'system') {
    return (
      <div className={styles.systemRow}>
        <p className={styles.systemText}>{message.content}</p>
      </div>
    );
  }
  return (
    <AssistantMessageCell
      message={message}
      isLive={isLive}
      liveContent={liveContent}
      progress={progress}
      reasoning={reasoning}
      activity={runId ? activity : undefined}
      runSummary={runSummary ?? null}
      showAvatar={showAvatar}
      onOpenInspector={runId ? () => onOpenInspector(runId) : undefined}
      onRetry={composerDisabled ? undefined : onRetry}
      onPrompt={composerDisabled ? undefined : onPrompt}
      composerDisabled={composerDisabled}
    />
  );
}

/* `useAdoptedRun` and `useTaskRecord` are imported from `../run/runAdopt`
 * above; the local placeholder that used to live here was removed once
 * those hooks were extracted. */
