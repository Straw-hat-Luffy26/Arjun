/**
 * Shared conversation state.
 *
 * The chat surface and the app shell both need to know what conversation is
 * active and which assistant cell is currently streaming. When those two
 * surfaces called `useConversation()` independently, each got its OWN
 * `useState` slice — different `conversation`, different `streamingContent`,
 * different `isStreaming` — so the header and the message list drifted out
 * of sync. The header would say "1 message" while the body rendered four.
 *
 * The fix is to host the state in a single `ConversationProvider` near the
 * root of the tree and have every caller read it through this context. The
 * reducer for a streaming run lives inside the provider as a per-run
 * closure so events for one run never touch another run's cell. See the
 * comment on `RunReducer` for the per-run isolation contract.
 */
import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from 'react';
import {
  agentService,
  listenAttachmentProgress,
  type AgentEvent,
  type AgentEventEnvelope,
  type ChatMessage,
  type Classification,
  type Conversation,
  endingFromFinishReason,
  type RunOutcomeKind,
  type RunSummary,
  type ComposerAttachment,
} from '../services/agent.service';
import type { OcrDetent } from '../services/ocr.service';
import {
  applyProgress,
  type ProgressInput,
  type ProgressStep,
} from '../components/chat/runProgress';

const LAST_CONVERSATION_KEY = 'arjun.conversation.last';

function rememberConversation(id: string | null) {
  try {
    if (id) sessionStorage.setItem(LAST_CONVERSATION_KEY, id);
    else sessionStorage.removeItem(LAST_CONVERSATION_KEY);
  } catch {
    // A browser with storage disabled loses reattachment across a reload and
    // nothing else. Not worth failing a chat over.
  }
}

function lastConversation(): string | null {
  try {
    return sessionStorage.getItem(LAST_CONVERSATION_KEY);
  } catch {
    return null;
  }
}

function newMessageId(role: 'user' | 'assistant'): string {
  const prefix = role === 'user' ? 'u' : 'a';
  return `${prefix}-${crypto.randomUUID()}`;
}

const STREAMING_PERSIST_DEBOUNCE_MS = 400;
const STREAMING_MIRROR_DEBOUNCE_MS = 30;

/**
 * Longest reasoning kept in the live buffer, in characters.
 *
 * A reasoning pass is unbounded — a model can think for as long as its budget
 * allows — and this buffer is a string in React state that re-renders a panel
 * on every flush. Bounded to the most recent stretch, which is the part
 * anyone is reading; the panel says when it has dropped the beginning rather
 * than presenting a truncated thought as a whole one.
 */
const LIVE_REASONING_LIMIT = 24000;

/** The reasoning a turn has produced so far, and whether it is all of it. */
export interface LiveReasoning {
  text: string;
  /** True once the oldest characters were dropped to bound the buffer. */
  trimmed: boolean;
}

export type MessageEvent = Extract<
  AgentEvent,
  { type: 'message_start' | 'message_update' | 'message_end' }
>;

export function isMessageEvent(event: AgentEvent): event is MessageEvent {
  return (
    event.type === 'message_start' ||
    event.type === 'message_update' ||
    event.type === 'message_end'
  );
}

/**
 * Events that describe progress rather than content.
 *
 * Both carry the assistant `messageId`, which is what lets a reducer accept
 * one before the server has issued its own run id without ever accepting
 * another run's.
 */
export type ProgressEvent = Extract<
  AgentEvent,
  { type: 'run_stage' | 'model_thinking' }
>;

export function isProgressEvent(event: AgentEvent): event is ProgressEvent {
  return event.type === 'run_stage' || event.type === 'model_thinking';
}

/** The `messageId` an event names, or null if it names none. */
export function eventMessageId(event: AgentEvent): string | null {
  if (isMessageEvent(event)) return event.messageId;
  if (event.type === 'model_thinking') return event.messageId;
  if (event.type === 'run_stage') {
    return typeof event.messageId === 'string' ? event.messageId : null;
  }
  return null;
}

/**
 * Runs `work` while holding a lock, and releases it however `work` leaves.
 *
 * ## Why this is a function
 *
 * The lock it owns is the composer's: it stops a second turn starting while one
 * is in flight. Taking it is one line and releasing it is one line, and the
 * distance between them was where the bug lived — the lock was taken, then five
 * awaits happened, and only then did a `try` begin. Any of those five throwing
 * left the lock set with nothing to clear it. The composer then ignored every
 * later turn in silence: no error, no spinner, nothing at all.
 *
 * Written as a helper so the release cannot drift away from the acquisition
 * again, and so the invariant can be driven directly in a test — including the
 * part that matters, which is that the *next* send works.
 *
 * Returns `undefined` without running `work` when the lock is already held.
 * Rethrows whatever `work` threw, after releasing.
 */
export async function runExclusive<T>(
  lock: { current: boolean },
  work: () => Promise<T>,
): Promise<T | undefined> {
  if (lock.current) return undefined;
  lock.current = true;
  try {
    return await work();
  } finally {
    lock.current = false;
  }
}

/**
 * Which conversation a session should open, on mount.
 *
 * ## Why this is a function and not four lines in an effect
 *
 * It *was* four lines in a mount-only effect, and one of them read the
 * `conversation` state variable the effect had closed over. A mount-only
 * closure holds the state as it was at mount — `null`, permanently — so the
 * fallback `if (!conversation)` was always true. A session that successfully
 * restored its remembered thread went on to create a second, empty one and made
 * *that* the open conversation. The restored thread was still on disk; it
 * simply was not the one on screen, and the person's history looked as though
 * it had been lost.
 *
 * Pulled out here so the decision is made from values passed in rather than
 * values captured, and so it can be driven in a test without a DOM.
 */
export async function restoreOrCreate(deps: {
  /** The id the last session remembered, or null. */
  remembered: string | null;
  getConversation: (id: string) => Promise<Conversation | null>;
  createConversation: (title: string) => Promise<Conversation>;
  /** Called when the remembered id names nothing, so it is not tried again. */
  forget: () => void;
}): Promise<{ conversation: Conversation; created: boolean }> {
  if (deps.remembered) {
    const restored = await deps.getConversation(deps.remembered).catch(() => null);
    if (restored) return { conversation: restored, created: false };
    // Deleted, or belonging to another user. Not an error, and not a reason to
    // keep asking for it.
    deps.forget();
  }
  const created = await deps.createConversation('New conversation');
  return { conversation: created, created: true };
}

/**
 * Which conversation a scripted `arjun:trigger-send` turn belongs in.
 *
 * A titled event asks for a thread of its own — that is what the title is for.
 * An untitled one continues whatever is open.
 *
 * The handler used to create the titled conversation and then call `send`,
 * whose captured `conversation` was the mount-time `null`, so `send` created a
 * *second* conversation and put the turn there. One demo click, two threads,
 * and the titled one left empty. Deciding the target here and handing it to
 * `sendTo` explicitly is what makes that unrepresentable.
 */
export async function targetForTrigger(deps: {
  title?: string;
  /** The conversation open right now, read from a ref rather than a render. */
  current: Conversation | null;
  createConversation: (title: string) => Promise<Conversation>;
}): Promise<{ conversation: Conversation | null; created: boolean }> {
  if (deps.title) {
    const created = await deps.createConversation(deps.title);
    return { conversation: created, created: true };
  }
  return { conversation: deps.current, created: false };
}

/**
 * The streaming reducer for a single run.
 *
 * Each `send()` creates exactly one `RunReducer`. The reducer holds the
 * per-run state — messageId, conversationId, runId, live content, persist
 * timer — as plain fields on a normal object, not as React state and not
 * as module-level refs. Two runs in flight at once do not interfere
 * because each one only reads and writes its own fields.
 *
 * The reducer is wired to the event channel via a Tauri listener that
 * forwards envelopes through `envelope.runId`; the listener consults the
 * registry of `RunReducer` instances and routes each envelope to the
 * reducer whose `runId` or `messageId` it matches. Envelopes that match
 * no run are dropped.
 */
export class RunReducer {
  /** Whether the live content buffer has unsent deltas. */
  private dirty = false;
  /** Local content buffer, in arrival order, with repeats collapsed. */
  private content = '';
  /** Mirror debounce timer (pushes content to React state at a throttled rate). */
  private mirrorTimer: number | null = null;
  /**
   * The model's reasoning for this turn, as it arrives.
   *
   * Deliberately a second buffer rather than part of `content`, and the
   * separation is the whole safety property. `content` is persisted on a
   * timer, sent as `finalContent`, resolved against by the verifier and
   * written into the audit record; reasoning belongs in none of those. It is
   * published to the surface, shown while the turn runs, and dropped when the
   * reducer is disposed.
   */
  private reasoning = '';
  /** True once the buffer has dropped its oldest characters. */
  private reasoningTrimmed = false;
  /** Mirror timer for the reasoning buffer. Same rate, separate schedule. */
  private reasoningTimer: number | null = null;
  /** Persist debounce timer (writes the latest snapshot to disk). */
  private persistTimer: number | null = null;
  /** Whether `message_end` has been received. The reducer accepts at most
   * one of these for the lifetime of the run. */
  private receivedEnd = false;
  /** Server-issued runId, captured from the first `plan_ready` envelope. */
  private actualRunId: string | null = null;
  /** Timestamp the run started, used for elapsed-time metrics. */
  readonly startedAt: number;
  /** Set when `dispose()` runs. Late events are dropped on the floor. */
  private disposed = false;
  /**
   * The steps this turn has been through, newest last.
   *
   * Held on the reducer rather than in React state for the same reason the
   * content buffer is: two runs in flight each own their own list, and an
   * event that matched neither cannot append to either.
   */
  private progress: ProgressStep[] = [];

  constructor(
    private readonly registry: RunReducerRegistry,
    readonly runId: string,
    readonly conversationId: string,
    readonly messageId: string,
  ) {
    this.startedAt = Date.now();
    // Seeded before any event arrives, so the cell has a line to show during
    // the IPC round trip that reserves the turn. It names work that is
    // genuinely under way — the request has been handed over — rather than
    // filling the gap with a spinner that means nothing.
    this.pushProgress({ kind: 'submitted' });
  }

  /**
   * Folds in progress that arrived on a channel other than `agent://event`.
   *
   * The document reader reports its pages on `attachment:progress`, which is
   * an application-wide channel; the registry routes one of those to this
   * reducer only when it names this reducer's message. A read that names no
   * message reaches nobody, which is the safe direction to fail.
   */
  ingest(input: ProgressInput): void {
    if (this.disposed) return;
    this.pushProgress(input);
  }

  /**
   * Records the server's run id, and tells the rest of the app about it.
   *
   * ## Why publishing it matters more than holding it
   *
   * Two ids name a turn. The caller mints a correlation id before the request
   * goes out, because it needs *something* to route events by while the run is
   * being created; the server then mints the run's real id, which is what the
   * task record, the audit trail, the event stream and the runtime's own table
   * of live runs are all keyed by.
   *
   * This reducer has always learned the real one and always kept it to itself.
   * Everything outside went on holding the correlation id, and three things
   * that address a run by id were therefore addressing one that did not exist:
   *
   * - **Stop.** The composer sent the correlation id to `agent_abort_run`. The
   *   core found no such run, the runtime found no such run, and the call
   *   returned `false` — a stop that did nothing at all, silently, while the
   *   model carried on using the machine.
   * - **The context meter.** It subscribes to `context_ledger` events filtered
   *   by run id. None ever matched, so the meter read "No context yet" for the
   *   whole of every run and only filled in once the run had finished.
   * - **Reattachment.** A window following the active run followed an id
   *   nothing would ever emit under.
   *
   * Called from both places that can learn the id — `plan_ready`, and the first
   * message event from a model fast enough to beat it — so there is one path
   * that records it and one path that announces it.
   */
  private learnRunId(runId: string): void {
    if (this.actualRunId === runId) return;
    this.actualRunId = runId;
    this.registry.publishRunId(this.messageId, runId);
  }

  /** Folds one progress event in and publishes the new list. */
  private pushProgress(input: ProgressInput): void {
    const next = applyProgress(this.progress, input, Date.now());
    if (next === this.progress) return;
    this.progress = next;
    this.registry.publishProgress(this.messageId, next);
  }

  /** Push the latest content into React state on the next microtask. */
  private scheduleMirror(): void {
    if (this.mirrorTimer !== null) return;
    this.mirrorTimer = window.setTimeout(() => {
      this.mirrorTimer = null;
      this.registry.publishContent(this.messageId, this.content);
    }, STREAMING_MIRROR_DEBOUNCE_MS);
  }

  /** Push the latest reasoning into React state, at the mirror rate. */
  private scheduleReasoningMirror(): void {
    if (this.reasoningTimer !== null) return;
    this.reasoningTimer = window.setTimeout(() => {
      this.reasoningTimer = null;
      this.publishReasoning();
    }, STREAMING_MIRROR_DEBOUNCE_MS);
  }

  /** Publish now, cancelling any pending mirror. */
  private flushReasoningMirror(): void {
    if (this.reasoningTimer !== null) {
      window.clearTimeout(this.reasoningTimer);
      this.reasoningTimer = null;
    }
    this.publishReasoning();
  }

  private publishReasoning(): void {
    this.registry.publishReasoning(this.messageId, {
      text: this.reasoning,
      trimmed: this.reasoningTrimmed,
    });
  }

  /** Persist the current content snapshot to the conversation store. */
  private schedulePersist(): void {
    this.dirty = true;
    if (this.persistTimer !== null) return;
    this.persistTimer = window.setTimeout(() => {
      this.persistTimer = null;
      if (!this.dirty) return;
      this.dirty = false;
      void agentService
        .updateStreamingContent(
          this.conversationId,
          this.messageId,
          this.content,
        )
        .then((next) => {
          if (next) this.registry.publishConversation(next);
        })
        .catch(() => undefined);
    }, STREAMING_PERSIST_DEBOUNCE_MS);
  }

  /** Force any pending persist to run now. Used on `message_end`. */
  private flushPersist(): void {
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
    if (!this.dirty) return;
    this.dirty = false;
    void agentService
      .updateStreamingContent(
        this.conversationId,
        this.messageId,
        this.content,
      )
      .then((next) => {
        if (next) this.registry.publishConversation(next);
      })
      .catch(() => undefined);
  }

  /**
   * Apply one envelope to the reducer. Returns true if the envelope was
   * consumed (i.e. matched this run), false if it should be passed on to
   * any other reducers in the registry.
   */
  apply(envelope: AgentEventEnvelope): boolean {
    if (this.disposed) return false;
    const event = envelope.event;

    // `plan_ready` is the one envelope that teaches this reducer the
    // server's run id, by echoing back the correlation id the caller sent.
    if (event.type === 'plan_ready') {
      if (event.correlationId === this.runId) {
        this.learnRunId(envelope.runId);
        return true;
      }
      return this.actualRunId !== null && envelope.runId === this.actualRunId;
    }

    // Two ids can identify this run, and both are needed.
    //
    // The server's run id is authoritative once `plan_ready` has taught it
    // to us. Before that — which is most of a cold turn, because the stages
    // that report attachment reading, routing and model loading all happen
    // before the run has an id — the envelope carries the caller's own
    // correlation id, which is this reducer's `runId`. Matching only the
    // first would drop every stage emitted before generation started, which
    // is exactly the silence this work exists to remove.
    const namedMessage = eventMessageId(event);
    const matchesRun =
      (this.actualRunId !== null && envelope.runId === this.actualRunId) ||
      envelope.runId === this.runId;
    const matchesMessage =
      namedMessage !== null && namedMessage === this.messageId;
    if (!matchesRun && !matchesMessage) return false;

    // An event that names a message must name OURS, whatever run it claims.
    // Without this, a run holding two assistant cells would let the second
    // cell's tokens land in the first.
    if (namedMessage !== null && namedMessage !== this.messageId) return false;

    // A message event is stamped with the server's run id even before
    // `plan_ready` arrives, so a fast model whose first `message_start`
    // beats the plan still teaches us the id.
    if (
      this.actualRunId === null &&
      matchesMessage &&
      envelope.runId !== this.runId
    ) {
      this.learnRunId(envelope.runId);
    }

    if (isProgressEvent(event)) {
      if (event.type === 'model_thinking') {
        this.pushProgress({
          kind: 'thinking',
          state: event.state,
          characters: event.characters,
          elapsedMs: event.elapsedMs,
        });
        // A turn can reason more than once — before its first tool call, and
        // again after each result. The blocks are separated rather than run
        // together, so the panel reads as the passes of thought it was.
        if (event.state === 'start' && this.reasoning.length > 0) {
          this.reasoning += String.fromCharCode(10, 10);
        }
        if (typeof event.delta === 'string' && event.delta.length > 0) {
          this.reasoning += event.delta;
          if (this.reasoning.length > LIVE_REASONING_LIMIT) {
            this.reasoning = this.reasoning.slice(-LIVE_REASONING_LIMIT);
            this.reasoningTrimmed = true;
          }
          this.scheduleReasoningMirror();
        }
        // The closing frame carries the tail the ticks had not reached, so it
        // is published at once rather than left to a timer the end of the run
        // would cancel.
        if (event.state === 'end') this.flushReasoningMirror();
      } else {
        const { type, stage, elapsedMs, ...detail } = event;
        void type;
        this.pushProgress({
          kind: 'stage',
          stage,
          elapsedMs,
          detail: detail as Record<string, unknown>,
        });
      }
      return true;
    }

    if (!isMessageEvent(event)) return true; // consume but no-op

    switch (event.type) {
      case 'message_start':
        this.content = '';
        this.registry.publishContent(this.messageId, this.content);
        return true;
      case 'message_update': {
        // The exact bytes the model produced, in the order it produced them.
        //
        // This used to be `collapseRepeats(next, ...)`, which rewrote the text
        // *before* it became `this.content` — and `this.content` is what gets
        // persisted, what is sent as `finalContent`, and therefore what the
        // verifier resolves citations against and what the audit record holds.
        // A display convenience was editing the evidence. Worse, it collapses
        // by sentence and by repeated substring, so a code block with two
        // identical lines, a JSON array with repeated values, or a table with a
        // repeated cell came out altered in the file the person then signed.
        //
        // Repetition is now collapsed at render time and nowhere else — see
        // `collapseForDisplay`, which says when it has done it.
        this.content = this.content + event.delta;
        // The first visible token is the only place composition can be
        // observed starting. No stage on the Rust side can know it, because
        // the Rust side is blocked on the loop by then.
        if (this.content.length > 0) this.pushProgress({ kind: 'text' });
        this.scheduleMirror();
        this.schedulePersist();
        return true;
      }
      case 'message_end': {
        if (this.receivedEnd) return true;
        this.receivedEnd = true;
        // Closes whatever was still open. A panel left showing an unfinished
        // "Writing the answer" under a finished answer is the same class of
        // lie as showing nothing at all.
        this.pushProgress({ kind: 'done' });
        if (this.mirrorTimer !== null) {
          window.clearTimeout(this.mirrorTimer);
          this.mirrorTimer = null;
        }
        this.registry.publishContent(this.messageId, this.content);
        this.flushPersist();
        const elapsed = Date.now() - this.startedAt;
        const runId = this.actualRunId ?? this.runId;
        const finalContent = this.content;
        void agentService
          .completeMessage({
            conversationId: this.conversationId,
            messageId: this.messageId,
            runId,
            finalContent,
            elapsedMs: elapsed,
            // What this writer honestly knows, which is narrower than the
            // run's ending. `length` and `error` are facts about the model's
            // last turn that the reducer can see for itself; everything else
            // is left for the run's own write, which is the authority and
            // lands after this one.
            ...endingFromFinishReason(event.finishReason),
            tokensIn: event.tokensIn,
            tokensOut: event.tokensOut,
          })
          .then((next) => {
            if (next) this.registry.publishConversation(next);
            this.registry.onRunDone(this);
          })
          .catch(() => {
            this.registry.onRunDone(this);
          });
        return true;
      }
    }
  }

  /**
   * Tear the reducer down. Used when the run completes, fails, or is
   * cancelled — the reducer is removed from the registry so its memory
   * can be reclaimed and so the event listener stops trying to route
   * to it.
   */
  dispose(): void {
    this.disposed = true;
    if (this.mirrorTimer !== null) {
      window.clearTimeout(this.mirrorTimer);
      this.mirrorTimer = null;
    }
    if (this.reasoningTimer !== null) {
      window.clearTimeout(this.reasoningTimer);
      this.reasoningTimer = null;
    }
    if (this.persistTimer !== null) {
      window.clearTimeout(this.persistTimer);
      this.persistTimer = null;
    }
  }
}

/**
 * The registry of currently-active run reducers.
 *
 * Holds the `Map<runId, RunReducer>` and routes events to the right
 * reducer. Also publishes state changes (content updates, conversation
 * updates) up to the React context so the UI re-renders.
 */
export class RunReducerRegistry {
  private readonly reducers = new Map<string, RunReducer>();
  private readonly byMessageId = new Map<string, RunReducer>();
  private readonly unsubscribers = new Map<string, () => void>();
  private activeUnlisten: (() => void) | null = null;

  constructor(
    private readonly publish: {
      onContent: (messageId: string, content: string) => void;
      /**
       * Optional, unlike the rest. Reasoning is the only stream a consumer can
       * reasonably not want: it is live-only decoration, and a caller that
       * renders finished conversations has nothing to do with it. The others
       * are load-bearing and stay required.
       */
      onReasoning?: (messageId: string, reasoning: LiveReasoning) => void;
      onProgress: (messageId: string, steps: ProgressStep[]) => void;
      onConversation: (next: Conversation) => void;
      /**
       * The server's own id for the run streaming into `messageId`.
       *
       * Optional for the same reason `onReasoning` is: a consumer that renders
       * finished conversations has no live run to address. Everything that
       * *does* address one — Stop, the context meter, reattachment — needs
       * this, because the id it otherwise holds is the caller's correlation id
       * and nothing on the other side has heard of it.
       */
      onRunId?: (messageId: string, runId: string) => void;
      onRunDone: (reducer: RunReducer) => void;
    },
  ) {}

  /**
   * Register a new run and subscribe to the event channel. The returned
   * function tears the registration down.
   */
  register(reducer: RunReducer): () => void {
    this.reducers.set(reducer.runId, reducer);
    this.byMessageId.set(reducer.messageId, reducer);
    if (this.reducers.size === 1) {
      this.activeUnlisten = this.subscribe();
    }
    let disposed = false;
    return () => {
      if (disposed) return;
      disposed = true;
      this.reducers.delete(reducer.runId);
      this.byMessageId.delete(reducer.messageId);
      const unlisten = this.unsubscribers.get(reducer.runId);
      if (unlisten) {
        this.unsubscribers.delete(reducer.runId);
        unlisten();
      }
      reducer.dispose();
      if (this.reducers.size === 0 && this.activeUnlisten) {
        this.activeUnlisten();
        this.activeUnlisten = null;
      }
    };
  }

  /** Per-reducer subscriber: each reducer gets its own listener so it
   * owns its filter logic in one place. */
  private subscribe(): () => void {
    let stopped = false;
    let unlisten: (() => void) | null = null;
    void agentService.subscribe((envelope) => {
      if (stopped) return;
      // Try the runId-keyed reducer first, then fall back to the
      // messageId-keyed one (for envelopes that arrive before the
      // server has stamped its own runId, like the first
      // `message_start` from a fast-streaming model).
      let reducer = this.reducers.get(envelope.runId);
      if (!reducer) {
        // The table is keyed by the caller's own run id, so every envelope
        // stamped with the *server's* id misses it. Both message events and
        // progress events name the assistant message, and that name is what
        // finds the reducer in that case.
        const named = eventMessageId(envelope.event);
        if (named !== null) reducer = this.byMessageId.get(named);
      }
      if (!reducer) return;
      reducer.apply(envelope);
    }).then((fn) => {
      if (stopped) fn();
      else unlisten = fn;
    });
    return () => {
      stopped = true;
      if (unlisten) unlisten();
    };
  }

  publishContent(messageId: string, content: string): void {
    this.publish.onContent(messageId, content);
  }

  publishReasoning(messageId: string, reasoning: LiveReasoning): void {
    this.publish.onReasoning?.(messageId, reasoning);
  }

  publishRunId(messageId: string, runId: string): void {
    this.publish.onRunId?.(messageId, runId);
  }

  publishProgress(messageId: string, steps: ProgressStep[]): void {
    this.publish.onProgress(messageId, steps);
  }

  /** Hands an off-channel progress event to the reducer that owns it. */
  ingestForMessage(messageId: string, input: ProgressInput): void {
    this.byMessageId.get(messageId)?.ingest(input);
  }

  publishConversation(next: Conversation): void {
    this.publish.onConversation(next);
  }

  onRunDone(reducer: RunReducer): void {
    this.publish.onRunDone(reducer);
    // The reducer already wrote the final state. The caller (the
    // reducer itself) is responsible for unregistering.
  }

  /** Iterate over the active reducers. Used for tests and cleanup. */
  values(): IterableIterator<RunReducer> {
    return this.reducers.values();
  }
}

/**
 * Collapses repeated content **for display only**, and says whether it did.
 *
 * ## Why this is not on the storage path
 *
 * Small models fall into loops and emit the same sentence two or three times.
 * The raw stream faithfully reflects the model, and the duplication makes an
 * answer look broken — so it is worth hiding, on screen.
 *
 * It was not doing that. The collapse ran inside the streaming reducer, on the
 * buffer that is persisted, sent as `finalContent`, resolved against by the
 * verifier and written into the audit record. A display convenience was
 * editing the evidence, and doing it by sentence and by repeated substring: a
 * code block with two identical lines, a JSON array with repeated values, a
 * table with a repeated cell all came out altered in the file somebody then
 * signed.
 *
 * So it is a rendering step. It returns the collapsed text *and* the fact that
 * it collapsed something, so the surface can say so rather than quietly
 * present an edited answer as the model's words. Nothing calls it on the way
 * to disk.
 *
 * Fenced code is left alone entirely: a repeated line in code is code.
 *
 * Two passes:
 *  1. Sentence-level: if the last N characters end with the same
 *     sentence that already appears earlier, strip the earlier copy.
 *  2. Word-level: if the last 200 chars contain an immediate repeat
 *     of a chunk (8..200 chars), strip one copy.
 */
export interface DisplayText {
  /** What to render. */
  text: string;
  /** Whether anything was removed to produce it. */
  collapsed: boolean;
}

/**
 * The display form of an answer, and whether it differs from what was stored.
 *
 * The only entry point the UI should use. `collapseRepeats` below is the
 * mechanism; this is the contract, and it is the one that carries the flag.
 */
export function collapseForDisplay(stored: string): DisplayText {
  // Fenced code is never touched: a repeated line inside a code block is code,
  // and a JSON array with two equal elements is data. Rather than trying to
  // collapse around the fences, an answer containing any is shown verbatim —
  // the failure this guards against is far worse than a visible repetition.
  if (stored.includes('```')) return { text: stored, collapsed: false };

  const text = collapseRepeats(stored);
  return { text, collapsed: text !== stored };
}

function collapseRepeats(s: string, deltaHadTerminator = true): string {
  if (s.length < 20) return s;

  // The sentence pass is the expensive half: it builds a `RegExp` and runs it
  // over the *whole* accumulated answer, so running it on every delta makes
  // the cost of streaming quadratic in the length of the answer. At a few
  // hundred characters that is invisible; at ten thousand it is the reducer
  // stuttering the stream it exists to smooth.
  //
  // It is also unnecessary. The pass can only find a new duplicate once a new
  // sentence has been *completed*, and that cannot happen in a delta with no
  // terminator in it. Skipping those is not an approximation — the result is
  // the same string, arrived at without re-scanning the answer a thousand
  // times. The word-level pass below is already bounded to the last 400
  // characters and runs every time.
  if (!deltaHadTerminator) return collapseTail(s);

  const sentenceRe = /[.!?]\s+([^.!?]{8,200})[.!?]/g;
  const tail1 = s.slice(-500);
  const tailSentences: string[] = [];
  let m: RegExpExecArray | null;
  while ((m = sentenceRe.exec(tail1)) !== null) {
    tailSentences.push(m[1]);
  }
  if (tailSentences.length > 0) {
    const last = tailSentences[tailSentences.length - 1];
    const escaped = last.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const dupRe = new RegExp(
      `(?:^|[.!?]\\s+)${escaped}[.!?](?:\\s+${escaped}[.!?])+`,
      'g',
    );
    if (dupRe.test(s)) {
      return s.replace(dupRe, (match) => {
        const parts = match.split(
          new RegExp(`(?<=[.!?])\\s+(?=${escaped}[.!?])`),
        );
        return parts[parts.length - 1] ?? match;
      });
    }
  }

  return collapseTail(s);
}

/**
 * The cheap half: strip an immediate repeat of the last 8..200 characters.
 *
 * Bounded to the last 400 characters, so its cost does not grow with the
 * length of the answer.
 */
function collapseTail(s: string): string {
  const tail = s.slice(-400);
  for (let len = Math.min(200, Math.floor(tail.length / 2)); len >= 8; len -= 1) {
    const last = tail.slice(-len);
    const before = tail.slice(-len * 2, -len);
    if (last === before) {
      return s.slice(0, s.length - len);
    }
  }
  return s;
}

export interface UseConversation {
  conversation: Conversation | null;
  conversations: Conversation[];
  isStreaming: boolean;
  activeMessageId: string | null;
  activeRunId: string | null;
  streamingContents: Map<string, string>;
  /**
   * What each in-flight turn is doing, keyed by assistant `messageId`.
   *
   * Keyed by message rather than by run because that is what the cell that
   * renders it is keyed by, and a map from the thing rendering to the thing
   * rendered cannot put one turn's progress under another's answer.
   */
  progressByMessage: Map<string, ProgressStep[]>;
  /**
   * The reasoning each in-flight turn has produced, keyed by assistant
   * `messageId`.
   *
   * Live only, and absent for every message that is not currently streaming.
   * Nothing here is stored: reopening a conversation shows the answers and
   * none of the thinking, because the thinking was never written down.
   */
  reasoningByMessage: Map<string, LiveReasoning>;
  send: (
    prompt: string,
    classification?: Classification,
    options?: {
      scenarioInstructions?: string;
      attachments?: ComposerAttachment[];
      ocrDetent?: OcrDetent;
    },
  ) => Promise<void>;
  open: (conversationId: string) => Promise<void>;
  newConversation: () => Promise<void>;
  refresh: () => Promise<void>;
  complete: (args: {
    finalContent: string;
    elapsedMs: number;
    modelName?: string;
    modelRole?: string;
    usedFallback?: boolean;
    error?: string;
    failed?: boolean;
  }) => Promise<void>;
  replay: (userMessage: ChatMessage) => Promise<void>;
}

const ConversationContext = createContext<UseConversation | null>(null);

export function ConversationProvider({ children }: { children: React.ReactNode }) {
  const [conversation, setConversation] = useState<Conversation | null>(null);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [activeMessageId, setActiveMessageId] = useState<string | null>(null);
  const [activeRunId, setActiveRunId] = useState<string | null>(null);
  const [streamingContents, setStreamingContents] = useState<Map<string, string>>(
    () => new Map(),
  );
  const [progressByMessage, setProgressByMessage] = useState<
    Map<string, ProgressStep[]>
  >(() => new Map());
  const [reasoningByMessage, setReasoningByMessage] = useState<
    Map<string, LiveReasoning>
  >(() => new Map());

  /**
   * A ref-based lock so two rapid `send()` invocations cannot both
   * proceed. `isStreaming` lives in React state and is one tick behind
   * the truth; a ref check inside `send` is the only synchronous way
   * to catch a double-press. The lock is released in `send`'s `finally`
   * regardless of how `start()` resolves.
   */
  const sendingRef = useRef(false);

  /**
   * The assistant cell currently streaming, readable synchronously.
   *
   * `activeMessageId` lives in React state and is a render behind, and the
   * registry callback below has to answer "is this the run the surface is
   * showing?" the moment an event arrives. Two runs can be registered at once —
   * the composer's lock stops a second *send*, not a run adopted from
   * elsewhere — so without this check the second run's id would land in
   * `activeRunId` and Stop would abort the wrong turn.
   *
   * Written through {@link setActiveCell}, which sets both halves at once so
   * they cannot drift.
   */
  const activeMessageIdRef = useRef<string | null>(null);
  const setActiveCell = useCallback((messageId: string | null) => {
    activeMessageIdRef.current = messageId;
    setActiveMessageId(messageId);
  }, []);

  // The registry holds the per-run reducers and the shared event
  // subscription. The set of callbacks it calls is stable for the
  // lifetime of the provider.
  const registryRef = useRef<RunReducerRegistry | null>(null);
  if (registryRef.current === null) {
    registryRef.current = new RunReducerRegistry({
      onContent: (messageId, content) => {
        setStreamingContents((prev) => {
          const next = new Map(prev);
          next.set(messageId, content);
          return next;
        });
      },
      onReasoning: (messageId, reasoning) => {
        setReasoningByMessage((prev) => {
          const next = new Map(prev);
          next.set(messageId, reasoning);
          return next;
        });
      },
      onProgress: (messageId, steps) => {
        setProgressByMessage((prev) => {
          const next = new Map(prev);
          next.set(messageId, steps);
          return next;
        });
      },
      onConversation: (next) => {
        setConversation(next);
      },
      // The correlation id gives way to the run's own, the moment there is one.
      //
      // Only for the cell the surface is actually showing: a second run in
      // flight has its own id and its own cell, and letting it overwrite this
      // would point Stop at the wrong turn. Read from a ref rather than from
      // state because this fires between renders.
      onRunId: (messageId, runId) => {
        if (activeMessageIdRef.current !== messageId) return;
        setActiveRunId(runId);
      },
      onRunDone: () => {
        // The reducer handles its own teardown. We just mark the run
        // as no longer streaming if no other runs are active.
        if (registryRef.current && registryRef.current.values().next().done) {
          setIsStreaming(false);
          activeMessageIdRef.current = null;
          setActiveMessageId(null);
          setActiveRunId(null);
        }
      },
    });
  }

  // The document reader's own page counter, folded into the same step list
  // as everything else so a person sees one account of the turn rather than
  // two. Subscribed once for the life of the provider; the routing is by the
  // message id the reader stamps, never by which turn happens to be newest.
  useEffect(() => {
    const sub = listenAttachmentProgress((payload) => {
      const registry = registryRef.current;
      if (!registry || !payload.messageId) return;
      registry.ingestForMessage(payload.messageId, {
        kind: 'attachmentPage',
        name: payload.name,
        page: payload.page,
        pages: payload.pages,
        phase: payload.phase,
      });
    });
    return () => {
      void sub.then((un) => un());
    };
  }, []);

  const refresh = useCallback(async () => {
    try {
      const list = await agentService.listConversations();
      setConversations(list);
    } catch {
      // A failing list call should not break the chat.
    }
  }, []);

  const reloadActive = useCallback(async (id: string | null) => {
    if (!id) {
      setConversation(null);
      return;
    }
    try {
      const next = await agentService.getConversation(id);
      setConversation(next);
    } catch {
      setConversation(null);
    }
  }, []);

  const open = useCallback(
    async (conversationId: string) => {
      rememberConversation(conversationId);
      await reloadActive(conversationId);
      await refresh();
    },
    [reloadActive, refresh],
  );

  const newConversation = useCallback(async () => {
    const created = await agentService.createConversation('New conversation');
    rememberConversation(created.id);
    await reloadActive(created.id);
    await refresh();
  }, [refresh, reloadActive]);

  /**
   * Sends a turn into a conversation the caller names.
   *
   * ## Why the conversation is a parameter
   *
   * It used to be read from the `conversation` state variable this callback
   * closed over. Every consumer that was not re-rendered by React got the value
   * as it stood when *their* closure was made — and the one consumer that
   * matters, the `arjun:trigger-send` listener, is registered in a mount-only
   * effect. Its `send` therefore saw `conversation === null` forever, so a demo
   * event that had just created a conversation went on to create a *second*
   * one and sent its turn there. Two conversations, one turn, and the titled
   * one left empty.
   *
   * Passing the target in makes that impossible to get wrong: a caller that
   * knows which conversation it means says so, and `send` supplies the live one
   * from a ref rather than from a captured render.
   */
  const sendTo = useCallback(
    async (
      target: Conversation | null,
      prompt: string,
      classification?: Classification,
      options?: {
        scenarioInstructions?: string;
        attachments?: ComposerAttachment[];
        /** Where the accuracy-to-speed slider was left. Reading only. */
        ocrDetent?: OcrDetent;
      },
    ) => {
      // A turn with an attachment and no words is still a turn: "read this"
      // is implied by attaching it. Refusing here is how an attached document
      // silently goes nowhere.
      if (!prompt.trim() && !(options?.attachments ?? []).length) return;

      // ── The lock, and everything it covers ──────────────────────────
      //
      // `sendingRef` stops a second turn starting while one is in flight.
      // `runExclusive` owns taking it and releasing it, so there is no path
      // through this function on which it is left set.
      //
      // There used to be several. The lock was taken here and the `try` did not
      // begin until after creating a conversation, reserving the turn,
      // refreshing the list and registering the reducer. Any of those throwing
      // left `sendingRef.current === true` with nothing to clear it, and the
      // composer was dead for the rest of the session: every later send
      // returned early and did nothing at all, silently. One transient backend
      // hiccup cost the person every turn after it.
      await runExclusive(sendingRef, async () => {
      let unregister: (() => void) | null = null;
      let conv: Conversation | null = target;
      let runId: string | null = null;
      let messageId: string | null = null;
      let failure: Error | null = null;
      try {
        const registry = registryRef.current;
        if (!registry) {
          // Nothing is listening for events yet, so a run started now would
          // stream into nothing. A race with mounting, not a failure.
          return;
        }

        if (!conv) {
          const title =
            prompt
              .split('\n')
              .map((line) => line.trim())
              .find((line) => line.length > 0) ?? 'New conversation';
          const created = await agentService.createConversation(title.slice(0, 80));
          conv = created;
          rememberConversation(created.id);
        }

        runId = crypto.randomUUID();
        messageId = newMessageId('assistant');
        const cellId = messageId;

        // Reserve the user message and the assistant cell on the
        // conversation, and bind the run id so the runtime can route
        // streaming events to the right cell. Each call is its own
        // server-issued `messageId`, and the per-run reducer captures
        // the id so subsequent events only update THIS cell.
        const updated = await agentService.appendTurn(
          conv.id,
          runId,
          cellId,
          prompt,
        );
        if (updated) {
          setConversation(updated);
          // Clear any stale streaming content for this new cell.
          setStreamingContents((prev) => {
            if (!prev.has(cellId)) return prev;
            const next = new Map(prev);
            next.set(cellId, '');
            return next;
          });
        }
        // Best-effort: a sidebar that failed to reload is stale, not a reason
        // to abandon a turn the person has already committed to.
        await refresh().catch(() => undefined);

        setIsStreaming(true);
        setActiveCell(cellId);
        setActiveRunId(runId);

        // Each run gets its OWN reducer. Two concurrent runs would each
        // have one, with no shared refs and no cross-contamination.
        const reducer = new RunReducer(registry, runId, conv.id, cellId);
        unregister = registry.register(reducer);

        const summary: RunSummary | null = await agentService.start({
          prompt,
          classification,
          scenarioInstructions: options?.scenarioInstructions,
          // Bound to THIS request. The backend keeps nothing between runs, so
          // a later turn cannot inherit this turn's document.
          attachments: options?.attachments,
          ocrDetent: options?.ocrDetent,
          conversationId: conv.id,
          messageId,
          correlationId: runId,
        });
        // `start` returning is the authoritative end-of-run signal.
        // The reducer's `message_end` handler will also have fired by
        // here in the normal path; resetting state unconditionally
        // covers runs whose summary lacks a `messageId` (refused,
        // tool-only, transient failure).
        if (summary?.messageId) {
          const next = await agentService
            .getConversation(conv.id)
            .catch(() => null);
          if (next) setConversation(next);
          await refresh();
        }
      } catch (error) {
        failure = error instanceof Error ? error : new Error(String(error));
      } finally {
        // ── Released on every path ────────────────────────────────────
        //
        // Including the ones that failed before there was a run, a cell, or
        // even a conversation. Every step below is written to tolerate the
        // half-built state its own step never reached, because the whole point
        // of this block is that it runs when an earlier one did not.

        // The reducer may have already torn itself down on `message_end`; if
        // not (the run was cancelled or the connection failed), tear it down
        // now. Null when the failure happened before it was registered.
        unregister?.();
        setIsStreaming(false);
        setActiveCell(null);
        setActiveRunId(null);

        if (failure && conv && runId && messageId) {
          // There is a cell on disk showing as streaming for a run that will
          // never stream. Closing it is what stops a spinner outliving the
          // turn by the rest of the session.
          const message = failure.message;
          await agentService
            .completeMessage({
              conversationId: conv.id,
              messageId,
              runId,
              finalContent: '',
              elapsedMs: 0,
              error: message,
              outcome: 'failed',
              failed: true,
            })
            .catch(() => undefined);
          const next = await agentService
            .getConversation(conv.id)
            .catch(() => null);
          if (next) setConversation(next);
        } else if (failure) {
          // The turn was never reserved, so there is nothing on disk to close.
          // The list is reloaded anyway: a conversation may have been created
          // before the failure, and a sidebar that does not show it would be
          // the only trace of a thread the person cannot find.
          await refresh().catch(() => undefined);
        }
      }
      });
    },
    // No `conversation` dependency: the target is a parameter now, so this
    // callback is stable and a consumer holding an old copy of it cannot be
    // holding an old copy of the conversation with it.
    [refresh],
  );

  /**
   * The conversation as it stands, for callers that were not re-rendered.
   *
   * A mount-only effect's closure is made once and never again, so anything it
   * reads out of a render is frozen at `null`. A ref is the value now.
   */
  const conversationRef = useRef<Conversation | null>(null);
  useEffect(() => {
    conversationRef.current = conversation;
  }, [conversation]);

  /**
   * The current `sendTo`, for the mount-only listener.
   *
   * `sendTo` is stable today, but a listener registered once and holding a
   * callback by value is exactly the shape that broke here. Reading it through
   * a ref means a future dependency added to `sendTo` cannot quietly
   * reintroduce the bug.
   */
  const sendToRef = useRef(sendTo);
  useEffect(() => {
    sendToRef.current = sendTo;
  }, [sendTo]);

  /** Sends into whichever conversation is open right now. */
  const send = useCallback(
    (
      prompt: string,
      classification?: Classification,
      options?: {
        scenarioInstructions?: string;
        attachments?: ComposerAttachment[];
        ocrDetent?: OcrDetent;
      },
    ) => sendTo(conversationRef.current, prompt, classification, options),
    [sendTo],
  );

  const complete = useCallback(
    async (args: {
      finalContent: string;
      elapsedMs: number;
      modelName?: string;
      modelRole?: string;
      usedFallback?: boolean;
      error?: string;
      /** How the run ended, when the caller knows. See {@link RunOutcomeKind}. */
      outcome?: RunOutcomeKind;
      failed?: boolean;
    }) => {
      if (!conversation || !activeMessageId || !activeRunId) return;
      const next = await agentService
        .completeMessage({
          conversationId: conversation.id,
          messageId: activeMessageId,
          runId: activeRunId,
          finalContent: args.finalContent,
          elapsedMs: args.elapsedMs,
          modelName: args.modelName,
          modelRole: args.modelRole,
          usedFallback: args.usedFallback,
          error: args.error,
          outcome: args.outcome,
          failed: args.failed ?? false,
        })
        .catch(() => null);
      if (next) setConversation(next);
      await refresh();
      setIsStreaming(false);
      setActiveCell(null);
      setActiveRunId(null);
    },
    [conversation, activeMessageId, activeRunId, refresh],
  );

  const replay = useCallback(
    async (userMessage: ChatMessage) => {
      if (userMessage.role !== 'user') return;
      if (!conversation) return;
      await send(userMessage.content);
    },
    [conversation, send],
  );

  // Restore the last conversation on mount. Falls back to a fresh one
  // if the remembered id is gone.
  //
  // ## Why nothing here reads `conversation`
  //
  // This effect runs once, so its closure holds the state as it was at mount:
  // `conversation === null`, permanently. The fallback used to be written
  // `if (!conversation)`, which was therefore *always* true — so a session that
  // successfully restored its remembered thread went on to create a second,
  // empty one and made that the open conversation. The restored thread was
  // still on disk; it simply was not the one on screen.
  //
  // What was restored is tracked in a local instead. It is the value this
  // function actually computed, and no render can stale it.
  useEffect(() => {
    let cancelled = false;
    void (async () => {
      const { conversation: active, created } = await restoreOrCreate({
        remembered: lastConversation(),
        getConversation: (id) => agentService.getConversation(id),
        createConversation: (title) => agentService.createConversation(title),
        forget: () => rememberConversation(null),
      });
      if (cancelled) return;
      if (created) rememberConversation(active.id);
      setConversation(active);
      await refresh();
    })();

    /**
     * A scripted turn from the demonstrator.
     *
     * The conversation it should go into is decided *here* and handed to
     * `sendTo` explicitly. It used to call `send`, whose captured
     * `conversation` was the mount-time `null` — so a titled event created its
     * conversation, and then `send` created a second one and put the turn
     * there. The titled thread the person was shown stayed empty.
     */
    const onTrigger = (event: Event) => {
      const detail = (event as CustomEvent<{
        prompt: string;
        title?: string;
        scenarioInstructions?: string;
        classification?: Classification;
        /** The skill this scenario asks the run to load. */
        skillId?: string;
        /** The scenario's checked-in documents, as bytes. */
        attachments?: ComposerAttachment[];
      }>).detail;
      if (!detail || !detail.prompt) return;
      void (async () => {
        const { conversation: target, created } = await targetForTrigger({
          title: detail.title,
          current: conversationRef.current,
          createConversation: (title) => agentService.createConversation(title),
        });
        if (created && target) {
          rememberConversation(target.id);
          setConversation(target);
          await refresh();
        }
        await sendToRef.current(target, detail.prompt, detail.classification, {
          scenarioInstructions: detail.scenarioInstructions,
          // The documents the scenario says are attached. Without these the
          // run was asked to cross-reference a drawing it had never been
          // given.
          attachments: detail.attachments,
        });
      })();
    };
    window.addEventListener('arjun:trigger-send', onTrigger);

    return () => {
      cancelled = true;
      window.removeEventListener('arjun:trigger-send', onTrigger);
    };
    // Mount-only by design. Everything it needs at fire time is read through a
    // ref, so there is nothing here that could go stale — which is the point.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Drop streaming content entries for messages that no longer exist in
  // the conversation. When a conversation is reloaded (e.g. after a
  // remount) the persisted message set may have shrunk; keeping stale
  // streaming entries around would let the wrong cell render.
  useEffect(() => {
    if (!conversation) return;
    setStreamingContents((prev) => {
      if (prev.size === 0) return prev;
      const liveIds = new Set(
        conversation.messages
          .filter((m) => m.status === 'streaming')
          .map((m) => m.id),
      );
      let changed = false;
      const next = new Map<string, string>();
      for (const [id, content] of prev) {
        if (liveIds.has(id)) {
          next.set(id, content);
        } else {
          changed = true;
        }
      }
      return changed ? next : prev;
    });
    // Progress is pruned on a different rule from content: a step list stays
    // after its turn finishes, because "what did it spend forty seconds on"
    // is a question asked *after* the answer arrives, not during. What it
    // does not survive is leaving the conversation — the durable account of
    // a past run is the task record behind "View details", not a list this
    // session happened to still be holding.
    setProgressByMessage((prev) => {
      if (prev.size === 0) return prev;
      const known = new Set(conversation.messages.map((m) => m.id));
      let changed = false;
      const next = new Map<string, ProgressStep[]>();
      for (const [id, steps] of prev) {
        if (known.has(id)) next.set(id, steps);
        else changed = true;
      }
      return changed ? next : prev;
    });
  }, [conversation]);

  const value = useMemo<UseConversation>(
    () => ({
      conversation,
      conversations,
      isStreaming,
      activeMessageId,
      activeRunId,
      streamingContents,
      progressByMessage,
      reasoningByMessage,
      send,
      open,
      newConversation,
      refresh,
      complete,
      replay,
    }),
    [
      conversation,
      conversations,
      isStreaming,
      activeMessageId,
      activeRunId,
      streamingContents,
      progressByMessage,
      reasoningByMessage,
      send,
      open,
      newConversation,
      refresh,
      complete,
      replay,
    ],
  );

  return (
    <ConversationContext.Provider value={value}>
      {children}
    </ConversationContext.Provider>
  );
}

/**
 * Read the shared conversation state.
 *
 * Every component in the tree that needs to know the active conversation
 * or its streaming cell should call this hook. The hook is intentionally
 * context-backed so the header (rendered by the app shell) and the
 * message list (rendered by the chat surface) read the SAME state.
 */
export function useConversation(): UseConversation {
  const ctx = useContext(ConversationContext);
  if (!ctx) {
    throw new Error(
      'useConversation must be used within a ConversationProvider',
    );
  }
  return ctx;
}
