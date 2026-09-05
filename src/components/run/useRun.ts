import { useCallback, useEffect, useRef, useState } from 'react';
import { labelForTool } from '../../services/toolNames';
import {
  agentService,
  type AgentEvent,
  type Classification,
  type ComposerAttachment,
  type DurableEvent,
  type RunSummary,
} from '../../services/agent.service';
import {
  applyDurableEvent,
  applyLiveEvent,
  fromSnapshot,
  receive,
  IDLE,
  type Activity,
  type RunPhase,
  type RunViewState,
} from './recovery';

/**
 * Driving one agent run, and holding what it has done so far.
 *
 * ## Two channels, two jobs
 *
 * The backend publishes on both, and this hook watches both, because they
 * answer different questions:
 *
 * - `subscribe` — the loop's own progress. Immediate, fine-grained, and
 *   best-effort: the backend drops rather than blocks, so that a slow window
 *   cannot stall a run. Good for showing a tool starting the moment it starts.
 * - `subscribeDurable` — rows that are on disk, each carrying its sequence
 *   number. Slower by a write, and *checkable*: a number that jumps means
 *   something never arrived.
 *
 * Taking responsiveness from one and correctness from the other is the whole
 * design. Watching only the first gives a trace that can silently be missing a
 * line; watching only the second makes the screen feel a beat behind.
 *
 * ## Why the hook reattaches on mount
 *
 * Both channels only exist while this component is mounted. Before the durable
 * record, a remount lost the trace outright and a restart lost the knowledge
 * that the run had ever happened — the run itself carried on in the backend
 * with nobody watching, and there was no way to find it again.
 *
 * So on mount the hook asks the record two questions: is there a run still
 * going that this person started, and — if the browser kept a run id from
 * before — what did it end up doing.
 */

export type { Activity, RunPhase, RunViewState };

/** Where the last run id is kept across a remount.
 *
 *  Session storage rather than a module variable: a module variable is lost on
 *  a full reload, which is the case that matters most. It holds an identifier
 *  and nothing else — the state it names is on the backend. */
const LAST_RUN_KEY = 'arjun.run.last';

function rememberRun(runId: string | null) {
  try {
    if (runId) sessionStorage.setItem(LAST_RUN_KEY, runId);
    else sessionStorage.removeItem(LAST_RUN_KEY);
  } catch {
    // A browser with storage disabled loses reattachment across a reload and
    // nothing else. Not worth failing a run over.
  }
}

function lastRun(): string | null {
  try {
    return sessionStorage.getItem(LAST_RUN_KEY);
  } catch {
    return null;
  }
}

/**
 * How a tool name reads in the trace.
 *
 * Re-exported rather than defined here. The table used to live in this file
 * *and* in `AssistantMessageCell.tsx`, both keyed on the pre-namespace
 * spelling, while live events carry the current one — so every row displayed
 * the raw wire name. See `services/toolNames.ts`.
 */
export const labelFor = labelForTool;

export function useRun() {
  const [state, setState] = useState<RunViewState>(IDLE);

  /** Our run's id, for filtering events. In a ref because the subscriber
   *  closure is created once and has to see the current value rather than the
   *  one that existed when it was made. */
  const runIdRef = useRef<string | null>(null);
  const correlationRef = useRef<string | null>(null);
  /** The last durable sequence number applied. Mirrors `state.seq`, in a ref so
   *  the subscriber can compare against it without being recreated. */
  const seqRef = useRef(0);
  /** Set while a reconciliation is in flight, so a burst of out-of-order events
   *  triggers one snapshot fetch rather than one per event. */
  const reconcilingRef = useRef(false);

  const reset = useCallback(() => {
    runIdRef.current = null;
    correlationRef.current = null;
    seqRef.current = 0;
    rememberRun(null);
    setState(IDLE);
  }, []);

  /**
   * Reads a run back off the record and adopts it.
   *
   * The snapshot first, then the events after it — rather than every event —
   * because the snapshot exists precisely so that opening a screen does not
   * mean replaying a run's whole history.
   */
  const adopt = useCallback(async (runId: string) => {
    const snapshot = await agentService.snapshot(runId).catch(() => null);
    if (!snapshot) return null;

    let recovered = fromSnapshot(snapshot);
    try {
      const page = await agentService.events(runId, snapshot.seq);
      for (const event of page.events) {
        recovered = applyDurableEvent(recovered, {
          runId: event.runId,
          seq: event.seq,
          eventId: event.eventId,
          eventType: event.eventType,
          at: event.at,
          actor: event.actor,
          schemaVersion: event.schemaVersion,
          payload: event.payload,
        });
      }
      if (page.unreadable.length > 0) recovered = { ...recovered, historyIncomplete: true };
    } catch {
      // The snapshot alone is still a true account of the run up to its own
      // sequence number, and saying so is better than showing nothing.
    }

    runIdRef.current = runId;
    seqRef.current = recovered.seq;
    rememberRun(runId);
    setState(recovered);
    return recovered;
  }, []);

  /**
   * Repairs the view after a gap in the durable sequence.
   *
   * Debounced through a ref rather than a timer: several out-of-order events
   * usually arrive together, and each one would otherwise start its own
   * snapshot fetch. One fetch answers all of them, because the snapshot is the
   * authority regardless of how many events were missed.
   */
  const reconcile = useCallback(
    async (runId: string) => {
      if (reconcilingRef.current) return;
      reconcilingRef.current = true;
      try {
        await adopt(runId);
      } finally {
        reconcilingRef.current = false;
      }
    },
    [adopt],
  );

  /** The loop's own progress. Applied for responsiveness, never for counting. */
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    void agentService
      .subscribe(({ runId, event }: { runId: string; event: AgentEvent }) => {
        // Lock onto our run the first time it identifies itself, then ignore
        // everything else. Without this, a second window's run would write
        // into this one's trace.
        if (
          runIdRef.current === null &&
          event.type === 'plan_ready' &&
          event.correlationId &&
          event.correlationId === correlationRef.current
        ) {
          runIdRef.current = runId;
          rememberRun(runId);
          setState(previous => ({ ...previous, runId }));
        }
        if (runIdRef.current !== runId) return;
        setState(previous => applyLiveEvent(previous, event));
      })
      .then(fn => {
        // Unmounted before the listener was registered: tear it down at once
        // rather than leave it updating a component that is gone.
        if (cancelled) fn();
        else unlisten = fn;
      });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  /** The durable history. Applied in order, and checked for gaps. */
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;

    void agentService
      .subscribeDurable((event: DurableEvent) => {
        if (runIdRef.current !== event.runId) return;

        switch (receive(seqRef.current, event.seq).action) {
          case 'ignore':
            // A reconnect re-sending what we already had.
            return;
          case 'reconcile':
            // At least one event never arrived. Applying this one on top would
            // produce a state that quietly disagrees with the record, so the
            // record is asked instead.
            void reconcile(event.runId);
            return;
          case 'apply':
            seqRef.current = event.seq;
            setState(previous => applyDurableEvent(previous, event));
        }
      })
      .then(fn => {
        if (cancelled) fn();
        else unlisten = fn;
      });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [reconcile]);

  /**
   * Finds a run to reattach to, once, on mount.
   *
   * A run still going wins over the last one this window saw: it is the one
   * with something left to watch. Failing that, the remembered id is looked up
   * so a reload lands on the run's outcome rather than on an empty composer.
   */
  useEffect(() => {
    let cancelled = false;

    void (async () => {
      // Already ours — a remount inside one session, with the ref intact.
      if (runIdRef.current) return;

      const live = await agentService.activeTasks().catch(() => []);
      const candidate = live[0]?.runId ?? lastRun();
      if (!candidate || cancelled) return;
      // The person started something while this was in flight. `start` sets the
      // correlation id synchronously before its own await, so it is the signal
      // that exists this early — without it, a lookup begun on mount would
      // land afterwards and replace the run they just asked for.
      if (runIdRef.current || correlationRef.current) return;

      const recovered = await adopt(candidate);
      // The remembered id pointed at a run the record has since forgotten —
      // a reset database, or a task from a machine this profile has moved off.
      if (!recovered) rememberRun(null);
    })();

    return () => {
      cancelled = true;
    };
    // Once, on mount. `adopt` is stable.
  }, [adopt]);

  const start = useCallback(
    async (
      prompt: string,
      classification?: Classification,
      options?: {
        /** Caller-supplied correlation id. Used by the demo page so its
         *  events are not misattributed to another window's run. */
        correlationId?: string;
        /**
         * Extra framing for a scripted scenario, appended beneath ARJUN's own
         * instructions rather than replacing them.
         */
        scenarioInstructions?: string;
        /**
         * Documents this run is given.
         *
         * The demonstrator's scenarios say their documents are "attached";
         * until this existed nothing was, so a scenario asked the model to
         * cross-reference a drawing it had never been given.
         */
        attachments?: ComposerAttachment[];
      },
    ) => {
      const correlationId = options?.correlationId ?? crypto.randomUUID();
      correlationRef.current = correlationId;
      runIdRef.current = null;
      seqRef.current = 0;
      rememberRun(null);

      setState({ ...IDLE, phase: 'starting', prompt });

      try {
        // No `conversationId` and no `messageId`: this caller has no chat cell
        // to reserve. The backend settles both before the run starts and
        // returns them on the summary — see `resolve_turn_identity`. Sending
        // `messageId: null` used to reach the runtime as a malformed request,
        // so every run started from here failed before a model was asked
        // anything.
        const summary = await agentService.start({
          prompt,
          classification,
          correlationId,
          scenarioInstructions: options?.scenarioInstructions,
          attachments: options?.attachments,
        });
        // The summary is complete where the event stream is best-effort, so it
        // wins: the plan it carries is the one that was actually enforced.
        rememberRun(summary.runId);
        setState(previous => ({
          ...previous,
          phase: summary.outcome?.kind === 'paused' ? 'paused' : 'finished',
          state: summary.outcome?.kind === 'paused' ? 'paused' : 'completed',
          runId: summary.runId,
          plan: summary.plan,
          turns: summary.turns,
          summary,
        }));
        return summary;
      } catch (error) {
        // The run is over and the record knows how it ended — stopped by a
        // person, out of budget, refused by policy, or genuinely broken — and
        // those read very differently. Ask, rather than painting all of them as
        // a failure.
        const message = error instanceof Error ? error.message : String(error);
        const runId = runIdRef.current;
        if (runId) {
          const recovered = await adopt(runId).catch(() => null);
          if (recovered) return null;
        }
        setState(previous => ({ ...previous, phase: 'failed', state: 'failed', error: message }));
        return null;
      }
    },
    [adopt],
  );

  const abort = useCallback(async () => {
    const runId = runIdRef.current;
    if (!runId) return;
    // A run that finished just before the button was pressed resolves `false`.
    // That is an ordinary race rather than a failure, so nothing is surfaced.
    await agentService.abort(runId).catch(() => undefined);
  }, []);

  const steer = useCallback(async (text: string) => {
    const runId = runIdRef.current;
    if (!runId || !text.trim()) return false;
    setState(previous => ({ ...previous, steering: true }));
    try {
      return await agentService.steer(runId, text);
    } catch {
      return false;
    } finally {
      setState(previous => ({ ...previous, steering: false }));
    }
  }, []);

  // `adopt` is returned so a provider can follow a run it did not start --
  // the chat surface issues its own run id before the backend replies, and the
  // dashboard has to be able to show that run rather than a different one.
  return { state, start, abort, steer, reset, adopt };
}

/** Whether a run is still going, for disabling the composer. */
export const isBusy = (phase: RunPhase) => phase === 'starting' || phase === 'running';

/** Whether a `RunSummary` is present, for callers narrowing the state. */
export const hasSummary = (
  state: RunViewState,
): state is RunViewState & { summary: RunSummary } => state.summary !== null;
