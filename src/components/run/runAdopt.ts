import { useEffect, useRef, useState } from 'react';
import {
  agentService,
  listenAttachmentContext,
  type AgentEventEnvelope,
  type ArtifactReport,
  type AttachmentContextEvent,
  type CompactionRecord,
  type ContextLedgerRecord,
  type RunSummary,
} from '../../services/agent.service';
import {
  applyDurableEvent,
  applyLiveEvent,
  fromSnapshot,
  receive,
  IDLE,
  type Activity,
  type RunViewState,
} from './recovery';
import { liveCompaction, mergeCompaction } from './context-ledger';

/**
 * Adopt a single run by id, without going through `useRun`.
 *
 * The chat surface needs the activity list and the `RunViewState` for
 * one run at a time, and spinning up the full `useRun` for that would
 * also subscribe to durable/snapshot reconciliation for the *last* run
 * the user started. This module does the minimum: read the snapshot,
 * apply any events after it, subscribe to the live + durable channels
 * while the run is in flight, and hand the resulting `RunViewState`
 * back to the caller.
 */

export interface AdoptedRun {
  view: RunViewState;
  activity: Activity[];
}

export async function adoptRun(
  runId: string,
  onUpdate?: (next: AdoptedRun) => void,
): Promise<AdoptedRun | null> {
  const snapshot = await agentService.snapshot(runId).catch(() => null);
  if (!snapshot) return null;

  let state: RunViewState = fromSnapshot(snapshot);
  let activity: Activity[] = state.activity;
  const emit = () => onUpdate?.({ view: state, activity });
  emit();

  try {
    const page = await agentService.events(runId, snapshot.seq);
    for (const event of page.events) {
      state = applyDurableEvent(state, event);
      activity = state.activity;
    }
    emit();
  } catch {
    // Snapshot alone is fine.
  }

  // Live + durable subscribers. These do not block adoption: the run
  // may have finished before the chat surface opens the inspector,
  // and in that case the subscribers just no-op.
  const live = await agentService.subscribe(
    ({ runId: r, event }: AgentEventEnvelope) => {
      if (r !== runId) return;
      state = applyLiveEvent(state, event);
      activity = state.activity;
      emit();
    },
    runId,
  );
  const durable = await agentService.subscribeDurable(event => {
    if (event.runId !== runId) return;
    if (receive(state.seq, event.seq).action !== 'apply') return;
    state = applyDurableEvent(state, event);
    activity = state.activity;
    emit();
  }, runId);

  // The caller is expected to call this when unmounting. Returning the
  // teardown so the chat surface can wire it to its own effect.
  (state as unknown as { _adoptTeardown?: () => void })._adoptTeardown = () => {
    live();
    durable();
  };

  return { view: state, activity };
}

/**
 * React hook wrapper around `adoptRun`. Returns `null` when no run is
 * being adopted so the caller can render a placeholder cheaply.
 */
export function useAdoptedRun(runId: string | null) {
  const [state, setState] = useState<AdoptedRun | null>(null);

  useEffect(() => {
    if (!runId) {
      setState(null);
      return;
    }
    let cancelled = false;
    let teardown: (() => void) | null = null;
    void (async () => {
      const adopted = await adoptRun(runId, next => {
        if (cancelled) return;
        setState(next);
      });
      if (cancelled) return;
      if (adopted) {
        setState(adopted);
        teardown = (adopted.view as unknown as { _adoptTeardown?: () => void })
          ._adoptTeardown ?? null;
      }
    })();
    return () => {
      cancelled = true;
      teardown?.();
    };
  }, [runId]);

  return state;
}

/**
 * Activity for every run in a conversation, keyed by run id.
 *
 * The chat surface needs each assistant cell to show what its own run
 * actually did, which is more than `useAdoptedRun` was built for. Two
 * different costs are involved, so this hook pays them differently:
 *
 *  - The run in flight is adopted in full (snapshot, catch-up events,
 *    then live + durable subscriptions) so its rows appear as the tools
 *    run.
 *  - Runs that already finished are read once from their snapshot. No
 *    subscriptions, and `fetched` makes it once per run for the life of
 *    the surface rather than once per render.
 *
 * A snapshot that fails to load leaves the run's entry alone instead of
 * writing an empty list, so a transient backend error shows the rows we
 * already had rather than blanking the turn.
 */
export function useConversationActivity(
  runIds: string[],
  liveRunId: string | null,
): Map<string, Activity[]> {
  const [byRun, setByRun] = useState<Map<string, Activity[]>>(new Map());
  const fetched = useRef<Set<string>>(new Set());

  // `runIds` is a fresh array every render; the joined key is what
  // actually changes when the conversation gains or loses a run.
  const runKey = runIds.join(',');

  useEffect(() => {
    const pending = runIds.filter(
      id => id !== liveRunId && !fetched.current.has(id),
    );
    if (pending.length === 0) return;
    for (const id of pending) fetched.current.add(id);

    let cancelled = false;
    void (async () => {
      const results = await Promise.all(
        pending.map(async id => {
          const snapshot = await agentService.snapshot(id).catch(() => null);
          return [id, snapshot ? fromSnapshot(snapshot).activity : null] as const;
        }),
      );
      if (cancelled) return;
      setByRun(prev => {
        const next = new Map(prev);
        let changed = false;
        for (const [id, activity] of results) {
          if (!activity) continue;
          next.set(id, activity);
          changed = true;
        }
        return changed ? next : prev;
      });
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey, liveRunId]);

  useEffect(() => {
    if (!liveRunId) return;
    let cancelled = false;
    let teardown: (() => void) | null = null;
    void (async () => {
      const adopted = await adoptRun(liveRunId, next => {
        if (cancelled) return;
        setByRun(prev => new Map(prev).set(liveRunId, next.activity));
      });
      if (cancelled) return;
      if (adopted) {
        setByRun(prev => new Map(prev).set(liveRunId, adopted.activity));
        teardown =
          (adopted.view as unknown as { _adoptTeardown?: () => void })
            ._adoptTeardown ?? null;
      }
    })();
    return () => {
      cancelled = true;
      teardown?.();
    };
  }, [liveRunId]);

  return byRun;
}

/**
 * The files every run in a conversation produced, keyed by run id.
 *
 * ## Why this exists rather than reading `runSummary`
 *
 * The chat cell already had an artifact list, and it drew nothing. It was
 * conditioned on `runSummary`, which `ChatSurface` supplies only for the run
 * whose inspector is open:
 *
 * ```tsx
 * runSummary={
 *   inspectorRunId && runsByMessageId.get(m.id) === inspectorRunId
 *     ? taskSummary ?? null : null
 * }
 * ```
 *
 * So a produced file could only ever appear *after* somebody clicked "View
 * details" on that particular message — and a person who has just been told
 * "the PDF is saved as sum-of-2-numbers.pdf" has no reason to go hunting in an
 * inspector for it. The deliverable was on disk, recorded, and unreachable
 * without a click nothing prompted.
 *
 * Artifacts are not inspector detail. They are part of the answer, the same way
 * the reply text is, so they are fetched for every run in the conversation
 * independently of whether anything is being inspected.
 *
 * ## Shaped after `useConversationActivity`
 *
 * The same two costs and the same treatment: the live run is skipped, because
 * its record is not written until it ends, and each finished run is read once
 * for the life of the surface rather than once per render. When the live run
 * ends `liveRunId` becomes null, the effect runs again, and the run that was
 * live is picked up on that pass — which is what makes a file appear as soon as
 * the turn finishes.
 *
 * A failed read leaves the run's entry alone rather than writing an empty list,
 * so a transient backend error shows the rows we already had instead of
 * blanking them.
 */
export function useRunArtifacts(
  runIds: string[],
  liveRunId: string | null,
): Map<string, ArtifactReport[]> {
  const [byRun, setByRun] = useState<Map<string, ArtifactReport[]>>(new Map());
  const fetched = useRef<Set<string>>(new Set());

  const runKey = runIds.join(',');

  useEffect(() => {
    const pending = runIds.filter(id => id !== liveRunId && !fetched.current.has(id));
    if (pending.length === 0) return;
    for (const id of pending) fetched.current.add(id);

    let cancelled = false;
    void (async () => {
      const results = await Promise.all(
        pending.map(
          async id =>
            [id, await agentService.taskArtifacts(id).catch(() => null)] as const,
        ),
      );
      if (cancelled) return;
      setByRun(prev => {
        const next = new Map(prev);
        let changed = false;
        for (const [id, artifacts] of results) {
          // Null is "the read failed"; an empty array is "this run produced
          // nothing", and each is recorded as itself. Storing the empty list
          // matters: it is what distinguishes the two on a later render.
          if (!artifacts) continue;
          next.set(id, artifacts);
          changed = true;
        }
        return changed ? next : prev;
      });
    })();
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runKey, liveRunId]);

  return byRun;
}

/**
 * Fetch the `TaskRecord` for one run, in a hook. Used by the inspector
 * to read the final answer + plan + verification + artifacts that
 * `RunView` shows.
 */
export function useTaskRecord(runId: string | null) {
  const [record, setRecord] = useState<RunSummary | null>(null);
  useEffect(() => {
    if (!runId) {
      setRecord(null);
      return;
    }
    let cancelled = false;
    void (async () => {
      try {
        const task = await agentService.task(runId);
        if (cancelled) return;
        setRecord({
          runId: task.runId,
          text: task.answer,
          turns: task.turns,
          // Records written before the typed ending existed carry only the
          // failure sentence. Read back as `failed` when there is one and
          // `completed` when there is not — which is what the record actually
          // says, rather than a state it never recorded.
          outcome:
            task.outcome ??
            (task.failure
              ? { kind: 'failed', detail: task.failure }
              : { kind: 'completed' }),
          routing: task.routing,
          endpoint: task.endpoint,
          plan: task.plan,
          verification: task.verification,
          artifacts: task.artifacts,
          // The record was read back off disk, so the store this run needed
          // was working. That is a fact about the read that just succeeded,
          // not an assumption about the installation now.
          audit: { state: 'durable' },
        });
      } catch {
        if (!cancelled) setRecord(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [runId]);
  return record;
}

/**
 * How much the meter knows, and whether it is still finding out.
 *
 * Three states the surface used to render identically, as one grey chip
 * reading "No context yet":
 *
 * - `loading` — the stored reading has been asked for and has not come back.
 * - `empty` — it came back, and there is no reading yet. The ordinary state of
 *   every run for its first few seconds, and of a conversation nobody has sent
 *   a turn in.
 * - `failed` — the read genuinely failed. The meter is showing nothing and can
 *   say why.
 *
 * Told apart because the right response differs: wait, send a message, or
 * report a fault. A person watching a meter that never filled could not tell
 * which of the three they were looking at.
 */
export type ContextLedgerStatus = 'idle' | 'loading' | 'ready' | 'empty' | 'failed';

export interface ContextLedgerView {
  ledger: ContextLedgerRecord | null;
  compactions: CompactionRecord[];
  attachments: AttachmentContextEvent[];
  status: ContextLedgerStatus;
  /** The sentence to show when `status` is `failed`. Never model output. */
  error: string | null;
}

/**
 * Read the context ledger, the compactions and the attachment costs for a turn.
 *
 * ## Two keys, and why it is not one
 *
 * `runId` is not stable across a turn. The composer mints a correlation id
 * before the request goes out, the run reports its own id on `plan_ready`, and
 * the surface reconciles to the real one the moment it arrives. So the id this
 * hook is given *changes mid-turn*, by design.
 *
 * The ledger and the compactions belong to the run, are stamped with the run's
 * own id, and are correctly re-fetched and re-subscribed when it changes — no
 * ledger event is ever emitted under the correlation id, because the runtime is
 * the only thing that emits them and it does not exist until the run does.
 *
 * The attachment costs belong to the *turn*. They are published before the run
 * has an id at all, which is the whole point of them: a document's price is
 * known while the OCR model is still finishing, and the meter should show it
 * then. Keying those on `runId` would throw them away at the exact moment the
 * id was reconciled — the correlation-id subscription that received them torn
 * down and its state cleared, seconds after they arrived.
 *
 * So attachments are keyed on `messageId`, which the surface reserves before
 * the turn starts and which nothing changes for its whole life.
 *
 * ## What each effect clears
 *
 * Both clear their own state when their own key changes. That is the fix for
 * the retention: switching runs, or conversations, used to leave the previous
 * run's ledger, compactions and attachments on screen until new ones happened
 * to arrive — and because the stored fetch merged with `current ??`, the *new*
 * run's stored reading was then discarded in favour of the old run's. The meter
 * showed one run's numbers under another run's name, indefinitely.
 *
 * ## The race
 *
 * The stored fetch and the live subscription both write the ledger, and the
 * fetch can land after an event describing a later moment. `liveArrived`
 * settles it: the fetch applies only while nothing newer has arrived, and it is
 * scoped to this effect run so it cannot leak across a key change.
 */
export function useContextLedger(
  runId: string | null,
  /**
   * The assistant cell this turn is streaming into.
   *
   * Scopes the attachment events, which are published on an application-wide
   * channel: without it the meter folded in every document any run read,
   * another window's included. Omitted for a finished run opened from the Tasks
   * screen, which has no live attachments to receive — the stored ledger
   * carries its documents as entities.
   */
  messageId?: string | null,
): ContextLedgerView {
  const [ledger, setLedger] = useState<ContextLedgerRecord | null>(null);
  const [compactions, setCompactions] = useState<CompactionRecord[]>([]);
  const [status, setStatus] = useState<ContextLedgerStatus>('idle');
  const [error, setError] = useState<string | null>(null);
  /**
   * Per-attachment costs, keyed by content hash.
   *
   * Held beside the ledger rather than inside it because the two arrive from
   * different places at different times: an attachment's cost is known while
   * the OCR model is still finishing and the run has not made a model call yet,
   * so there is no ledger to put it in.
   */
  const [attachments, setAttachments] = useState<AttachmentContextEvent[]>([]);

  // ── The run's own readings: ledger and compactions ────────────────────
  useEffect(() => {
    // Cleared on every key change, including to null. This is the retention
    // fix: nothing from the previous run survives into the next one's panel,
    // not even for the moment before its first event lands.
    setLedger(null);
    setCompactions([]);
    setError(null);

    if (!runId) {
      setStatus('idle');
      return;
    }
    setStatus('loading');

    let cancelled = false;
    // Set by the first live event. A stored reading describes an earlier moment
    // than any event that has already arrived, so once one has, the in-flight
    // fetch must not be allowed to write over it.
    let liveArrived = false;
    const unsubscribers: (() => void)[] = [];

    // Subscribed first, so the window in which an event can be missed is as
    // short as this side can make it. It is not zero: registering a Tauri
    // listener is itself asynchronous, so an event emitted in the next few
    // milliseconds reaches nobody. That is why the stored fetch below is not
    // merely a nicety for finished runs — it is also the backstop that fills in
    // whatever the subscription was too late for.
    void agentService
      .subscribe(({ event }: AgentEventEnvelope) => {
        if (cancelled) return;
        if (event.type === 'context_ledger') {
          liveArrived = true;
          setLedger(event.ledger);
          setStatus('ready');
          return;
        }
        if (event.type === 'context_compacted') {
          const record = liveCompaction(event);
          if (!record) return;
          liveArrived = true;
          setCompactions(current => mergeCompaction(current, record));
          // A compaction carries the ledger as it stood afterwards, so the
          // meter moves with it rather than waiting for the next turn's
          // reading — which is a whole model call away.
          setLedger(record.ledger);
          setStatus('ready');
        }
      }, runId)
      .then(un => {
        if (cancelled) un();
        else unsubscribers.push(un);
      });

    // The stored reading, so a finished run opened from the Tasks screen shows
    // its ledger without waiting for events that will never come.
    void (async () => {
      try {
        const snapshot = await agentService.taskContext(runId);
        if (cancelled) return;
        if (!snapshot) {
          // No record yet. Not an error: it is the ordinary state of a run in
          // its first seconds, and of every run that has not made a model call.
          setStatus(current => (current === 'loading' ? 'empty' : current));
          return;
        }
        // Applied only where nothing newer has landed. A live event describes a
        // later moment than this reply, which was in flight while it arrived.
        if (!liveArrived) {
          setLedger(snapshot.ledger ?? null);
        }
        setCompactions(current =>
          (snapshot.compactions ?? []).reduce(mergeCompaction, current),
        );
        setStatus(current =>
          current === 'ready' ||
          snapshot.ledger ||
          (snapshot.compactions ?? []).length > 0
            ? 'ready'
            : 'empty',
        );
      } catch (cause) {
        if (cancelled) return;
        // A real failure, distinguished from "no record yet" by the backend:
        // `agent_task_context` answers `null` for that and rejects only when
        // the read genuinely went wrong. A live event that has already arrived
        // outranks it — the meter is working, whatever the stored copy did.
        if (liveArrived) return;
        setError(cause instanceof Error ? cause.message : String(cause));
        setStatus('failed');
      }
    })();

    return () => {
      cancelled = true;
      for (const un of unsubscribers) un();
    };
  }, [runId]);

  // ── The turn's attachment costs ───────────────────────────────────────
  //
  // A separate effect with a separate key, so the run-id reconciliation that
  // re-runs the effect above does not discard documents that arrived under the
  // correlation id seconds earlier.
  useEffect(() => {
    setAttachments([]);
    if (!messageId) return;

    let cancelled = false;
    let unlisten: (() => void) | null = null;

    void listenAttachmentContext(payload => {
      if (cancelled) return;
      // Scoped to this turn. The channel is application-wide, so without this
      // the meter folded in every document any run read — another window's
      // included. An event naming no message reaches nobody, which is the safe
      // direction to fail.
      if (payload.messageId !== messageId) return;
      setAttachments(current => {
        // Keyed by content hash, so re-reading the same file replaces its row
        // rather than adding a second one for the same document.
        const at = current.findIndex(a => a.sha256 === payload.sha256);
        if (at === -1) return [...current, payload];
        const next = current.slice();
        next[at] = payload;
        return next;
      });
    }).then(un => {
      if (cancelled) un();
      else unlisten = un;
    });

    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [messageId]);

  return { ledger, compactions, attachments, status, error };
}

/**
 * Re-export the `IDLE` state so a chat surface can use the same
 * defaults as `useRun` without reaching into the reducer module.
 */
export { IDLE };
