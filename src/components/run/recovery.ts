import {
  isTerminal,
  type ActivityRecord,
  type AgentEvent,
  type DurableEvent,
  type PlanRecord,
  type RunState,
  type RunSummary,
  type TaskSnapshot,
  type UnknownEffect,
} from '../../services/agent.service';

/**
 * Rebuilding what a run has been doing, from whichever account is available.
 *
 * There are two, and they are not interchangeable:
 *
 * - **The live stream** (`AgentEvent`) is best-effort. The backend emits it so
 *   that a slow listener cannot stall a run, which is the right trade and means
 *   a line can go missing. It carries no sequence number, so a lost message and
 *   a quiet run are indistinguishable.
 * - **The durable stream** (`DurableEvent`) names rows that are on disk, in
 *   order, each with its sequence number. A gap in those numbers is detectable,
 *   and detecting it is what lets a window repair itself rather than drawing a
 *   trace with a hole in it.
 *
 * Recovery is therefore: read the snapshot, apply the durable events after it,
 * and keep watching the sequence. When it jumps, go back to the snapshot.
 *
 * These are plain functions over plain state rather than hook internals so that
 * the part with the actual logic in it can be tested without mounting anything.
 */

/** One thing the run did, in the order it did it. */
export interface Activity {
  id: string;
  tool: string;
  /**
   * `running` until the call comes back. `replayed` means the side effect had
   * already happened and was not performed a second time. `unknown` means it
   * was in flight when the process went away and nobody can say whether it
   * took — that one needs a person.
   */
  status: 'running' | 'done' | 'failed' | 'refused' | 'replayed' | 'unknown';
  /** Wall-clock time the tool started, ms since epoch. */
  startedAt?: number;
  /** Wall-clock time the tool ended, ms since epoch. Set when status is terminal. */
  endedAt?: number;
  /**
   * One-line description of what the tool was asked to do, in human terms.
   *
   * The backend redacts the raw arguments before they leave the
   * `agent://event` channel (they can carry document text); what arrives
   * here is a single sentence composed in Rust from the call's
   * shape. Empty when the tool author did not supply one.
   */
  inputSummary?: string;
  /**
   * One-line description of what the tool returned, in human terms.
   *
   * Same redaction policy as `inputSummary`. Empty for tools whose
   * outcome is fully captured by the artifact list or the trace.
   */
  outputSummary?: string;
  /**
   * Path of a file this tool produced or touched. Carried so the row
   * can link the user to the artifact without a second lookup.
   */
  artifactPath?: string;
  /** Human-readable failure message when status is `failed`. */
  errorMessage?: string;
}

export type RunPhase =
  | 'paused'
  | 'recovering'
  | 'idle'
  | 'starting'
  | 'running'
  | 'finished'
  | 'failed'
  /** A milestone the model finished. The plan pauses here so a
   *  person can confirm the model is on the right track before the
   *  next leg of work starts. The UI shows a gate; once the user
   *  approves, the phase returns to `running`. */
  | 'awaiting_milestone';

export interface RunState_ {
  phase: RunPhase;
}

export interface RunViewState {
  phase: RunPhase;
  prompt: string;
  runId: string | null;
  plan: PlanRecord | null;
  activity: Activity[];
  /** Set when the plan stopped the run before the loop was done. */
  stopped: string | null;
  /** Times the context was summarised so the run could continue. */
  compactions: number;
  turns: number;
  summary: RunSummary | null;
  error: string | null;
  /** True while a correction is being applied, so the control can say so. */
  steering: boolean;
  /**
   * True when this state was read back from the record rather than watched.
   *
   * Surfaced rather than hidden. A trace recovered after a restart is missing
   * whatever the live stream would have shown between the recorded events, and
   * a person reading it should know they are looking at a reconstruction.
   */
  recovered: boolean;
  /** Where the run is, once the record has been consulted. */
  state: RunState | null;
  /** The last durable event folded in. Ask for events after this to catch up. */
  seq: number;
  /** True when part of the history could not be read. */
  historyIncomplete: boolean;
  /** Side effects nobody can account for. Each needs a person to go and look. */
  unknownEffects: UnknownEffect[];
  /** Set when the run paused at a milestone checkpoint. The gate
   *  holds the id and intent the model produced; the user can
   *  approve to continue or reject to stop the run. */
  milestone: MilestoneGate | null;
}

/** A milestone the model just finished, awaiting a human gate. */
export interface MilestoneGate {
  checkpointId: string;
  ordinal: number;
  summary: string;
}

/** Kept for callers that imported the old name. */
export type { RunViewState as RunState2 };

export const IDLE: RunViewState = {
  phase: 'idle',
  prompt: '',
  runId: null,
  plan: null,
  activity: [],
  stopped: null,
  compactions: 0,
  turns: 0,
  summary: null,
  error: null,
  steering: false,
  recovered: false,
  state: null,
  seq: 0,
  historyIncomplete: false,
  unknownEffects: [],
  milestone: null,
};

/** What a recorded state means for the screen. */
export function phaseFor(state: RunState): RunPhase {
  if (state === 'paused' || state === 'recovering') return state;
  if (state === 'completed') return 'finished';
  if (isTerminal(state)) {
    // Cancelled, out of budget, refused by policy and degraded are all endings
    // the person needs to see as an ending rather than as a run that quietly
    // stopped updating.
    return 'failed';
  }
  return 'running';
}

/** How a state reads when the record is all there is to go on. */
export function describe(state: RunState): string {
  switch (state) {
    case 'paused': return 'Paused at a saved boundary. Resume to continue.';
    case 'recovering': return 'Interrupted. Ready for a recovery check by the original operator.';
    case 'stopped_by_length': return 'Stopped at the model output limit.';
    case 'created':
      return 'Accepted, and not yet started.';
    case 'classified':
      return 'Working out which model should take it.';
    case 'routed':
      return 'A model has been chosen.';
    case 'planned':
      return 'The plan is fixed. Starting.';
    case 'running':
    case 'tool_result_recorded':
      return 'Running.';
    case 'awaiting_approval':
      return 'Waiting for someone to approve an action.';
    case 'executing_tool':
      return 'Running a tool.';
    case 'verifying':
      return 'Checking the answer against the evidence.';
    case 'completed':
      return 'Finished.';
    case 'cancelled':
      return 'Stopped, because somebody stopped it.';
    case 'stopped_by_budget':
      return 'Stopped: it reached the limit the plan set for it.';
    case 'stopped_by_policy':
      return 'Stopped: it needed to do something it is not permitted to do.';
    case 'degraded_needs_human':
      return 'Interrupted. Somebody needs to look at this before it is relied on.';
    case 'failed':
    default:
      return 'Stopped: it did not finish.';
  }
}

const ACTIVITY_STATUSES = new Set([
  'running',
  'done',
  'failed',
  'refused',
  'replayed',
  'unknown',
]);

function activityFrom(record: ActivityRecord): Activity {
  return {
    id: record.toolCallId,
    tool: record.tool,
    // An unrecognised status comes from a build newer than this one. Shown as
    // still running rather than dropped: the row existing is the true part.
    status: (ACTIVITY_STATUSES.has(record.status)
      ? record.status
      : 'running') as Activity['status'],
  };
}

/**
 * The state a window adopts when it finds a run it did not start.
 *
 * Everything here came off the record, so `recovered` is set and stays set.
 */
export function fromSnapshot(snapshot: TaskSnapshot): RunViewState {
  return {
    ...IDLE,
    phase: phaseFor(snapshot.state),
    prompt: snapshot.prompt,
    runId: snapshot.runId,
    plan: snapshot.plan,
    activity: snapshot.activity.map(activityFrom),
    stopped: snapshot.stoppedBecause,
    compactions: snapshot.compactions,
    turns: snapshot.turns,
    // A recovered run has no summary: the answer lives in the task record, and
    // one still going has no answer yet.
    summary: null,
    error: isTerminal(snapshot.state)
      ? snapshot.failure ?? describe(snapshot.state)
      : null,
    recovered: true,
    state: snapshot.state,
    seq: snapshot.seq,
    historyIncomplete:
      snapshot.unreadableEvents.length > 0 || snapshot.anomalies.length > 0,
    unknownEffects: snapshot.unknownEffects,
  };
}

/** Reads a string out of a redacted payload, if it survived redaction. */
function text(payload: Record<string, unknown>, key: string): string | null {
  const value = payload[key];
  return typeof value === 'string' ? value : null;
}

function count(payload: Record<string, unknown>, key: string): number | null {
  const value = payload[key];
  return typeof value === 'number' ? value : null;
}

function settle(
  state: RunViewState,
  payload: Record<string, unknown>,
  status: Activity['status'],
): RunViewState {
  const id = text(payload, 'toolCallId');
  if (!id) return state;
  const inputSummary = text(payload, 'inputSummary') ?? undefined;
  const outputSummary = text(payload, 'outputSummary') ?? undefined;
  const artifactPath = text(payload, 'artifactPath') ?? undefined;
  const errorMessage = text(payload, 'errorMessage') ?? undefined;
  const at = text(payload, 'at');
  const endedAt = at ? Date.parse(at) : Date.now();
  const update = (item: Activity): Activity => {
    const startedAt = item.startedAt ?? endedAt;
    return {
      ...item,
      status,
      startedAt,
      endedAt,
      inputSummary: inputSummary ?? item.inputSummary,
      outputSummary: outputSummary ?? item.outputSummary,
      artifactPath: artifactPath ?? item.artifactPath,
      errorMessage: errorMessage ?? item.errorMessage,
    };
  };
  const known = state.activity.some(item => item.id === id);
  return {
    ...state,
    activity: known
      ? state.activity.map(item => (item.id === id ? update(item) : item))
      : // A refusal is the first thing heard about that call — it never got as
        // far as being authorised. Dropping it would make the trace say the
        // policy never did anything.
        [
          ...state.activity,
          {
            id,
            tool: text(payload, 'tool') ?? 'unknown',
            status,
            startedAt: endedAt,
            endedAt,
            inputSummary,
            outputSummary,
            artifactPath,
            errorMessage,
          },
        ],
  };
}

/** Which state each ending event puts the run in. */
const ENDING_STATES: Partial<Record<DurableEvent['eventType'], RunState>> = {
  runStoppedByLength: 'stopped_by_length',
  runCompleted: 'completed',
  runFailed: 'failed',
  runCancelled: 'cancelled',
  runStoppedByBudget: 'stopped_by_budget',
  runStoppedByPolicy: 'stopped_by_policy',
  runDegraded: 'degraded_needs_human',
  runTimedOut: 'stopped_by_budget',
  runInterrupted: 'degraded_needs_human',
};

/**
 * Folds one durable event into the state.
 *
 * Out-of-order and already-seen events are ignored rather than rejected: a
 * window catching up after a reconnect will legitimately re-send what it
 * already had.
 */
export function applyDurableEvent(state: RunViewState, event: DurableEvent): RunViewState {
  if (event.seq <= state.seq) return state;
  const at: RunViewState = { ...state, seq: event.seq };
  const payload = event.payload;

  const ending = ENDING_STATES[event.eventType];
  if (ending) {
    return {
      ...at,
      phase: phaseFor(ending),
      state: ending,
      error: text(payload, 'failure') ?? describe(ending),
    };
  }

  switch (event.eventType) {
    case 'runPaused':
      return { ...at, phase: 'paused', state: 'paused', error: null };
    case 'recoveryStarted':
      return { ...at, phase: 'recovering', state: 'recovering', error: null };
    case 'runResumed':
      return { ...at, phase: 'running', state: 'running', error: null, summary: null };
    case 'runCreated':
      return {
        ...at,
        runId: event.runId,
        prompt: text(payload, 'promptShown') ?? at.prompt,
        phase: 'running',
        state: 'created',
      };
    case 'runClassified':
      return { ...at, state: 'classified' };
    case 'runRouted':
      return { ...at, state: 'routed' };
    case 'planReady':
      return {
        ...at,
        state: 'planned',
        plan: (payload.plan as PlanRecord | undefined) ?? at.plan,
      };
    case 'runStarted':
      return { ...at, phase: 'running', state: 'running' };
    case 'planStep':
      return at.plan
        ? {
            ...at,
            plan: {
              ...at.plan,
              stepsTaken: count(payload, 'stepsTaken') ?? at.plan.stepsTaken,
            },
          }
        : at;
    case 'planStopped':
      return { ...at, stopped: text(payload, 'reason') };
    // Counted from the record rather than incremented by the live stream, so a
    // recovered trace and a watched one report the same numbers.
    case 'turnEnded':
      return { ...at, turns: at.turns + 1 };
    case 'contextCompacted':
      return { ...at, compactions: at.compactions + 1 };
    case 'approvalRequested':
      return { ...at, state: 'awaiting_approval' };
    case 'approvalDecided':
      return { ...at, state: 'running' };
    case 'milestoneReached': {
      // A milestone pauses the run. The phase flips to
      // `awaiting_milestone` so the controls can show the gate; the
      // gate object itself is what the user reads.
      const id = text(payload, 'checkpointId');
      const ordinal = count(payload, 'ordinal') ?? 0;
      const summary = text(payload, 'summary') ?? '';
      if (!id) return at;
      return {
        ...at,
        phase: 'awaiting_milestone',
        milestone: {
          checkpointId: id,
          ordinal,
          summary,
        },
      };
    }
    case 'milestoneAcknowledged': {
      // The user signed off. The phase goes back to running and
      // the gate clears. We do not clear `state` here because the
      // backend also emits a `runResumed` that brings the live
      // state forward; either is enough, both are safe.
      if (!at.milestone) return at;
      return { ...at, phase: 'running', milestone: null };
    }
    case 'toolAuthorized': {
      const id = text(payload, 'toolCallId');
      const next: RunViewState = { ...at, state: 'executing_tool' };
      if (!id || at.activity.some(item => item.id === id)) return next;
      return {
        ...next,
        activity: [
          ...at.activity,
          { id, tool: text(payload, 'tool') ?? 'unknown', status: 'running' },
        ],
      };
    }
    case 'toolSucceeded':
      return { ...settle(at, payload, 'done'), state: 'tool_result_recorded' };
    case 'toolFailed':
      return { ...settle(at, payload, 'failed'), state: 'tool_result_recorded' };
    case 'toolReplayed':
      return { ...settle(at, payload, 'replayed'), state: 'tool_result_recorded' };
    case 'toolRefused':
      return { ...settle(at, payload, 'refused'), state: 'running' };
    case 'toolEffectUnknown': {
      const effect: UnknownEffect = {
        idempotencyKey: text(payload, 'idempotencyKey') ?? '',
        tool: text(payload, 'tool') ?? 'unknown',
        target: text(payload, 'target') ?? '',
        at: event.at,
      };
      const settled = settle(at, payload, 'unknown');
      return {
        ...settled,
        unknownEffects: settled.unknownEffects.some(
          known => known.idempotencyKey === effect.idempotencyKey,
        )
          ? settled.unknownEffects
          : [...settled.unknownEffects, effect],
      };
    }
    case 'toolEffectReconciled': {
      const key = text(payload, 'idempotencyKey');
      return key
        ? {
            ...at,
            unknownEffects: at.unknownEffects.filter(
              effect => effect.idempotencyKey !== key,
            ),
          }
        : at;
    }
    case 'verificationStarted':
      return { ...at, state: 'verifying' };
    default:
      return at;
  }
}

/**
 * What a client should do with a durable event it has just received.
 *
 * Kept separate from applying it because the interesting case is the one where
 * it must *not* be applied: a sequence number more than one ahead means at
 * least one event never arrived, and folding this one on top would produce a
 * state that silently disagrees with the record.
 */
export type Reception =
  /** Exactly the next one. Apply it. */
  | { action: 'apply' }
  /** Already folded in. A reconnect re-sending what we had; ignore it. */
  | { action: 'ignore' }
  /** At least one event is missing. Fetch a snapshot and catch up. */
  | { action: 'reconcile'; missing: number };

/** Decides what to do with an event that has just arrived. */
export function receive(lastSeq: number, incoming: number): Reception {
  if (incoming <= lastSeq) return { action: 'ignore' };
  if (incoming === lastSeq + 1) return { action: 'apply' };
  return { action: 'reconcile', missing: incoming - lastSeq - 1 };
}

/**
 * Folds one live event into the state.
 *
 * Kept separate from the durable fold on purpose. They look similar and are
 * not: this one has no sequence number to check against, so it cannot tell a
 * repeat from a new event, and it must never be used to catch up a state that
 * was recovered from the record.
 */
export function applyLiveEvent(state: RunViewState, event: AgentEvent): RunViewState {
  switch (event.type) {
    case 'plan_ready':
      return { ...state, plan: event.plan, phase: 'running' };

    case 'plan_step':
      // The plan's own count, not one this side keeps: a dropped event would
      // leave a locally incremented counter permanently wrong.
      return state.plan
        ? { ...state, plan: { ...state.plan, stepsTaken: event.stepsTaken } }
        : state;

    case 'plan_stopped':
      return { ...state, stopped: event.reason };

    case 'milestone_reached':
      // The live stream and the durable stream both emit this; the
      // reducer is the same. Whichever side gets there first sets
      // the gate; the other side is a no-op.
      return {
        ...state,
        phase: 'awaiting_milestone',
        milestone: {
          checkpointId: event.checkpointId,
          ordinal: event.ordinal,
          summary: event.summary,
        },
      };

    case 'milestone_acknowledged':
      // The user signed off. Return to running and clear the gate.
      if (!state.milestone) return state;
      return { ...state, phase: 'running', milestone: null };

    case 'tool_execution_start':
      return state.activity.some(item => item.id === event.toolCallId)
        ? state
        : {
            ...state,
            activity: [
              ...state.activity,
              {
                id: event.toolCallId,
                tool: event.toolName,
                status: 'running',
                startedAt: Date.now(),
              },
            ],
          };

    case 'tool_execution_end': {
      // A call the gateway stopped before it ran is a different outcome from
      // one that ran and failed, and somebody reading the trace needs to be
      // able to tell them apart.
      const status: Activity['status'] = !event.isError
        ? 'done'
        : event.executionStarted === false
          ? 'refused'
          : 'failed';
      const endedAt = Date.now();
      return {
        ...state,
        activity: state.activity.map(item => {
          if (item.id !== event.toolCallId) return item;
          const startedAt = item.startedAt ?? endedAt;
          return { ...item, status, startedAt, endedAt };
        }),
      };
    }

    // `turn_end` and `context_compacted` are deliberately absent. They arrive
    // on both channels, and counting them here as well would double every
    // figure for a window that is watching both — which is every window.
    default:
      return state;
  }
}
