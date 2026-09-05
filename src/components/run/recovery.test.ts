import { describe, expect, it } from 'vitest';

import {
  applyDurableEvent,
  applyLiveEvent,
  fromSnapshot,
  receive,
  IDLE,
  phaseFor,
  type RunViewState,
} from './recovery';
import type {
  DurableEvent,
  RunState,
  TaskEventType,
  TaskSnapshot,
} from '../../services/agent.service';

it('restores pause and recovery without claiming completion and folds resume once', () => {
  expect(fromSnapshot(snapshot({ state: 'paused' })).phase).toBe('paused');
  expect(fromSnapshot(snapshot({ state: 'recovering' })).phase).toBe('recovering');
  const paused = applyDurableEvent(IDLE, event(1, 'runPaused'));
  expect(paused.state).toBe('paused');
  const resumed = applyDurableEvent(paused, event(2, 'runResumed'));
  expect(resumed.state).toBe('running');
  expect(resumed.summary).toBeNull();
  expect(applyDurableEvent(resumed, event(1, 'runPaused'))).toBe(resumed);
});

/**
 * What the window has to be able to reconstruct.
 *
 * The backend's own tests cover the record being written correctly. These cover
 * the other half: that a window handed that record puts the right thing on the
 * screen — and, the part only this side can do, that it notices when the record
 * it is being streamed has a hole in it.
 */

const snapshot = (over: Partial<TaskSnapshot> = {}): TaskSnapshot => ({
  runId: 'run-1',
  seq: 4,
  schemaVersion: 2,
  state: 'running',
  startedAt: '2026-08-27T10:00:00+00:00',
  updatedAt: '2026-08-27T10:00:20+00:00',
  deadline: '2026-08-27T10:10:00+00:00',
  actor: 'priya',
  prompt: 'draft an approval note',
  modelName: 'Qwen2.5 7B',
  classification: 'Internal',
  plan: null,
  activity: [
    { toolCallId: 'c1', tool: 'search_documents', status: 'done', at: '2026-08-27T10:00:10+00:00' },
    { toolCallId: 'c2', tool: 'create_docx', status: 'running', at: '2026-08-27T10:00:20+00:00' },
  ],
  turns: 2,
  compactions: 0,
  artifacts: [],
  approvalsPending: 0,
  unknownEffects: [],
  stoppedBecause: null,
  failure: null,
  answerHash: null,
  answerChars: 0,
  unreadableEvents: [],
  anomalies: [],
  ...over,
});

const event = (
  seq: number,
  eventType: TaskEventType,
  payload: Record<string, unknown> = {},
): DurableEvent => ({
  runId: 'run-1',
  seq,
  eventId: `e${seq}`,
  eventType,
  at: '2026-08-27T10:00:30+00:00',
  actor: 'priya',
  schemaVersion: 2,
  payload,
});

// == Event gaps trigger snapshot reconciliation ===========================

describe('deciding what to do with an event that has arrived', () => {
  it('applies the one that follows what we already have', () => {
    expect(receive(12, 13)).toEqual({ action: 'apply' });
  });

  it('ignores one it has already folded in', () => {
    // A reconnect re-sending what we had. Making this an error would push retry
    // logic into every caller.
    expect(receive(12, 12)).toEqual({ action: 'ignore' });
    expect(receive(12, 5)).toEqual({ action: 'ignore' });
  });

  it('reconciles when a sequence number jumps', () => {
    // The whole reason the durable channel carries a number. Folding seq 15 on
    // top of seq 12 would produce a state that silently disagrees with the
    // record — and nothing downstream would ever find out.
    expect(receive(12, 15)).toEqual({ action: 'reconcile', missing: 2 });
  });

  it('reconciles from the very first event when one was missed', () => {
    expect(receive(0, 1)).toEqual({ action: 'apply' });
    expect(receive(0, 2)).toEqual({ action: 'reconcile', missing: 1 });
  });

  it('reports how many were missed, not merely that some were', () => {
    // Not used to fetch them — the snapshot is what gets fetched — but it is
    // the number worth logging when this starts happening often.
    expect(receive(3, 9)).toEqual({ action: 'reconcile', missing: 5 });
  });
});

describe('the state after reconciling', () => {
  it('is the snapshot, not the events applied on top of a stale view', () => {
    // A window at seq 12 that missed 13 and 14 must not end up at 15 with two
    // events' worth of state missing. Adopting the snapshot replaces the view
    // wholesale, which is the only way to be sure.
    const stale: RunViewState = {
      ...fromSnapshot(snapshot({ seq: 12, turns: 99 })),
      turns: 99,
    };
    const repaired = fromSnapshot(snapshot({ seq: 15, turns: 4, state: 'verifying' }));

    expect(stale.turns).toBe(99);
    expect(repaired.turns).toBe(4);
    expect(repaired.seq).toBe(15);
    expect(repaired.state).toBe('verifying');
  });
});

// == Adopting a run off the record ========================================

describe('adopting a run off the record', () => {
  it('rebuilds the trace a remount would otherwise have lost', () => {
    const state = fromSnapshot(snapshot());

    expect(state.runId).toBe('run-1');
    expect(state.prompt).toBe('draft an approval note');
    expect(state.phase).toBe('running');
    expect(state.turns).toBe(2);
    expect(state.activity).toEqual([
      { id: 'c1', tool: 'search_documents', status: 'done' },
      { id: 'c2', tool: 'create_docx', status: 'running' },
    ]);
  });

  it('says the trace is a reconstruction rather than passing it off as live', () => {
    expect(fromSnapshot(snapshot()).recovered).toBe(true);
    expect(IDLE.recovered).toBe(false);
  });

  it('carries the sequence number so catching up does not start from the beginning', () => {
    expect(fromSnapshot(snapshot({ seq: 12 })).seq).toBe(12);
  });

  it('flags a history with a hole in it', () => {
    const unreadable = fromSnapshot(
      snapshot({
        unreadableEvents: [{ seq: 3, eventId: 'e3', problem: 'the payload is not readable JSON' }],
      }),
    );
    expect(unreadable.historyIncomplete).toBe(true);
  });

  it('flags a history two writers disagreed about', () => {
    const contested = fromSnapshot(
      snapshot({ anomalies: ['seq 7: a run_routed event would move the run back to routed'] }),
    );
    expect(contested.historyIncomplete).toBe(true);
  });

  it('shows a status from a newer backend as still running rather than dropping the row', () => {
    const state = fromSnapshot(
      snapshot({ activity: [{ toolCallId: 'c1', tool: 'create_docx', status: 'quantum', at: '' }] }),
    );
    expect(state.activity[0].status).toBe('running');
  });
});

// == The states ===========================================================

describe('the endings that are not failures', () => {
  it.each([
    ['cancelled', 'Stopped, because somebody stopped it.'],
    ['stopped_by_budget', 'Stopped: it reached the limit the plan set for it.'],
    ['stopped_by_policy', 'Stopped: it needed to do something it is not permitted to do.'],
    ['degraded_needs_human', 'Interrupted. Somebody needs to look at this before it is relied on.'],
  ] as const)('%s reads as itself and not as a crash', (state, sentence) => {
    const recovered = fromSnapshot(snapshot({ state, failure: sentence }));
    expect(recovered.state).toBe(state);
    expect(recovered.error).toBe(sentence);
    expect(recovered.phase).toBe('failed');
  });

  it('supplies a sentence when the record has no failure text', () => {
    // An ending with no words is worse than no ending: the screen goes quiet
    // and the person cannot tell it from a run still thinking.
    const state = fromSnapshot(snapshot({ state: 'degraded_needs_human', failure: null }));
    expect(state.error).toContain('Interrupted');
  });

  it('does not put an error on a run that is simply still going', () => {
    for (const live of ['created', 'routed', 'running', 'awaiting_approval', 'verifying'] as const) {
      expect(fromSnapshot(snapshot({ state: live })).error).toBeNull();
    }
  });

  it('maps each state to the phase the screen draws', () => {
    const cases: [RunState, string][] = [
      ['created', 'running'],
      ['classified', 'running'],
      ['routed', 'running'],
      ['planned', 'running'],
      ['running', 'running'],
      ['awaiting_approval', 'running'],
      ['executing_tool', 'running'],
      ['tool_result_recorded', 'running'],
      ['verifying', 'running'],
      ['completed', 'finished'],
      ['cancelled', 'failed'],
      ['failed', 'failed'],
      ['stopped_by_budget', 'failed'],
      ['stopped_by_policy', 'failed'],
      ['degraded_needs_human', 'failed'],
    ];
    for (const [state, phase] of cases) {
      expect(phaseFor(state)).toBe(phase);
    }
  });
});

// == Catching up ==========================================================

describe('catching up on the events after a snapshot', () => {
  const base = () => fromSnapshot(snapshot());

  it('walks the lifecycle states as the events arrive', () => {
    let state = fromSnapshot(snapshot({ seq: 0, state: 'created', activity: [] }));
    const walk: [TaskEventType, RunState][] = [
      ['runClassified', 'classified'],
      ['runRouted', 'routed'],
      ['planReady', 'planned'],
      ['runStarted', 'running'],
      ['toolAuthorized', 'executing_tool'],
      ['toolSucceeded', 'tool_result_recorded'],
      ['verificationStarted', 'verifying'],
      ['runCompleted', 'completed'],
    ];
    walk.forEach(([eventType, expected], index) => {
      state = applyDurableEvent(state, event(index + 1, eventType, { toolCallId: 'c9' }));
      expect(state.state).toBe(expected);
    });
  });

  it('ignores an event it has already folded in', () => {
    const state = base();
    const again = applyDurableEvent(state, event(4, 'runCompleted', { turns: 99 }));
    expect(again).toBe(state);
    expect(again.turns).toBe(2);
  });

  it('records a refusal even though no authorisation preceded it', () => {
    const state = applyDurableEvent(
      base(),
      event(5, 'toolRefused', { toolCallId: 'c9', tool: 'execute_code' }),
    );
    expect(state.activity).toHaveLength(3);
    expect(state.activity[2]).toMatchObject({ id: 'c9', tool: 'execute_code', status: 'refused' });
  });

  it('marks a replayed side effect as not repeated', () => {
    const state = applyDurableEvent(base(), event(5, 'toolReplayed', { toolCallId: 'c2' }));
    expect(state.activity[1].status).toBe('replayed');
  });

  it('surfaces a side effect nobody can account for', () => {
    // The one thing on this screen that asks a person to go and look at
    // something in the world.
    const state = applyDurableEvent(
      base(),
      event(5, 'toolEffectUnknown', {
        toolCallId: 'c2',
        tool: 'create_docx',
        target: 'note.docx',
        idempotencyKey: 'k1',
      }),
    );
    expect(state.activity[1].status).toBe('unknown');
    expect(state.unknownEffects).toEqual([
      { idempotencyKey: 'k1', tool: 'create_docx', target: 'note.docx', at: state.unknownEffects[0].at },
    ]);
  });

  it('stops asking once a person has accounted for it', () => {
    let state = applyDurableEvent(
      base(),
      event(5, 'toolEffectUnknown', {
        toolCallId: 'c2',
        tool: 'create_docx',
        target: 'note.docx',
        idempotencyKey: 'k1',
      }),
    );
    state = applyDurableEvent(state, event(6, 'toolEffectReconciled', { idempotencyKey: 'k1' }));
    expect(state.unknownEffects).toHaveLength(0);
  });

  it('takes the plan step count from the plan rather than counting locally', () => {
    let state: RunViewState = {
      ...base(),
      plan: {
        steps: [],
        maxSteps: 12,
        maxDurationSeconds: 600,
        permittedTools: [],
        repeatLimit: 3,
        stepsTaken: 1,
        stopReason: null,
        stoppedBecause: 'Still running.',
      },
    };
    state = applyDurableEvent(state, event(5, 'planStep', { stepsTaken: 7 }));
    expect(state.plan?.stepsTaken).toBe(7);
  });

  it('keeps the caveat that earlier turns were replaced by a summary', () => {
    let state = base();
    state = applyDurableEvent(state, event(5, 'contextCompacted', {}));
    state = applyDurableEvent(state, event(6, 'turnEnded'));
    expect(state.compactions).toBe(1);
    expect(state.turns).toBe(3);
  });

  it('reads a schema-1 ending as its schema-2 equivalent', () => {
    // A database written by an earlier build still folds rather than leaving
    // the run apparently unfinished forever.
    expect(applyDurableEvent(base(), event(5, 'runTimedOut')).state).toBe('stopped_by_budget');
    expect(applyDurableEvent(base(), event(5, 'runInterrupted')).state).toBe(
      'degraded_needs_human',
    );
  });
});

// == The live channel =====================================================

describe('the live stream, once it is believable again', () => {
  it('does not duplicate a call the recovered trace already has', () => {
    const state = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'tool_execution_start',
      toolCallId: 'c2',
      toolName: 'create_docx',
    });
    expect(state.activity).toHaveLength(2);
  });

  it('still adds a call the recovered trace has not seen', () => {
    const state = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'tool_execution_start',
      toolCallId: 'c3',
      toolName: 'validate_artifact',
    });
    expect(state.activity).toHaveLength(3);
  });

  it('tells a refusal apart from a tool that ran and failed', () => {
    const refused = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'tool_execution_end',
      toolCallId: 'c2',
      toolName: 'create_docx',
      isError: true,
      executionStarted: false,
    });
    const failed = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'tool_execution_end',
      toolCallId: 'c2',
      toolName: 'create_docx',
      isError: true,
    });
    expect(refused.activity[1].status).toBe('refused');
    expect(failed.activity[1].status).toBe('failed');
  });

  it('does not count turns or compactions, because the durable channel does', () => {
    // Both channels carry these. Counting them here as well would double every
    // figure for a window watching both — which is every window.
    const turned = applyLiveEvent(fromSnapshot(snapshot()), { type: 'turn_end' });
    const compacted = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'context_compacted',
      tokensBefore: 8000,
      tokensAfter: 2100,
      messagesSummarised: 14,
    });
    expect(turned.turns).toBe(2);
    expect(compacted.compactions).toBe(0);
  });
});

// == Confidential arguments are absent from UI events =====================

describe('what reaches the screen', () => {
  it('never sees a tool argument, because the payload it is given has none', () => {
    // The backend redacts on the way into the record, so the envelope this
    // side receives already carries hashes. This asserts the frontend does not
    // reintroduce the raw value from anywhere else — it reads `tool` and
    // `toolCallId` and nothing that could hold document text.
    const state = applyDurableEvent(
      fromSnapshot(snapshot()),
      event(5, 'toolSucceeded', {
        toolCallId: 'c2',
        tool: 'create_docx',
        detail: { sha256: 'abc123', chars: 812 },
      }),
    );
    const rendered = JSON.stringify(state);
    expect(rendered).not.toContain('abc123');
    expect(rendered).toContain('create_docx');
  });
});

// == Milestone gates =====================================================

describe('milestone checkpoints', () => {
  it('a milestoneReached event opens a gate with the checkpoint id and intent', () => {
    const state = applyDurableEvent(
      fromSnapshot(snapshot()),
      event(5, 'milestoneReached', {
        checkpointId: 'mtn-survey',
        ordinal: 1,
        summary: 'Surveyed the SOPs',
      }),
    );
    expect(state.phase).toBe('awaiting_milestone');
    expect(state.milestone).toEqual({
      checkpointId: 'mtn-survey',
      ordinal: 1,
      summary: 'Surveyed the SOPs',
    });
  });

  it('a milestoneReached event with no id is a no-op, not a stuck gate', () => {
    const state = applyDurableEvent(
      fromSnapshot(snapshot()),
      event(5, 'milestoneReached', { ordinal: 1, summary: 'Surveyed' }),
    );
    // Phase does not flip; the gate stays closed.
    expect(state.phase).toBe('running');
    expect(state.milestone).toBeNull();
  });

  it('a milestoneAcknowledged event returns the run to running and clears the gate', () => {
    const opened = applyDurableEvent(
      fromSnapshot(snapshot()),
      event(5, 'milestoneReached', {
        checkpointId: 'mtn-survey',
        ordinal: 1,
        summary: 'Surveyed',
      }),
    );
    const closed = applyDurableEvent(
      opened,
      event(6, 'milestoneAcknowledged', {
        checkpointId: 'mtn-survey',
        acknowledgedBy: 'priya',
      }),
    );
    expect(closed.phase).toBe('running');
    expect(closed.milestone).toBeNull();
  });

  it('the live channel opens the same gate the durable channel does', () => {
    const live = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'milestone_reached',
      checkpointId: 'mtn-survey',
      ordinal: 1,
      summary: 'Surveyed',
    });
    expect(live.phase).toBe('awaiting_milestone');
    expect(live.milestone?.checkpointId).toBe('mtn-survey');
  });

  it('the live channel closes the same gate the durable channel does', () => {
    const opened = applyLiveEvent(fromSnapshot(snapshot()), {
      type: 'milestone_reached',
      checkpointId: 'mtn-survey',
      ordinal: 1,
      summary: 'Surveyed',
    });
    const closed = applyLiveEvent(opened, {
      type: 'milestone_acknowledged',
      checkpointId: 'mtn-survey',
      acknowledgedBy: 'priya',
    });
    expect(closed.phase).toBe('running');
    expect(closed.milestone).toBeNull();
  });

  it('a milestoneAcknowledged event with no open gate is a no-op', () => {
    const state = applyDurableEvent(
      fromSnapshot(snapshot()),
      event(5, 'milestoneAcknowledged', {
        checkpointId: 'mtn-survey',
        acknowledgedBy: 'priya',
      }),
    );
    expect(state.phase).toBe('running');
    expect(state.milestone).toBeNull();
  });
});
