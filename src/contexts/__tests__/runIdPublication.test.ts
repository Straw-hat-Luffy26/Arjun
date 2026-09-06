/**
 * The surface has to learn the run's real id, not just hold its own.
 *
 * ## The defect
 *
 * Two ids name a turn. The composer mints a correlation id before the request
 * goes out, because it needs something to route events by while the run is
 * still being created. The server then mints the run's *own* id, and that is
 * what the task record, the audit trail, the event stream and the runtime's
 * table of live runs are all keyed by.
 *
 * `RunReducer` learned the real one from the first `plan_ready` and kept it in a
 * private field. Nothing else was ever told. So `activeRunId` — the value the
 * whole surface addresses a run by — stayed the correlation id for the life of
 * the turn, and three things broke together, all of them silently:
 *
 * - **Stop.** `agent_abort_run` was sent an id neither the core nor the runtime
 *   had heard of. It returned `false`, and the composer discarded the answer.
 *   The button cleared its spinner and the model carried on.
 * - **The context meter.** It filters `context_ledger` events by run id. None
 *   matched, so it read "No context yet" for the whole of every run.
 * - **Reattachment.** A window following the live run followed an id nothing
 *   would ever emit under.
 *
 * These tests pin the fix at its source: the reducer announces the id, it
 * announces the *server's*, exactly once, and only for the cell it owns.
 */
import { describe, expect, it, beforeAll } from 'vitest';
import { RunReducer, RunReducerRegistry } from '../ConversationContext';
import type { AgentEvent, AgentEventEnvelope } from '../../services/agent.service';

// `RunReducer` uses `window.setTimeout` for its debounces. vitest's `node`
// environment has no `window`; this is the same minimal polyfill the sibling
// reducer tests install.
beforeAll(() => {
  if (typeof globalThis.window === 'undefined') {
    (globalThis as unknown as { window: unknown }).window = {
      setTimeout: (handler: (...args: unknown[]) => void, ms: number) =>
        setTimeout(handler, ms) as unknown as number,
      clearTimeout: (id: number) => clearTimeout(id),
    };
  }
});

/** A registry that records what it was told, and nothing else. */
function recordingRegistry() {
  const announced: Array<{ messageId: string; runId: string }> = [];
  const registry = new RunReducerRegistry({
    onContent: () => undefined,
    onReasoning: () => undefined,
    onProgress: () => undefined,
    onConversation: () => undefined,
    onRunId: (messageId, runId) => announced.push({ messageId, runId }),
    onRunDone: () => undefined,
  });
  return { registry, announced };
}

function envelope(runId: string, event: AgentEvent): AgentEventEnvelope {
  return { runId, event };
}

/** The frame that echoes the caller's correlation id back with the real run id. */
function planReady(serverRunId: string, correlationId: string): AgentEventEnvelope {
  return envelope(serverRunId, {
    type: 'plan_ready',
    correlationId,
  } as unknown as AgentEvent);
}

describe('a reducer announces the run id the server issued', () => {
  it('announces it from plan_ready, which is where the id first arrives', () => {
    const { registry, announced } = recordingRegistry();
    const reducer = new RunReducer(registry, 'correlation-1', 'conv-1', 'msg-1');

    reducer.apply(planReady('server-run-1', 'correlation-1'));

    expect(announced).toEqual([{ messageId: 'msg-1', runId: 'server-run-1' }]);
    reducer.dispose();
  });

  /**
   * The id it announces is the server's, never the one it was constructed with.
   * That distinction is the entire defect: an announcement carrying the
   * correlation id would be as useless as no announcement at all.
   */
  it('announces the server id and not its own correlation id', () => {
    const { registry, announced } = recordingRegistry();
    const reducer = new RunReducer(registry, 'correlation-1', 'conv-1', 'msg-1');

    reducer.apply(planReady('server-run-1', 'correlation-1'));

    expect(announced[0]?.runId).not.toBe('correlation-1');
    reducer.dispose();
  });

  /**
   * A model fast enough to stream before its plan is published teaches the id
   * on a message event instead. Both paths must announce, or Stop works for
   * slow models and not for fast ones.
   */
  it('announces it from an early message event when the model beats plan_ready', () => {
    const { registry, announced } = recordingRegistry();
    const reducer = new RunReducer(registry, 'correlation-1', 'conv-1', 'msg-1');

    reducer.apply(
      envelope('server-run-1', {
        type: 'message_start',
        messageId: 'msg-1',
        role: 'assistant',
      }),
    );

    expect(announced).toEqual([{ messageId: 'msg-1', runId: 'server-run-1' }]);
    reducer.dispose();
  });

  it('announces once, however many events carry the id', () => {
    const { registry, announced } = recordingRegistry();
    const reducer = new RunReducer(registry, 'correlation-1', 'conv-1', 'msg-1');

    reducer.apply(planReady('server-run-1', 'correlation-1'));
    reducer.apply(
      envelope('server-run-1', {
        type: 'message_start',
        messageId: 'msg-1',
        role: 'assistant',
      }),
    );
    reducer.apply(
      envelope('server-run-1', {
        type: 'message_update',
        messageId: 'msg-1',
        delta: 'x',
      }),
    );

    expect(announced).toHaveLength(1);
    reducer.dispose();
  });

  /**
   * Two runs can be registered at once, and each must announce only its own.
   * A reducer that announced an id it read off another run's frame would point
   * Stop at the wrong turn — a worse failure than the one being fixed, because
   * that one stops work somebody wanted.
   */
  it('does not announce another run’s id', () => {
    const { registry, announced } = recordingRegistry();
    const mine = new RunReducer(registry, 'correlation-a', 'conv-1', 'msg-a');

    // A plan for somebody else's turn, echoing somebody else's correlation id.
    mine.apply(planReady('server-run-b', 'correlation-b'));
    // And a message event naming the other run's cell.
    mine.apply(
      envelope('server-run-b', {
        type: 'message_start',
        messageId: 'msg-b',
        role: 'assistant',
      }),
    );

    expect(announced).toEqual([]);
    mine.dispose();
  });

  /**
   * The callback is optional, like `onReasoning`: a consumer that renders
   * finished conversations has no live run to address. A reducer must not throw
   * because nobody is listening.
   */
  it('works for a registry that does not want to be told', () => {
    const registry = new RunReducerRegistry({
      onContent: () => undefined,
      onProgress: () => undefined,
      onConversation: () => undefined,
      onRunDone: () => undefined,
    });
    const reducer = new RunReducer(registry, 'correlation-1', 'conv-1', 'msg-1');

    expect(() => reducer.apply(planReady('server-run-1', 'correlation-1'))).not.toThrow();
    reducer.dispose();
  });
});
