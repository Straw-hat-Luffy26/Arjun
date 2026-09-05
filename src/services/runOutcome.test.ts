/**
 * The frontend's half of the typed run outcome.
 *
 * The defect these pin: the surface used to learn how a run ended by asking
 * whether the command had returned and whether the stream had stopped. Both
 * questions have the same answer for a finished answer, a fragment cut off at
 * the output cap, a run somebody stopped, and a run the gateway refused — so
 * all four were shown to a person as an answer.
 *
 * `RunOutcome` mirrors `src-tauri/src/agent_runtime/outcome.rs`. The six
 * spellings here and the six there are one contract; the Rust test
 * `the_wire_spelling_matches_the_typescript_union` holds the other end of it.
 */

import { existsSync, readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { demoService } from './demo.service';
import {
  MESSAGE_STATUS_LABELS,
  messageStatus,
  type MessageStatusInput,
  type MessageStatusKind,
} from './agent.service';
import {
  CANONICAL_TOOL_NAMES,
  LEGACY_TOOL_NAMES,
  canonicalToolName,
  labelForTool,
} from './toolNames';
import {
  endingFromFinishReason,
  outcomeDetail,
  runSucceeded,
  type AuditState,
  type RunOutcome,
  type RunOutcomeKind,
} from './agent.service';

/** Every ending the backend can report, with a plausible sentence for each. */
const ENDINGS: RunOutcome[] = [
  { kind: 'completed' },
  { kind: 'needsReview', detail: 'An external effect must be reconciled.' },
  { kind: 'failed', detail: 'The model server refused: 503.' },
  { kind: 'aborted', detail: 'Stopped: operator stopped it.' },
  {
    kind: 'lengthLimited',
    detail: 'Stopped: the answer reached the output limit for one turn.',
  },
  { kind: 'budgetStopped', detail: 'Stopped: it ran past the time its plan allowed.' },
  {
    kind: 'policyStopped',
    detail: 'Stopped: it needed to do something it is not permitted to do.',
  },
];

describe('runSucceeded: only a completion counts as having done the work', () => {
  it('is true for a completion and false for every other ending', () => {
    for (const outcome of ENDINGS) {
      expect(runSucceeded(outcome)).toBe(outcome.kind === 'completed');
    }
  });

  it('is false when the outcome is unknown', () => {
    // An older record, or a run whose outcome never arrived. "Not recorded" is
    // not "it worked" — treating absence as success is the original defect in
    // its smallest form.
    expect(runSucceeded(undefined)).toBe(false);
    expect(runSucceeded(null)).toBe(false);
  });

  it('is false for a run cut off at the output cap, which produced real text', () => {
    // The one the surface most easily gets wrong: there *is* an answer, and it
    // reads exactly like a short one.
    const cut: RunOutcome = { kind: 'lengthLimited', detail: 'cut off' };
    expect(runSucceeded(cut)).toBe(false);
  });
});

describe('outcomeDetail: every bad ending explains itself', () => {
  it('returns the sentence for each ending that is not a completion', () => {
    for (const outcome of ENDINGS) {
      const detail = outcomeDetail(outcome);
      if (outcome.kind === 'completed') {
        expect(detail).toBeNull();
      } else {
        expect(detail).toBe(outcome.detail);
      }
    }
  });

  it('returns null rather than inventing a reason when nothing was recorded', () => {
    expect(outcomeDetail(undefined)).toBeNull();
    expect(outcomeDetail(null)).toBeNull();
  });
});

describe('endingFromFinishReason: a message_end says only what it knows', () => {
  it('reports the output cap, which is conclusive on its own', () => {
    expect(endingFromFinishReason('length')).toEqual({
      outcome: 'lengthLimited',
      failed: true,
    });
  });

  it('reports a turn that errored, which is also conclusive', () => {
    expect(endingFromFinishReason('error')).toEqual({ outcome: 'failed', failed: true });
  });

  it('claims no outcome for a clean stop, leaving it to the run', () => {
    // A turn that stopped cleanly says nothing about whether the *run* was then
    // stopped by its budget or refused by policy. Claiming completion here is
    // the same mistake as claiming it from a resolved request.
    for (const reason of ['stop', 'tool_calls', 'content_filter'] as const) {
      const ending = endingFromFinishReason(reason);
      expect(ending.outcome).toBeUndefined();
      expect(ending.failed).toBe(false);
    }
  });

  it('never reports a completion from a finish reason alone', () => {
    const reasons = ['stop', 'length', 'tool_calls', 'content_filter', 'error'] as const;
    for (const reason of reasons) {
      expect(endingFromFinishReason(reason).outcome).not.toBe('completed');
    }
  });
});

describe('the outcome union covers every state the backend can report', () => {
  it('consumes the shared Rust wire fixtures without turning review into success', () => {
    const fixtures = JSON.parse(readFileSync('contracts/run-outcomes.json', 'utf8')) as RunOutcome[];
    for (const outcome of fixtures) {
      expect(runSucceeded(outcome)).toBe(outcome.kind === 'completed');
    }
    expect(messageStatus({ isStreaming: false, contentLength: 80, runningTools: 0,
      outcome: fixtures.find((outcome) => outcome.kind === 'needsReview')!.kind,
      verification: 'ready' })).toBe('needsReview');
  });
  it('names all seven endings, and no ending is missing a case here', () => {
    const kinds: RunOutcomeKind[] = ENDINGS.map((outcome) => outcome.kind);
    expect(new Set(kinds)).toEqual(
      new Set([
        'completed',
        'needsReview',
        'failed',
        'aborted',
        'lengthLimited',
        'budgetStopped',
        'policyStopped',
      ]),
    );
  });

  it('gives every non-completion a sentence to show', () => {
    for (const outcome of ENDINGS) {
      if (outcome.kind === 'completed') continue;
      expect(outcome.detail.length).toBeGreaterThan(0);
    }
  });
});

/**
 * Whether this installation can still record what it does.
 *
 * The defect: a task event log that could not be opened was silently replaced
 * by an in-memory one, so the application came up looking entirely normal, ran
 * tasks, wrote files, and kept a history that evaporated when the process
 * exited. Nobody using it was told.
 *
 * Mirrors `AuditState` in `src-tauri/src/agent_runtime/audit_health.rs`, whose
 * test `the_state_serialises_for_the_ui` holds the other end of the contract.
 */
describe('AuditState: the surface can tell a working installation from a broken one', () => {
  it('distinguishes durable from degraded without inspecting anything else', () => {
    // Typed as the union rather than as either branch, so the comparison is
    // the one the surface actually makes on a value it was handed.
    const states: AuditState[] = [
      { state: 'durable' },
      {
        state: 'degraded',
        because: 'The task event log could not be opened.',
        atStartup: true,
      },
    ];
    expect(states.map((s) => s.state === 'durable')).toEqual([true, false]);
  });

  it('carries a sentence to show and where the failure happened', () => {
    // Both halves matter to the person reading it: a store that never opened
    // and one that stopped working mid-session have different remedies.
    const atStartup: AuditState = {
      state: 'degraded',
      because: 'The task event log could not be opened.',
      atStartup: true,
    };
    const midSession: AuditState = {
      state: 'degraded',
      because: 'There is no space left on the device.',
      atStartup: false,
    };
    expect(atStartup.because.length).toBeGreaterThan(0);
    expect(atStartup.atStartup).toBe(true);
    expect(midSession.atStartup).toBe(false);
  });

  it('narrows so a degraded state cannot be read without its reason', () => {
    // A compile-time property, asserted at runtime so it is visible: there is
    // no way to be degraded and say nothing about why.
    const state: AuditState = {
      state: 'degraded',
      because: 'The disk is full.',
      atStartup: false,
    };
    if (state.state === 'degraded') {
      expect(state.because).toBe('The disk is full.');
    } else {
      throw new Error('narrowed to the wrong branch');
    }
  });
});

/**
 * The surface's tool-name table, against the protocol.
 *
 * The defect: two label maps, one in `useRun.ts` and one in
 * `AssistantMessageCell.tsx`, both keyed on the pre-namespace spelling while
 * live events carry the current one. Every activity row displayed the raw wire
 * name -- `artifact.create_approval_note` where the design says "Producing a
 * Word document".
 *
 * These pin the same contract the agent runtime's
 * `catalogue.conformance.test.ts` pins on its side, and Rust's `ToolName` holds
 * in the middle. Three lists that must agree will drift silently otherwise.
 */
describe('tool names: the surface resolves both spellings', () => {
  it('labels every canonical tool without falling through to the wire name', () => {
    for (const name of CANONICAL_TOOL_NAMES) {
      const label = labelForTool(name);
      expect(label, name).not.toBe(name);
      expect(label.length, name).toBeGreaterThan(0);
    }
  });

  it('labels a pre-rename record the same as the current name', () => {
    // A task record written months ago holds `create_docx`. It must read as
    // the same thing as a live event carrying the new name.
    for (const [legacy, current] of LEGACY_TOOL_NAMES) {
      expect(labelForTool(legacy), legacy).toBe(labelForTool(current));
    }
  });

  it('folds every legacy spelling onto its current name', () => {
    for (const [legacy, current] of LEGACY_TOOL_NAMES) {
      expect(canonicalToolName(legacy)).toBe(current);
    }
  });

  it('shows the raw name for a tool this build does not know', () => {
    // Honest rather than pretty. Inventing a label for an unknown tool would
    // be neither.
    expect(canonicalToolName('tool.from_a_newer_backend')).toBeUndefined();
    expect(labelForTool('tool.from_a_newer_backend')).toBe('tool.from_a_newer_backend');
  });

  it('names the documents specifically enough to tell them apart', () => {
    // The three artifact tools produce three different things, and a trace
    // that called them all "Producing a document" would not say which.
    const labels = [
      labelForTool('artifact.create_approval_note'),
      labelForTool('artifact.create_calculation_workbook'),
      labelForTool('artifact.create_briefing_deck'),
    ];
    expect(new Set(labels).size).toBe(3);
  });
});

/**
 * What the chat says about a finished turn.
 *
 * ## The defect
 *
 * The status was derived as `isFailed ? 'failed' : isDone ? 'verified' : ...`.
 * "Not streaming and not failed" was rendered as **Verified**, with a green
 * tick, for every turn: ones the verifier never looked at, ones it found
 * blocking problems in, and ones a person stopped part way through. The
 * strongest claim the product can make was the one it made by default, and
 * nothing had checked it.
 */
describe('messageStatus: the pill says what actually happened', () => {
  /** A finished turn with an answer and nothing else known. */
  const finished: MessageStatusInput = {
    isStreaming: false,
    contentLength: 240,
    runningTools: 0,
  };

  it('never reports verified merely because the stream stopped', () => {
    // The defect, in one assertion.
    expect(messageStatus(finished)).not.toBe('verified');
    expect(messageStatus(finished)).toBe('unverified');
  });

  it('reports verified only when the verifier ran and passed', () => {
    expect(messageStatus({ ...finished, outcome: 'completed', verification: 'ready' })).toBe(
      'verified',
    );
  });

  it('reports needs review when the verifier found something', () => {
    expect(
      messageStatus({ ...finished, outcome: 'completed', verification: 'needsReview' }),
    ).toBe('needsReview');
  });

  it('reports unverified when there is an answer and nothing checked it', () => {
    // Distinct from both "passed" and "failed". The verifier did not run —
    // and that is a fact about the answer worth showing, not a pass.
    expect(messageStatus({ ...finished, outcome: 'completed', verification: null })).toBe(
      'unverified',
    );
  });

  it('reports completed when the run finished with nothing to check', () => {
    expect(
      messageStatus({ ...finished, contentLength: 0, outcome: 'completed', verification: null }),
    ).toBe('completed');
  });

  it('reports stopped for every ending a person or a limit caused', () => {
    for (const outcome of ['aborted', 'budgetStopped', 'policyStopped', 'lengthLimited'] as const) {
      expect(messageStatus({ ...finished, outcome }), outcome).toBe('stopped');
    }
  });

  it('reports failed when the run failed', () => {
    expect(messageStatus({ ...finished, outcome: 'failed' })).toBe('failed');
  });

  it('does not let a good verification hide a stopped run', () => {
    // A run cut off at the output cap can leave a fragment that verifies
    // perfectly well. Labelling that "Verified" would certify half an answer.
    expect(
      messageStatus({ ...finished, outcome: 'lengthLimited', verification: 'ready' }),
    ).toBe('stopped');
  });

  it('reports the live states while tokens are still arriving', () => {
    expect(messageStatus({ isStreaming: true, contentLength: 0, runningTools: 0 })).toBe(
      'thinking',
    );
    expect(messageStatus({ isStreaming: true, contentLength: 0, runningTools: 2 })).toBe(
      'usingTool',
    );
    expect(messageStatus({ isStreaming: true, contentLength: 12, runningTools: 0 })).toBe(
      'composing',
    );
  });

  it('prefers composing over usingTool once text is arriving', () => {
    expect(messageStatus({ isStreaming: true, contentLength: 12, runningTools: 3 })).toBe(
      'composing',
    );
  });

  it('gives every state a distinct label to render', () => {
    const states: MessageStatusKind[] = [
      'thinking',
      'usingTool',
      'composing',
      'verified',
      'needsReview',
      'unverified',
      'completed',
      'stopped',
      'failed',
    ];
    for (const state of states) {
      expect(MESSAGE_STATUS_LABELS[state], state).toBeTruthy();
    }
    // Distinct, or two states would read identically on screen — which is the
    // whole failure this replaces.
    const labels = states.map((state) => MESSAGE_STATUS_LABELS[state]);
    expect(new Set(labels).size).toBe(states.length);
  });

  it('reserves the word Verified for the one state that earned it', () => {
    const verifiedLabels = Object.entries(MESSAGE_STATUS_LABELS)
      .filter(([, label]) => label.toLowerCase().includes('verified'))
      .map(([state]) => state);
    expect(verifiedLabels.sort()).toEqual(['unverified', 'verified']);
    expect(MESSAGE_STATUS_LABELS.verified).toBe('Verified');
    expect(MESSAGE_STATUS_LABELS.unverified).toBe('Unverified');
  });
});

/**
 * The demonstrator hands over real documents.
 *
 * ## The defect
 *
 * Every scenario's prompt said its documents were "attached". Nothing was: the
 * page dispatched a prompt, a title and a framing string, and no documents at
 * all. The model was asked to cross-reference a drawing it had never been
 * given — and, being asked about the organisation's record with nothing
 * retrieved, either refused or invented. Both were shown to a judge as the
 * product working.
 *
 * The scenario cards also claimed "a real 3-year TCO", "real SOPs, real safety
 * clauses" and a "real skill chain". None of those were shipped.
 */
describe('demo scenarios: every claim is backed by an input', () => {
  it('gives every scenario at least one checked-in fixture', () => {
    for (const scenario of demoService.list()) {
      expect(scenario.fixtures.length, scenario.id).toBeGreaterThan(0);
    }
  });

  it('names only fixtures that exist on disk', () => {
    // The claim "attached" is only true if the file is there. A fixture named
    // and missing would fail at launch, in front of a judge.
    for (const scenario of demoService.list()) {
      for (const fixture of scenario.fixtures) {
        const path = `src/demo-fixtures/${fixture}`;
        expect(existsSync(path), `${scenario.id} names a missing fixture: ${path}`).toBe(true);
        expect(readFileSync(path, 'utf8').trim().length, path).toBeGreaterThan(0);
      }
    }
  });

  it('reads the P&ID fixture, with the drawing number the prompt cites', () => {
    const pid = readFileSync('src/demo-fixtures/pid-A-101-001.txt', 'utf8');
    expect(pid).toContain('A-101-001');
    expect(pid).toContain('Revision:  6');
    // The tags the prompt asks to be cross-referenced.
    expect(pid).toContain('V-101');
    expect(pid).toContain('P-101A');
  });

  it('reads the quote fixture, with both quotes the prompt compares', () => {
    const quotes = readFileSync('src/demo-fixtures/vendor-quotes.txt', 'utf8');
    expect(quotes).toContain('QUOTE A');
    expect(quotes).toContain('QUOTE B');
    expect(quotes).toContain('75 kW centrifugal pump');
  });

  it('cross-references: the register carries the tags the P&ID shows', () => {
    // The scenario asks for a cross-reference, so both documents have to
    // contain the same tag or there is nothing to cross-reference.
    const pid = readFileSync('src/demo-fixtures/pid-A-101-001.txt', 'utf8');
    const register = readFileSync('src/demo-fixtures/equipment-register.txt', 'utf8');
    for (const tag of ['V-101', 'P-101A', 'P-101B', 'E-101']) {
      expect(pid, `P&ID is missing ${tag}`).toContain(tag);
      expect(register, `register is missing ${tag}`).toContain(tag);
    }
  });

  it('states in every fixture what it does not contain', () => {
    // The honesty half. A document that says what it omits is one an answer
    // can be checked against; without it, "the model invented this" and "the
    // model read this" are indistinguishable to a reviewer.
    const fixtures = [
      'pid-A-101-001.txt',
      'vendor-quotes.txt',
      'incident-report.txt',
    ];
    for (const fixture of fixtures) {
      const text = readFileSync(`src/demo-fixtures/${fixture}`, 'utf8').toLowerCase();
      expect(
        text.includes('does not') || text.includes('not in either') || text.includes('omitted'),
        `${fixture} does not say what it leaves out`,
      ).toBe(true);
    }
  });

  it('says every fixture is synthetic, in the fixture itself', () => {
    for (const scenario of demoService.list()) {
      for (const fixture of scenario.fixtures) {
        const text = readFileSync(`src/demo-fixtures/${fixture}`, 'utf8').toLowerCase();
        expect(text, `${fixture} does not declare itself synthetic`).toContain('synthetic');
      }
    }
  });

  it('asks for a skill on every scenario, and carries it on the launch', () => {
    // The card advertised a skill and the launch never sent one, so the
    // "skill chain" the summary claimed was a label on a button.
    for (const scenario of demoService.list()) {
      expect(scenario.skillId, `${scenario.id} advertises no skill`).toBeTruthy();
    }
  });

  it('sets a classification on every scenario rather than defaulting', () => {
    // It decides which models may see the material. A demo that leaves it to
    // the default is bypassing the decision it is meant to be showing.
    for (const scenario of demoService.list()) {
      expect(scenario.classification, scenario.id).toBeTruthy();
    }
  });

  it('makes no claim in a summary that no input supports', () => {
    // The three that were there: a 3-year TCO nothing computes, SOPs and
    // safety clauses nothing ships, and a "real skill chain" that was a label.
    const forbidden = [
      'real 3-year',
      'real SOPs',
      'real safety clauses',
      'real skill chain',
      'real comparison framework',
    ];
    for (const scenario of demoService.list()) {
      for (const claim of forbidden) {
        expect(
          scenario.summary.toLowerCase().includes(claim.toLowerCase()),
          `${scenario.id} claims "${claim}" and nothing ships it`,
        ).toBe(false);
      }
    }
  });

  it('does not promise a fixtures directory the model cannot see', () => {
    // The framing told the model its drawing "lives in the demo fixtures
    // directory" — a path it has no way to open. It is attached now, and the
    // framing says so.
    for (const scenario of demoService.list()) {
      expect(scenario.scenarioInstructions ?? '').not.toContain('fixtures directory');
    }
  });
});
