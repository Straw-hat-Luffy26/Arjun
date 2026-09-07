/**
 * The progress reducer's contract.
 *
 * Three things are being pinned here, and they are the three this feature can
 * regress into being dishonest about:
 *
 *  1. **No step exists for work that has not started.** Every row in the panel
 *     traces back to an event the backend emitted when it began that work.
 *  2. **No detail is invented.** Page counts, character counts and durations
 *     appear only when the event carried them.
 *  3. **No reasoning is ever rendered.** The thinking row is built from a size
 *     and a duration; there is no field it could carry text in.
 */
import { describe, expect, it } from 'vitest';
import {
  applyProgress,
  group,
  humanMs,
  summariseProgress,
  type ProgressInput,
  type ProgressStep,
} from './runProgress';

/** Folds a script of events, advancing the clock a fixed step between them. */
function fold(inputs: ProgressInput[], start = 1_000, step = 100): ProgressStep[] {
  let steps: ProgressStep[] = [];
  inputs.forEach((input, i) => {
    steps = applyProgress(steps, input, start + i * step);
  });
  return steps;
}

const stage = (
  name: string,
  detail: Record<string, unknown> = {},
  elapsedMs = 0,
): ProgressInput => ({ kind: 'stage', stage: name, elapsedMs, detail });

describe('applyProgress: only real work becomes a step', () => {
  it('starts with nothing at all', () => {
    expect(applyProgress([], { kind: 'done' }, 0)).toEqual([]);
  });

  it('the submitted step is raised once and never duplicated', () => {
    const steps = fold([{ kind: 'submitted' }, { kind: 'submitted' }]);
    expect(steps).toHaveLength(1);
    expect(steps[0].label).toBe('Sending the request');
  });

  it('an unrecognised stage adds nothing rather than throwing', () => {
    const steps = fold([{ kind: 'submitted' }, stage('somethingNewFromTheBackend')]);
    expect(steps).toHaveLength(1);
  });

  it('a warm model produces no loading step, because nothing was loaded', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('accepted'),
      stage('routing'),
      stage('routed', { modelName: 'gemma-4-E4B-it', role: 'reasoning' }),
      stage('modelReady', { warm: true, tookMs: 0 }),
    ]);
    expect(steps.map(s => s.kind)).not.toContain('loading');
  });

  it('a cold model produces a loading step that settles when the model is ready', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('accepted'),
      stage('loadingModel', { modelName: 'gemma-4-E4B-it', fullyOnGpu: true }),
      stage('modelReady', { warm: false, tookMs: 41_200 }),
    ]);
    const loading = steps.find(s => s.kind === 'loading');
    expect(loading?.label).toBe('Loaded the model');
    expect(loading?.endedAt).toBeDefined();
    // The duration is NOT repeated into the detail: the panel times every row
    // itself and prints that on the right, so putting it here too showed the
    // same number twice on one line.
    expect(loading?.detail).toBeUndefined();
  });

  it('keeps a detail the loading step already had rather than clearing it', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('loadingModel', { modelName: 'gemma-4-E4B-it', fullyOnGpu: false }),
      stage('modelReady', { warm: false, tookMs: 12_000 }),
    ]);
    expect(steps.find(s => s.kind === 'loading')?.detail).toBe('partly on the CPU');
  });
});

describe('applyProgress: the order of a turn', () => {
  it('follows the real sequence and leaves exactly one step open', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('accepted'),
      stage('routing'),
      stage('routed', { modelName: 'gemma-4-E4B-it', role: 'reasoning' }),
      stage('planning'),
      stage('generating'),
      { kind: 'thinking', state: 'start', characters: 0, elapsedMs: 0 },
    ]);
    expect(steps.map(s => s.kind)).toEqual([
      'submitted',
      'understanding',
      'routing',
      'planning',
      'starting',
      'thinking',
    ]);
    expect(steps.filter(s => s.endedAt === undefined)).toHaveLength(1);
    expect(steps[steps.length - 1].kind).toBe('thinking');
  });

  it('the first visible token opens the writing step and closes the rest', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('generating'),
      { kind: 'thinking', state: 'start', characters: 10, elapsedMs: 0 },
      { kind: 'thinking', state: 'end', characters: 900, elapsedMs: 12_000 },
      { kind: 'text' },
    ]);
    const open = steps.filter(s => s.endedAt === undefined);
    expect(open).toHaveLength(1);
    expect(open[0].kind).toBe('writing');
    expect(open[0].label).toBe('Writing the answer');
  });

  it('repeated text events do not add a second writing row', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('generating'),
      { kind: 'text' },
      { kind: 'text' },
      { kind: 'text' },
    ]);
    expect(steps.filter(s => s.kind === 'writing')).toHaveLength(1);
  });

  it('done closes everything that was still open', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('generating'),
      { kind: 'text' },
      { kind: 'done' },
    ]);
    expect(steps.every(s => s.endedAt !== undefined)).toBe(true);
  });
});

describe('applyProgress: attachments report what the reader actually knew', () => {
  it('names the file and its position when there is more than one', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('readingAttachment', { name: 'scan.pdf', index: 2, of: 3 }),
    ]);
    expect(steps[steps.length - 1].label).toBe('Reading scan.pdf (2 of 3)');
  });

  it('omits the position for a single attachment', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('readingAttachment', { name: 'scan.pdf', index: 1, of: 1 }),
    ]);
    expect(steps[steps.length - 1].label).toBe('Reading scan.pdf');
  });

  it('shows a page counter only when the reader reported real page numbers', () => {
    const withPages = fold([
      { kind: 'submitted' },
      stage('readingAttachment', { name: 'scan.pdf', index: 1, of: 1 }),
      { kind: 'attachmentPage', name: 'scan.pdf', page: 3, pages: 6, phase: 'understanding' },
    ]);
    // The verb is part of the detail now: 'understanding' is a vision model on
    // the GPU and 'extracting' is a text layer being lifted off, and they run at
    // very different speeds. See the large-document steps below.
    expect(withPages[withPages.length - 1].detail).toBe('reading page 3 of 6');

    const withoutPages = fold([
      { kind: 'submitted' },
      stage('readingAttachment', { name: 'photo.png', index: 1, of: 1 }),
      { kind: 'attachmentPage', name: 'photo.png', page: null, pages: null, phase: 'reading' },
    ]);
    expect(withoutPages[withoutPages.length - 1].detail).toBeUndefined();
  });

  it('does not paginate a single-page file', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('readingAttachment', { name: 'photo.png', index: 1, of: 1 }),
      { kind: 'attachmentPage', name: 'photo.png', page: 1, pages: 1, phase: 'understanding' },
    ]);
    expect(steps[steps.length - 1].detail).toBeUndefined();
  });

  it('settles the reading step with counts the backend measured', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('readingAttachment', { name: 'scan.pdf', index: 1, of: 1 }),
      stage('attachmentsRead', { files: 1, pages: 6, characters: 1559 }),
    ]);
    const reading = steps.find(s => s.kind === 'reading');
    expect(reading?.label).toBe('Read the attachments');
    expect(reading?.detail).toBe('1 file · 6 pages · 1,559 characters');
  });
});

describe('applyProgress: the thinking row carries no reasoning', () => {
  const secret = 'the model private reasoning that must never be shown';

  it('reports size and duration and nothing else', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('generating'),
      { kind: 'thinking', state: 'start', characters: 0, elapsedMs: 0 },
      { kind: 'thinking', state: 'active', characters: 1240, elapsedMs: 9_000 },
    ]);
    const thinking = steps[steps.length - 1];
    // "Reasoning" rather than "Thinking": this row lives in the Activity
    // timeline, and the panel that shows the reasoning prose is the one called
    // Thinking. The two used to share the word and neither was the other.
    expect(thinking.label).toBe('Reasoning');
    expect(thinking.detail).toBe('1,240 characters so far');
  });

  it('cannot render text even if a caller tried to smuggle it through', () => {
    // Asserts the rendered result rather than the input type, because a
    // future field carrying text would pass a type check and fail this.
    const steps = fold([
      { kind: 'submitted' },
      { kind: 'thinking', state: 'start', characters: secret.length, elapsedMs: 0 },
      { kind: 'thinking', state: 'end', characters: secret.length, elapsedMs: 4_000 },
    ]);
    const rendered = JSON.stringify(steps);
    expect(rendered).not.toContain(secret);
    expect(rendered).not.toContain('reasoning that');
  });

  it('the finished row reports the size it produced, timed by the panel', () => {
    const steps = fold([
      { kind: 'submitted' },
      { kind: 'thinking', state: 'start', characters: 10, elapsedMs: 0 },
      { kind: 'thinking', state: 'end', characters: 3200, elapsedMs: 92_000 },
    ]);
    const thinking = steps.find(s => s.kind === 'thinking');
    expect(thinking?.label).toBe('Thought it through');
    expect(thinking?.detail).toBe('3,200 characters of private reasoning');
    expect(thinking?.endedAt).toBeDefined();
  });
});

describe('summariseProgress', () => {
  it('names what is happening now while the turn is live', () => {
    const steps = fold([{ kind: 'submitted' }, stage('generating')]);
    expect(summariseProgress(steps, true, 2_000)).toBe('Starting generation');
  });

  it('reports the total once the turn is over', () => {
    const steps = fold([{ kind: 'submitted' }, stage('generating'), { kind: 'done' }], 0, 1_500);
    expect(summariseProgress(steps, false, 10_000)).toBe('Worked for 3.0s');
  });

  it('says so plainly when there are no steps', () => {
    expect(summariseProgress([], false, 0)).toBe('No steps recorded');
  });
});

describe('the large-document steps say what actually happened', () => {
  it('names the indexing step with the passages the cut really produced', () => {
    const steps = fold([
      { kind: 'submitted' },
      stage('accepted'),
      stage('readingAttachment', { name: 'drawing.pdf', index: 1, of: 1 }),
      stage('attachmentsRead', { files: 1, pages: 42, characters: 120_000 }),
      stage('indexingDocument', { documents: 1, pages: 42, pagesRead: 42, chunks: 128 }),
    ]);
    const indexing = steps.find(step => step.kind === 'indexing');
    expect(indexing?.label).toBe('Indexing the document');
    expect(indexing?.detail).toContain('128 sections');
    expect(indexing?.detail).toContain('42 pages');
  });

  /**
   * The completeness claim, in the one place a person actually reads. A scan
   * whose page 17 produced nothing is not a scan that was read, and rounding
   * "41 of 42" up to "42" here would be the same lie the prompt refuses to
   * tell.
   */
  it('does not round an unread page up into a fully read document', () => {
    const steps = fold([
      stage('indexingDocument', { documents: 1, pages: 42, pagesRead: 41, chunks: 120 }),
    ]);
    expect(steps.find(step => step.kind === 'indexing')?.detail).toContain('41 of 42 pages');
  });

  it('reports what the turn could afford against what exists', () => {
    const steps = fold([
      stage('indexingDocument', { documents: 1, pages: 42, pagesRead: 42, chunks: 128 }),
      stage('modelReady', { warm: true }),
      stage('selectingContext', {
        documents: 1,
        chunksTotal: 128,
        chunksIncluded: 18,
        chunksOmitted: 110,
        servedWindow: 8192,
        measuredTokens: 6052,
        refits: 0,
        complete: false,
      }),
    ]);
    const selecting = steps.find(step => step.kind === 'selecting');
    expect(selecting?.label).toBe('Preparing the relevant sections');
    expect(selecting?.detail).toContain('18 of 128 sections');
    expect(selecting?.detail).toContain('6,052 of 8,192 tokens');
  });

  /**
   * A token figure in this row is a count from the server's own tokeniser. A
   * server that does not offer one sends null, and the row must then say
   * nothing about tokens rather than showing an estimate in a measurement's
   * place.
   */
  it('shows no token figure when nobody counted one', () => {
    const steps = fold([
      stage('selectingContext', {
        documents: 1,
        chunksTotal: 40,
        chunksIncluded: 12,
        servedWindow: 8192,
        measuredTokens: null,
        refits: 0,
      }),
    ]);
    const selecting = steps.find(step => step.kind === 'selecting');
    expect(selecting?.detail).toContain('12 of 40 sections');
    expect(selecting?.detail).not.toContain('tokens');
  });

  /**
   * Indexing and selection are minutes apart on a cold start — the model has to
   * load in between. One row with one duration would put that whole wait on
   * whichever of them it was attached to.
   */
  it('keeps indexing and selection as two steps with two durations', () => {
    const steps = fold([
      stage('indexingDocument', { documents: 1, pages: 42, pagesRead: 42, chunks: 128 }),
      stage('loadingModel', { modelName: 'Qwen3.5 9B' }),
      stage('modelReady', { warm: false }),
      stage('selectingContext', { chunksTotal: 128, chunksIncluded: 18 }),
    ]);
    expect(steps.filter(step => step.kind === 'indexing')).toHaveLength(1);
    expect(steps.filter(step => step.kind === 'selecting')).toHaveLength(1);
    expect(steps.find(step => step.kind === 'indexing')?.endedAt).toBeDefined();
  });

  /**
   * Reading a page with a vision model and lifting a text layer off one are
   * different work at very different speeds. A person watching a forty-page
   * scan crawl is owed the difference.
   */
  it('distinguishes reading a page from extracting one', () => {
    const reading = fold([
      stage('readingAttachment', { name: 'scan.pdf', index: 1, of: 1 }),
      { kind: 'attachmentPage', name: 'scan.pdf', page: 17, pages: 42, phase: 'understanding' },
    ]);
    expect(reading.find(step => step.kind === 'reading')?.detail).toBe('reading page 17 of 42');

    const extracting = fold([
      stage('readingAttachment', { name: 'report.pdf', index: 1, of: 1 }),
      { kind: 'attachmentPage', name: 'report.pdf', page: 3, pages: 12, phase: 'extracting' },
    ]);
    expect(extracting.find(step => step.kind === 'reading')?.detail).toBe(
      'extracting page 3 of 12',
    );
  });
});

describe('formatting helpers', () => {
  it('groups thousands without depending on the machine locale', () => {
    expect(group(1559)).toBe('1,559');
    expect(group(999)).toBe('999');
    expect(group(1234567)).toBe('1,234,567');
  });

  it('reports durations at a readable resolution', () => {
    expect(humanMs(450)).toBe('0.5s');
    expect(humanMs(1_600)).toBe('1.6s');
    expect(humanMs(92_000)).toBe('1m 32s');
  });
});
