/**
 * What a turn is doing, as a list a person can read.
 *
 * ## Why this is a reducer and not component state
 *
 * The events that describe a turn arrive on three channels — `run_stage` from
 * the Rust command, `model_thinking` and `message_*` from the agent runtime,
 * and `attachment:progress` from the document reader — and they interleave.
 * Folding them in a component would put the ordering rules next to the markup
 * and make them impossible to test. Here they are a pure function over a list,
 * so the ordering contract is a unit test rather than a screenshot.
 *
 * ## The honesty rules
 *
 * 1. **A step is only added when the work it names has actually started.**
 *    There is no step for "probably loading" and no step that is inferred
 *    from elapsed time.
 * 2. **A step's detail is only what the backend measured.** Page counts,
 *    character counts and durations are passed through; none is derived from
 *    a guess about how long something usually takes.
 * 3. **No step ever carries the model's reasoning.** The thinking step is
 *    built from `model_thinking`, which carries a size and a duration and no
 *    text at all. See `agent-runtime/src/run.ts`.
 * 4. **There are no percentages.** Nothing in this pipeline knows how much of
 *    a turn is left, so nothing claims to.
 */

/** The kinds of work a turn does, in the order they normally happen. */
export type ProgressKind =
  | 'submitted'
  | 'understanding'
  | 'reading'
  | 'indexing'
  | 'selecting'
  | 'routing'
  | 'loading'
  | 'planning'
  | 'starting'
  | 'thinking'
  | 'writing'
  | 'verifying';

export interface ProgressStep {
  /** Stable within a turn. Used as the React key; never an array index. */
  id: string;
  kind: ProgressKind;
  /** The line shown to the person. Already safe to render. */
  label: string;
  /**
   * A second line, when the backend measured something worth saying —
   * "page 3 of 6", "1,559 characters". Absent rather than invented.
   */
  detail?: string;
  startedAt: number;
  /** Set when the step finished. An open step is the one still running. */
  endedAt?: number;
}

/** The events this reducer folds. A subset of the app's event shapes. */
export type ProgressInput =
  | { kind: 'submitted' }
  | {
      kind: 'stage';
      stage: string;
      elapsedMs: number;
      detail: Record<string, unknown>;
    }
  | { kind: 'thinking'; state: 'start' | 'active' | 'end'; characters: number; elapsedMs: number }
  | { kind: 'text' }
  | { kind: 'attachmentPage'; name: string; page: number | null; pages: number | null; phase: string }
  | { kind: 'done' };

/** Closes whatever is still open. */
function closeOpen(steps: ProgressStep[], at: number): ProgressStep[] {
  return steps.map(step => (step.endedAt === undefined ? { ...step, endedAt: at } : step));
}

/** Closes the open steps and starts a new one. */
function advance(
  steps: ProgressStep[],
  at: number,
  kind: ProgressKind,
  label: string,
  detail?: string,
): ProgressStep[] {
  const open = steps.find(step => step.endedAt === undefined);
  // Re-entering the same kind is an update, not a second row. A model that
  // alternates between reasoning and writing would otherwise produce a
  // column of identical "Thinking" lines.
  if (open && open.kind === kind) {
    return steps.map(step =>
      step.id === open.id ? { ...step, label, detail: detail ?? step.detail } : step,
    );
  }
  return [
    ...closeOpen(steps, at),
    { id: `${kind}-${steps.length}-${at}`, kind, label, detail, startedAt: at },
  ];
}

/** Updates the open step's detail without starting a new one. */
function annotate(steps: ProgressStep[], detail: string): ProgressStep[] {
  const open = steps.find(step => step.endedAt === undefined);
  if (!open) return steps;
  return steps.map(step => (step.id === open.id ? { ...step, detail } : step));
}

/** Closes the most recent step of a kind and gives it a finished label. */
function settle(
  steps: ProgressStep[],
  at: number,
  kind: ProgressKind,
  label: string,
  detail?: string,
): ProgressStep[] {
  for (let i = steps.length - 1; i >= 0; i--) {
    if (steps[i].kind === kind) {
      const next = steps.slice();
      next[i] = {
        ...next[i],
        label,
        detail: detail ?? next[i].detail,
        endedAt: next[i].endedAt ?? at,
      };
      return next;
    }
  }
  return steps;
}

function str(detail: Record<string, unknown>, key: string): string | undefined {
  const value = detail[key];
  return typeof value === 'string' && value.length > 0 ? value : undefined;
}

function num(detail: Record<string, unknown>, key: string): number | undefined {
  const value = detail[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

/** `1559` becomes `1,559`. Locale-free so tests do not depend on the machine. */
export function group(n: number): string {
  return n.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',');
}

/** `1600` becomes `1.6s`; `450` becomes `0.5s`; `92000` becomes `1m 32s`. */
export function humanMs(ms: number): string {
  if (ms < 1000) return `${(ms / 1000).toFixed(1)}s`;
  if (ms < 60_000) return `${(ms / 1000).toFixed(1)}s`;
  const minutes = Math.floor(ms / 60_000);
  const seconds = Math.round((ms % 60_000) / 1000);
  return `${minutes}m ${seconds}s`;
}

/**
 * Folds one event into the step list.
 *
 * Pure and total: an event it does not recognise returns the list unchanged,
 * so a backend that grows a stage before the UI knows about it degrades to
 * showing one line fewer rather than to throwing inside a reducer.
 */
export function applyProgress(
  steps: ProgressStep[],
  input: ProgressInput,
  at: number,
): ProgressStep[] {
  switch (input.kind) {
    case 'submitted':
      // The one step this side raises on its own, and it is not a guess: the
      // request has been handed to the backend and the turn is being written
      // to disk. It is what fills the round trip before `accepted` comes back.
      if (steps.length > 0) return steps;
      return advance(steps, at, 'submitted', 'Sending the request');

    case 'stage':
      return applyStage(steps, input.stage, input.detail, at);

    case 'thinking': {
      if (input.state === 'end') {
        return settle(
          steps,
          at,
          'thinking',
          'Thought it through',
          // No duration here: the row already carries its own measured time
          // on the right. Printing it twice reads as two different numbers.
          `${group(input.characters)} characters of private reasoning`,
        );
      }
      // Size and duration only. The reasoning text itself never reaches this
      // file — it goes to the Thinking panel, which is a separate surface.
      //
      // Labelled "Reasoning" rather than "Thinking" because this row lives in
      // the Activity timeline, which used to be titled "Thinking" as well. A
      // panel called Thinking containing a step called Thinking, neither of
      // which was the model's reasoning, is the confusion this rename ends.
      // The row is still worth having: it marks how long the reasoning pass
      // took, which the prose panel does not.
      const detail =
        input.characters > 0
          ? `${group(input.characters)} characters so far`
          : undefined;
      return advance(steps, at, 'thinking', 'Reasoning', detail);
    }

    case 'text':
      // The first visible token is the only reliable signal that composition
      // has started; no stage on the Rust side can know it.
      return advance(steps, at, 'writing', 'Writing the answer');

    case 'attachmentPage': {
      if (input.phase === 'done') return steps;
      // Page numbers only when the reader actually reported them. A one-page
      // file, or a reader that does not paginate, gets the plain line.
      if (input.page !== null && input.pages !== null && input.pages > 1) {
        // Which phase it is in, because they are different work at very
        // different speeds: 'understanding' is a vision model on the GPU at
        // seconds per page, 'extracting' is a text layer being lifted off at
        // hundreds. A person watching a forty-page scan crawl is owed the
        // difference between the two.
        const verb = input.phase === 'understanding' ? 'Reading' : 'Extracting';
        return annotate(steps, `${verb.toLowerCase()} page ${input.page} of ${input.pages}`);
      }
      // The page count without a page number is still worth saying: it is the
      // first moment anyone knows how long this will take.
      if (input.page === null && input.pages !== null && input.pages > 1) {
        return annotate(steps, `${input.pages} pages`);
      }
      return steps;
    }

    case 'done':
      return closeOpen(steps, at);
  }
}

function applyStage(
  steps: ProgressStep[],
  stage: string,
  detail: Record<string, unknown>,
  at: number,
): ProgressStep[] {
  switch (stage) {
    case 'accepted':
      return advance(steps, at, 'understanding', 'Understanding the request');

    case 'readingAttachment': {
      const name = str(detail, 'name') ?? 'the attachment';
      const index = num(detail, 'index');
      const of = num(detail, 'of');
      const which = index && of && of > 1 ? ` (${index} of ${of})` : '';
      return advance(steps, at, 'reading', `Reading ${name}${which}`);
    }

    case 'attachmentsRead': {
      const files = num(detail, 'files') ?? 0;
      const pages = num(detail, 'pages') ?? 0;
      const characters = num(detail, 'characters') ?? 0;
      const parts = [
        `${files} file${files === 1 ? '' : 's'}`,
        `${pages} page${pages === 1 ? '' : 's'}`,
        `${group(characters)} characters`,
      ];
      // Which reader ran. Without it this line reads the same whether the OCR
      // model transcribed every page or a parser lifted a text layer, and the
      // OCR readout panel below is empty in the second case for a reason
      // nothing on screen gives. A person who attached a scan and saw neither
      // concluded the model was broken.
      //
      // Omitted rather than guessed when the run predates these fields: an
      // older event says nothing about its reader, and "the text layer" would
      // be a claim about a read nobody recorded.
      const byModel = num(detail, 'filesReadByModel');
      const ocrModel = str(detail, 'ocrModelId');
      if (byModel !== undefined) {
        if (byModel === 0) {
          parts.push('read from the text layer, no model needed');
        } else if (ocrModel) {
          parts.push(
            byModel === files
              ? `read by ${ocrModel}`
              : `${byModel} of ${files} read by ${ocrModel}, the rest from the text layer`,
          );
        }
      }
      return settle(steps, at, 'reading', 'Read the attachments', parts.join(' · '));
    }

    case 'indexingDocument': {
      // Real work with a real duration: the pages are being cut into passages
      // that can be retrieved later. Named for what it is rather than for a
      // spinner, and the counts are the ones the cut actually produced.
      const chunks = num(detail, 'chunks');
      const pages = num(detail, 'pagesRead');
      const total = num(detail, 'pages');
      const parts: string[] = [];
      if (chunks !== undefined) {
        parts.push(`${group(chunks)} section${chunks === 1 ? '' : 's'}`);
      }
      if (pages !== undefined && total !== undefined) {
        // "41 of 42 pages" rather than "42 pages" whenever a page produced
        // nothing. The difference is the completeness claim, and rounding it up
        // here would be the same lie the prompt refuses to tell.
        parts.push(pages === total ? `${group(total)} pages` : `${group(pages)} of ${group(total)} pages`);
      }
      return advance(
        steps,
        at,
        'indexing',
        'Indexing the document',
        parts.length > 0 ? parts.join(' · ') : undefined,
      );
    }

    case 'selectingContext': {
      // What this turn could actually afford, against what exists. Both
      // numbers, always: "18 sections" alone reads as the whole document.
      const included = num(detail, 'chunksIncluded');
      const total = num(detail, 'chunksTotal');
      const measured = num(detail, 'measuredTokens');
      const window = num(detail, 'servedWindow');
      const parts: string[] = [];
      if (included !== undefined && total !== undefined) {
        parts.push(`${group(included)} of ${group(total)} sections`);
      }
      // Only when the server actually counted. An estimate must not be shown in
      // the place a measurement goes — see `measuredTokens`, which is null
      // whenever nobody counted.
      if (measured !== undefined && window !== undefined) {
        parts.push(`${group(measured)} of ${group(window)} tokens`);
      }
      const refits = num(detail, 'refits');
      if (refits !== undefined && refits > 0) {
        parts.push(`refitted ${group(refits)}×`);
      }
      // Its own row rather than a relabelling of the indexing one: indexing
      // happens the moment the pages are read, and this happens after the
      // model has loaded, which on a cold start is a minute later. Collapsing
      // them would put one duration on two pieces of work.
      return advance(
        steps,
        at,
        'selecting',
        'Preparing the relevant sections',
        parts.length > 0 ? parts.join(' · ') : undefined,
      );
    }

    case 'routing':
      return advance(steps, at, 'routing', 'Choosing a model');

    case 'routed': {
      const model = str(detail, 'modelName');
      const role = str(detail, 'role');
      return settle(
        steps,
        at,
        'routing',
        model ? `Chose ${model}` : 'Chose a model',
        role ?? undefined,
      );
    }

    case 'loadingModel': {
      const model = str(detail, 'modelName');
      const onGpu = detail['fullyOnGpu'];
      return advance(
        steps,
        at,
        'loading',
        model ? `Loading ${model}` : 'Loading the model',
        onGpu === false ? 'partly on the CPU' : undefined,
      );
    }

    case 'modelReady': {
      if (detail['warm'] === true) {
        // Nothing was loaded, so no loading step exists to settle. Saying the
        // model was already up is the honest account of a step that took no
        // time, and it is the difference a person is owed when the same
        // question answers in two seconds today and forty yesterday.
        return steps;
      }
      // The duration is not passed as detail: the loading row is timed by the
      // panel itself, and `tookMs` is the same measurement. One number, one
      // place. Whatever detail the loading step already had — an offload that
      // could not fit entirely on the GPU, say — is kept.
      return settle(steps, at, 'loading', 'Loaded the model');
    }

    case 'planning':
      return advance(steps, at, 'planning', 'Planning the work');

    case 'generating':
      return advance(steps, at, 'starting', 'Starting generation');

    case 'verifying':
      return advance(steps, at, 'verifying', 'Checking the answer');

    case 'complete':
      return closeOpen(steps, at);

    default:
      return steps;
  }
}

/**
 * The one-line summary shown when the panel is collapsed.
 *
 * While a turn is running it names what is happening now, because that is the
 * question being asked. Afterwards it reports the total, because the question
 * has become how long it took.
 */
export function summariseProgress(
  steps: ProgressStep[],
  live: boolean,
  now: number,
): string {
  if (steps.length === 0) return live ? 'Working' : 'No steps recorded';
  const open = steps.find(step => step.endedAt === undefined);
  if (live && open) return open.label;
  const first = steps[0].startedAt;
  const last = steps.reduce(
    (max, step) => Math.max(max, step.endedAt ?? now),
    first,
  );
  return `Worked for ${humanMs(last - first)}`;
}
