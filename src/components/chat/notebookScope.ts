/**
 * The `/notebook` command, and the scope one turn is frozen at.
 *
 * Pure functions, deliberately. Everything here is about *when* a chooser
 * opens, *what* it filters to, and *which* scope a message carries — three
 * decisions that would otherwise live inside a component's keydown handler,
 * where they could not be tested and could not be reasoned about.
 *
 * ## The two rules worth stating out loud
 *
 * **The command token never reaches the model.** Choosing a notebook rewrites
 * the draft to remove `/notebook…` and leaves everything else exactly as it
 * was. Somebody half-way through "what is the rated duty of P-101?" who types
 * `/notebook` to attach their sources must get their sentence back, not a
 * cleared box.
 *
 * **A turn's scope is frozen when it is submitted, not when it is sent.** A
 * message typed against notebook A and queued behind a run in flight is
 * answered from A, even if the chip says B by the time the queue drains. The
 * alternative — resolving the scope at send time — answers a queued question
 * from whatever happens to be selected a minute later, which nobody asked for
 * and nobody can see happening.
 */
import {
  allSources,
  noSources,
  someSources,
  type ResearchScope,
  type SourceSelection,
} from '../../services/agent.service';

/** The command word, including its slash. */
export const NOTEBOOK_COMMAND = '/notebook';

/** Where a `/notebook` command sits in the draft, and what it is searching. */
export interface CommandSpan {
  /** Index of the `/` in the draft. */
  start: number;
  /** Index one past the end of the command, exclusive. */
  end: number;
  /**
   * What has been typed after the command word, trimmed.
   *
   * Empty while the command word itself is still being typed, so the chooser
   * opens showing everything rather than filtering on a fragment of its name.
   */
  query: string;
  /** True once the whole word has been typed, so a selection may be made. */
  complete: boolean;
}

/**
 * Finds the `/notebook` command the caret is inside, if any.
 *
 * A command must start the draft or follow whitespace, so a path like
 * `docs/notebook.md` in the middle of a sentence does not open a chooser. It
 * runs to the end of its line, which is what lets a notebook name with spaces
 * in it be typed as a search.
 *
 * Returns `null` once the caret moves off the command, so the chooser closes
 * when you click away without needing a separate dismissal.
 */
export function findNotebookCommand(text: string, caret: number): CommandSpan | null {
  const clamped = Math.max(0, Math.min(caret, text.length));

  // The line the caret is on. A command cannot span a newline: otherwise
  // pressing Enter on a multi-line draft leaves the chooser open over text
  // that is no longer part of the command.
  const lineStart = text.lastIndexOf('\n', Math.max(0, clamped - 1)) + 1;
  const newline = text.indexOf('\n', clamped);
  const lineEnd = newline === -1 ? text.length : newline;
  const line = text.slice(lineStart, lineEnd);
  const caretInLine = clamped - lineStart;

  // The last slash at or before the caret that begins a word.
  let slash = -1;
  for (let i = caretInLine; i >= 0; i--) {
    if (line[i] !== '/') continue;
    if (i === 0 || /\s/.test(line[i - 1])) {
      slash = i;
      break;
    }
  }
  if (slash === -1) return null;

  const rest = line.slice(slash);
  const word = rest.split(/\s/, 1)[0];

  // Still typing the word: `/n`, `/note`. Opens so the chooser is visible
  // before the last letter, and offers no query of its own.
  if (NOTEBOOK_COMMAND.startsWith(word) && word !== NOTEBOOK_COMMAND) {
    return {
      start: lineStart + slash,
      end: lineStart + slash + word.length,
      query: '',
      complete: false,
    };
  }
  if (word !== NOTEBOOK_COMMAND) return null;

  return {
    start: lineStart + slash,
    end: lineEnd,
    query: rest.slice(NOTEBOOK_COMMAND.length).trim(),
    complete: true,
  };
}

/**
 * Removes the command from the draft, leaving the rest of the question.
 *
 * Collapses the whitespace the command sat between, so removing it from the
 * middle of a sentence does not leave a double space, and trims a trailing
 * space at the end so the draft does not look edited.
 */
export function consumeCommand(text: string, span: CommandSpan): string {
  const before = text.slice(0, span.start);
  const after = text.slice(span.end);
  const joined =
    before.length > 0 && after.length > 0 && /\s$/.test(before) && /^\s/.test(after)
      ? before + after.replace(/^\s+/, '')
      : before + after;
  // Only a trailing space is trimmed — a trailing newline is the person's.
  return joined.replace(/[^\S\n]+$/, '');
}

/** The caret position after the command is consumed. */
export function caretAfterConsume(span: CommandSpan): number {
  return span.start;
}

/** One notebook as the chooser needs to draw it. */
export interface ChoosableNotebook {
  id: string;
  name: string;
  /** Sources in the notebook, from its own record. */
  documentCount: number;
}

/**
 * Filters and orders the chooser's list.
 *
 * A name that *starts* with the query beats one that merely contains it, which
 * is what makes typing the first few letters of a notebook reach it first.
 * Ties fall back to the order the backend returned, so the list does not
 * reshuffle between keystrokes that match nothing new.
 */
export function filterNotebooks<T extends ChoosableNotebook>(
  notebooks: readonly T[],
  query: string,
): T[] {
  const needle = query.trim().toLowerCase();
  if (!needle) return [...notebooks];

  const scored: { notebook: T; rank: number; index: number }[] = [];
  notebooks.forEach((notebook, index) => {
    const name = notebook.name.toLowerCase();
    const rank = name.startsWith(needle) ? 0 : name.includes(needle) ? 1 : -1;
    if (rank >= 0) scored.push({ notebook, rank, index });
  });
  scored.sort((a, b) => a.rank - b.rank || a.index - b.index);
  return scored.map(entry => entry.notebook);
}

/**
 * Moves the highlighted row, wrapping at both ends.
 *
 * Wrapping rather than clamping because the list is short and somebody holding
 * Down expects to come back round, not to stick silently at the bottom.
 * Returns `-1` for an empty list, which is "nothing to select".
 */
export function moveHighlight(current: number, delta: number, length: number): number {
  if (length <= 0) return -1;
  const next = (current + delta) % length;
  return next < 0 ? next + length : next;
}

/**
 * What the composer is currently pointed at.
 *
 * One value rather than two pieces of state: a notebook with no selection and
 * a selection with no notebook are both meaningless, and keeping them apart is
 * how one of them ends up stale.
 */
export interface NotebookScope {
  notebookId: string;
  notebookName: string;
  /** Sources in the notebook right now, for the chip's wording. */
  documentCount: number;
  selection: SourceSelection;
}

/**
 * The scope a turn carries, frozen at the moment it is submitted.
 *
 * `all` is resolved to the concrete list of sources that existed when the turn
 * was frozen. Carrying the word "all" through a queue would mean a source
 * added while the question waited silently joined a turn scoped before it
 * existed — and the answer would cite a document the person had not added when
 * they asked.
 *
 * The one case where `all` survives is a notebook whose readable-source list
 * could not be established. Sending `{mode:'all'}` there is not a widening:
 * the backend resolves it against the same notebook under the same owner, and
 * the alternative — an empty subset — is refused rather than silently reading
 * everything.
 */
export function freezeScope(
  scope: NotebookScope | null,
  readableSha256s: readonly string[],
): ResearchScope | undefined {
  if (!scope) return undefined;

  let selection: SourceSelection;
  switch (scope.selection.mode) {
    case 'all':
      selection =
        readableSha256s.length > 0 ? someSources([...readableSha256s]) : allSources();
      break;
    case 'subset':
      selection = someSources([...scope.selection.sha256s]);
      break;
    case 'none':
      selection = noSources();
      break;
  }

  return { notebookId: scope.notebookId, sources: selection };
}

/**
 * Whether two scopes would send a question to the same sources.
 *
 * Used to tell whether a queued turn's frozen scope still matches the chip, so
 * the queue can say which notebook each waiting message is going to when they
 * are no longer the same one.
 */
export function sameScope(
  a: ResearchScope | undefined,
  b: ResearchScope | undefined,
): boolean {
  if (!a || !b) return a === b;
  if (a.notebookId !== b.notebookId) return false;
  if (a.sources.mode !== b.sources.mode) return false;
  if (a.sources.mode === 'subset' && b.sources.mode === 'subset') {
    if (a.sources.sha256s.length !== b.sources.sha256s.length) return false;
    const left = [...a.sources.sha256s].sort();
    const right = [...b.sources.sha256s].sort();
    return left.every((sha, i) => sha === right[i]);
  }
  return true;
}
