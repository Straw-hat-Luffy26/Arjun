/**
 * Reading a batch of "add these files to the notebook" outcomes.
 *
 * Pure, and separate from the screen, for the reason every decision in this
 * codebase is: vitest runs with `environment: 'node'` and there is no DOM, so a
 * component cannot be rendered in a test. Logic that lives in a component is
 * logic that is never tested. See `src/components/run/context-entities.ts`.
 *
 * The distinction this module exists to preserve: **three outcomes, not two**.
 * A file that could not be read and a file the notebook already had are both
 * "not added", and collapsing them into one number is how a screen ends up
 * telling somebody their document failed when it was simply already there.
 */
import type { AddedDocument } from '../../services/notebook.service';

export interface AddTally {
  /** Files this call actually put into the notebook. */
  added: number;
  /** Files the notebook already had. Not a failure. */
  already: number;
  /** Files that could not be read at all — no content address was produced. */
  failed: number;
}

/**
 * Counts a batch.
 *
 * `failed` is defined by the absence of a `sha256`, not by the presence of a
 * `problem`: a document that was read and stored but was already a member also
 * carries a problem sentence, and counting that as a failure would be the exact
 * conflation this module exists to prevent.
 */
export function tallyAdds(outcomes: readonly AddedDocument[]): AddTally {
  const added = outcomes.filter((outcome) => outcome.added).length;
  const failed = outcomes.filter((outcome) => !outcome.added && outcome.sha256 === null).length;
  return { added, already: outcomes.length - added - failed, failed };
}

/**
 * One sentence a person can read, naming only what happened.
 *
 * A count of zero is omitted rather than printed: "0 could not be read" invites
 * somebody to go looking for a failure that did not occur.
 */
export function summariseAdds(outcomes: readonly AddedDocument[]): string {
  const { added, already, failed } = tallyAdds(outcomes);

  const parts: string[] = [];
  if (added > 0) parts.push(`${added} added`);
  if (already > 0) parts.push(`${already} already here`);
  if (failed > 0) parts.push(`${failed} could not be read`);
  return parts.length > 0 ? parts.join(' · ') : 'Nothing to add.';
}

/**
 * The rows worth showing under the summary.
 *
 * Everything that carries a problem, whatever kind — a person looking at the
 * screen wants the names, and the summary line has already said how many.
 */
export function problemRows(outcomes: readonly AddedDocument[]): AddedDocument[] {
  return outcomes.filter((outcome) => outcome.problem !== null && outcome.problem !== undefined);
}
