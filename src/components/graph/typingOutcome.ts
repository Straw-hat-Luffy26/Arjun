/**
 * Reading the result of the typing pass honestly.
 *
 * Pure and separate from the component, because vitest here runs without a DOM
 * and logic inside a component is logic that never gets tested.
 *
 * ## What this module refuses to do
 *
 * It will not report a typing pass as a success by naming only what it kept. The
 * gate in `knowledge::graph::typing` drops any claim whose quote is not in the
 * passage it cited, and a pass that dropped most of what the model proposed is
 * reporting something real: the wrong model for the job, or a document of
 * scanned noise. Hiding that behind "38 terms typed" would turn a failing
 * extraction into what looks like a small graph — the exact substitution this
 * repository's rules exist to prevent.
 */
import type { TypingOutcome } from '../../services/notebook.service';

/** Proposals the verification gate rejected, for any reason. */
export function totalDropped(outcome: TypingOutcome): number {
  return (
    outcome.droppedUnknownType +
    outcome.droppedUnknownTerm +
    outcome.droppedUncited +
    outcome.droppedMisquoted
  );
}

/**
 * The headline: how much of the graph now carries a type.
 *
 * Always both numbers. "38 of 412 terms typed" is the truth; "38 terms typed"
 * invites the reader to assume the other 374 do not exist.
 */
export function describeCoverage(outcome: TypingOutcome): string {
  if (outcome.totalTerms === 0) return 'Nothing to type yet.';
  return `${outcome.typedTerms} of ${outcome.totalTerms} terms typed`;
}

/** What the pass did to documents. */
export function describeRun(outcome: TypingOutcome): string {
  const parts: string[] = [];
  if (outcome.documentsTyped > 0) parts.push(`${outcome.documentsTyped} read`);
  if (outcome.documentsSkipped > 0) parts.push(`${outcome.documentsSkipped} already typed`);
  if (outcome.documentsFailed > 0) parts.push(`${outcome.documentsFailed} failed`);
  return parts.length > 0 ? parts.join(' · ') : 'Nothing to do.';
}

/**
 * What the model offered and what survived, named by reason.
 *
 * Returns `null` when the model proposed nothing at all — there is no honest
 * sentence to write about a pass with no input, and a row of zeroes reads as a
 * malfunction.
 */
export function describeVerification(outcome: TypingOutcome): string | null {
  const proposed = outcome.proposedNodes + outcome.proposedEdges;
  if (proposed === 0) return null;

  const kept = outcome.keptNodes + outcome.keptEdges;
  const reasons: string[] = [];
  if (outcome.droppedMisquoted > 0) {
    reasons.push(`${outcome.droppedMisquoted} could not be found in the passage cited`);
  }
  if (outcome.droppedUncited > 0) {
    reasons.push(`${outcome.droppedUncited} cited a passage that was not sent`);
  }
  if (outcome.droppedUnknownTerm > 0) {
    reasons.push(`${outcome.droppedUnknownTerm} named a term not in the graph`);
  }
  if (outcome.droppedUnknownType > 0) {
    reasons.push(`${outcome.droppedUnknownType} used a type outside the list`);
  }

  const head = `${kept} of ${proposed} claims verified`;
  return reasons.length > 0 ? `${head} — ${reasons.join(', ')}.` : `${head}.`;
}

/**
 * Whether the yield is low enough to be worth flagging.
 *
 * A low yield is a finding, not a bug to hide: it usually means the model is too
 * small for the job or the document did not survive OCR. The threshold is
 * deliberately generous — this warns, it does not fail anything.
 */
export function yieldIsPoor(outcome: TypingOutcome): boolean {
  const proposed = outcome.proposedNodes + outcome.proposedEdges;
  if (proposed < 10) return false;
  const kept = outcome.keptNodes + outcome.keptEdges;
  return kept / proposed < 0.5;
}
