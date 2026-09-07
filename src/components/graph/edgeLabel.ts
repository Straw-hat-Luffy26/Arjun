/**
 * Saying what a link in the graph actually claims.
 *
 * Pure and separate from the component, for the reason given in
 * `./typingOutcome.ts`: vitest here runs without a DOM, and logic inside a
 * component is logic that never gets tested.
 *
 * ## Why this is one function and not a string in the JSX
 *
 * The statistical pass can observe exactly one thing about two terms — that
 * they were in the same passage, and in how many. It cannot observe that a
 * supplier supplies anything. A line drawn between two nodes with no words on
 * it invites the reader to supply the relation themselves, and they will supply
 * a plausible one; that is the same failure as printing a number nobody
 * measured. So every place the interface describes an edge comes through here,
 * and a named relation is shown only where the typing pass actually verified
 * one against a quoted passage.
 */
import type { GraphEdge } from '../../services/notebook.service';

function passages(weight: number): string {
  return `${weight} ${weight === 1 ? 'passage' : 'passages'}`;
}

/**
 * What an edge claims, in words, read from one end of it.
 *
 * `appearsIn` runs from a file to a term found in it, and is the one edge in
 * the graph with a direction — which is why it is the one edge drawn with an
 * arrowhead. The same fact reads differently from each end, so `from` says
 * which end the reader is standing on: a file *contains* a term, a term
 * *appears in* a file. Omit `from` and it is described from the file's side,
 * the direction the arrow points.
 *
 * `appears with` is the untyped co-occurrence and says only what was observed.
 * Anything else is a relation the typing pass supplied and the verification
 * gate accepted against a quoted passage.
 */
export function describeEdge(edge: GraphEdge, from?: string): string {
  if (edge.kind === 'appearsIn') {
    const atTerm = from !== undefined && from === edge.target;
    return atTerm
      ? `appears in, in ${passages(edge.weight)}`
      : `contains, in ${passages(edge.weight)}`;
  }
  return edge.relation
    ? `${edge.relation}, in ${passages(edge.weight)}`
    : `appears with, in ${passages(edge.weight)}`;
}
