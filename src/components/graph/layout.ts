/**
 * How big a node is drawn.
 *
 * All that is left of what was once a one-shot force layout. The physics moved
 * to `./simulation.ts`, which runs frame by frame and settles rather than
 * solving the arrangement once; this size rule stayed behind because both the
 * canvas and the simulation need it and neither owns it. Pure, so vitest —
 * which runs here with `environment: 'node'` and no DOM — can test it at all.
 */

/**
 * The radius to draw a node at, from how connected it is.
 *
 * Degree rather than occurrence count: the picture is about connection, and a
 * term mentioned often but linked to nothing is not the one to make large.
 * Bounded at both ends so a hub cannot swallow the canvas and a leaf stays
 * clickable.
 */
export function radiusFor(degree: number): number {
  const MIN = 4;
  const MAX = 18;
  return Math.min(MAX, MIN + Math.sqrt(Math.max(0, degree)) * 2.2);
}
