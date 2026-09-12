/**
 * Carrying a notebook from the Notebooks page to the main composer.
 *
 * ## Why this is not just an event
 *
 * The obvious implementation — dispatch a `CustomEvent` and let the chat
 * surface listen — does not work, and fails in a way that looks like nothing
 * happening. The chat surface is a route. While somebody is looking at the
 * Notebooks page it is *not mounted*, so it has no listener, so the event is
 * dispatched into nothing. By the time navigation mounts the surface the event
 * is long gone, and the person arrives at an empty composer having pressed a
 * button that appeared to do nothing.
 *
 * So the hand-off is a value, left where the surface will find it, plus an
 * event for the case where the surface *is* already mounted. Reading it clears
 * it: a hand-off is consumed once, and arriving at the chat a second time must
 * not silently re-attach a notebook somebody has since cleared.
 */

/** What the Notebooks page hands over. */
export interface NotebookHandoff {
  notebookId: string;
  notebookName: string;
  /**
   * The sources ticked on the Notebooks page, or `null` for "it had narrowed
   * nothing".
   *
   * `null` and `[]` are deliberately different. An empty list would mean the
   * person deselected everything, and reading that as "all of them" is the
   * exact mistake the selection type exists to prevent.
   */
  sourceSha256s: string[] | null;
  /** A question to place in the composer. Never sent on anybody's behalf. */
  prompt: string | null;
}

/** The event name, for a surface that is already on screen. */
export const HANDOFF_EVENT = 'arjun:ask-notebook';

/**
 * The one pending hand-off.
 *
 * Module-level rather than context state because the writer and the reader are
 * never mounted at the same time — which is the whole problem this solves.
 */
let pending: NotebookHandoff | null = null;

/** Leaves a notebook for the composer, and tells it if it is listening. */
export function handOffNotebook(handoff: NotebookHandoff): void {
  // The value first, and unconditionally. It is what makes the hand-off
  // survive the navigation; the event is only an optimisation for a surface
  // that happens to be mounted already.
  pending = handoff;
  // Guarded because the value half of this module has to work without a DOM —
  // in a test, and in any context where this is imported for its types alone.
  // A hand-off that threw here would be lost *and* would take its caller with
  // it, which is strictly worse than one that simply arrives a moment later
  // when the surface mounts and reads it.
  if (typeof window === 'undefined') return;
  window.dispatchEvent(new CustomEvent(HANDOFF_EVENT, { detail: handoff }));
}

/**
 * Takes the pending hand-off, if there is one, and clears it.
 *
 * Clearing on read is what stops one being applied twice: returning to the chat
 * later must not re-attach a notebook the person has since taken off.
 */
export function takePendingHandoff(): NotebookHandoff | null {
  const held = pending;
  pending = null;
  return held;
}

/** Drops a pending hand-off without applying it. For tests, and for teardown. */
export function clearPendingHandoff(): void {
  pending = null;
}

/**
 * Whether a value off the wire is a usable hand-off.
 *
 * The event carries whatever was dispatched, and a listener that trusts its own
 * `detail` shape will eventually read `undefined.notebookId`.
 */
export function isHandoff(value: unknown): value is NotebookHandoff {
  if (typeof value !== 'object' || value === null) return false;
  const candidate = value as Partial<NotebookHandoff>;
  return (
    typeof candidate.notebookId === 'string' &&
    candidate.notebookId.length > 0 &&
    typeof candidate.notebookName === 'string'
  );
}
