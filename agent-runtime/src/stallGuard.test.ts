/**
 * Whether a run that is merely slow is allowed to finish.
 *
 * The run had a ceiling on total time and nothing else, so a local 4B decoding
 * at five tokens a second - eight minutes of healthy work for one answer - was
 * stopped partway by a ten-minute budget and the person was shown "it ran past
 * the time its plan allowed" after waiting the whole ten minutes for nothing.
 *
 * A ceiling cannot tell slow from stuck. This guard can, because it asks a
 * different question: has anything happened lately?
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { createStallGuard } from "./run.js";

describe("the stall guard", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("stops a run that goes quiet", () => {
    const stalled = vi.fn();
    createStallGuard(1000, stalled).progress();

    vi.advanceTimersByTime(999);
    expect(stalled).not.toHaveBeenCalled();
    vi.advanceTimersByTime(1);
    expect(stalled).toHaveBeenCalledTimes(1);
  });

  /**
   * The case the whole change exists for: work that takes far longer than the
   * window, but never pauses within it.
   */
  it("lets a slow run continue for as long as it keeps producing", () => {
    const stalled = vi.fn();
    const guard = createStallGuard(1000, stalled);
    guard.progress();

    // Twenty windows' worth of elapsed time, a token arriving in each.
    for (let tick = 0; tick < 20; tick += 1) {
      vi.advanceTimersByTime(900);
      guard.progress();
    }

    expect(stalled).not.toHaveBeenCalled();
    // Total elapsed is eighteen times the window, which is the point: this run
    // would have been killed by any ceiling shorter than the work.
    vi.advanceTimersByTime(1000);
    expect(stalled).toHaveBeenCalledTimes(1);
  });

  it("fires once, not once per rearm", () => {
    const stalled = vi.fn();
    const guard = createStallGuard(1000, stalled);
    guard.progress();
    guard.progress();
    guard.progress();

    vi.advanceTimersByTime(5000);
    expect(stalled).toHaveBeenCalledTimes(1);
  });

  it("stays quiet once stopped, so a finished run cannot be aborted late", () => {
    const stalled = vi.fn();
    const guard = createStallGuard(1000, stalled);
    guard.progress();
    guard.stop();

    vi.advanceTimersByTime(10_000);
    expect(stalled).not.toHaveBeenCalled();
  });

  it("does nothing until the first sign of life is reported", () => {
    const stalled = vi.fn();
    createStallGuard(1000, stalled);

    vi.advanceTimersByTime(10_000);
    expect(stalled).not.toHaveBeenCalled();
  });
});
