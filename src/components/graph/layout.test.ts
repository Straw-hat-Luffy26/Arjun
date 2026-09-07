/**
 * What this pins: a hub that swallows the canvas, and a leaf too small to hit.
 *
 * The arrangement tests that used to live here moved to `./simulation.test.ts`
 * along with the physics. What stayed is the size rule, and its two bounds are
 * the point of it: an unbounded ceiling means one heavily linked term covers
 * everything else, and an unbounded floor means a term with no links is a few
 * pixels of nothing that cannot be clicked.
 */
import { describe, expect, it } from 'vitest';
import { radiusFor } from './layout';

describe('radiusFor', () => {
  it('grows with degree but stays within bounds', () => {
    expect(radiusFor(0)).toBeLessThan(radiusFor(4));
    expect(radiusFor(4)).toBeLessThan(radiusFor(25));
    expect(radiusFor(10_000)).toBeLessThanOrEqual(18);
    expect(radiusFor(-5)).toBeGreaterThanOrEqual(4);
  });
});
