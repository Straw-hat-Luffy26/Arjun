/**
 * What these pin: a graph that draws its nodes on top of one another, and a
 * graph that rearranges itself every time it is looked at.
 *
 * The first is why this module exists. The layout it replaces treated every
 * node as a point, so a notebook of eleven files painted eleven names in a
 * heap and nothing in the picture could be read. The second is the rule the old
 * layout held and this one has to keep: a person must be able to point at a
 * cluster, come back, and find it where they left it.
 */
import { describe, expect, it } from 'vitest';
import { ALPHA_MIN, GraphSimulation, seedFrom, type SimLink } from './simulation';

const BOX = { width: 400, height: 300 };

function nodes(count: number, radius = 8) {
  return Array.from({ length: count }, (_, i) => ({ id: `n${i}`, radius }));
}

/** Runs to rest, or gives up — a simulation that never settles is a bug. */
function settle(simulation: GraphSimulation, limit = 2000): number {
  let ticks = 0;
  while (simulation.step() && ticks < limit) ticks += 1;
  return ticks;
}

describe('GraphSimulation', () => {
  it('comes to rest instead of running forever', () => {
    const simulation = new GraphSimulation(nodes(12), [], BOX);
    const ticks = settle(simulation);
    expect(ticks).toBeLessThan(2000);
    expect(simulation.alpha).toBeLessThan(ALPHA_MIN);
  });

  it('settles to the same arrangement every time', () => {
    const links: SimLink[] = [
      { source: 'n0', target: 'n1' },
      { source: 'n1', target: 'n2' },
    ];
    const first = new GraphSimulation(nodes(6), links, BOX);
    const second = new GraphSimulation(nodes(6), links, BOX);
    settle(first);
    settle(second);
    expect(first.nodes.map((n) => [n.id, n.x, n.y])).toEqual(
      second.nodes.map((n) => [n.id, n.x, n.y]),
    );
  });

  it('does not depend on the order the nodes arrived in', () => {
    const links: SimLink[] = [{ source: 'n0', target: 'n3' }];
    const forward = new GraphSimulation(nodes(6), links, BOX);
    const reversed = new GraphSimulation([...nodes(6)].reverse(), links, BOX);
    settle(forward);
    settle(reversed);
    const sorted = (s: GraphSimulation) =>
      [...s.nodes].sort((a, b) => a.id.localeCompare(b.id)).map((n) => [n.id, n.x, n.y]);
    expect(sorted(reversed)).toEqual(sorted(forward));
  });

  it('keeps nodes at least their radii apart', () => {
    // The whole reason for the collide force. Eight nodes of radius 30 in a
    // 400x300 box have room, and none of them may overlap.
    const simulation = new GraphSimulation(nodes(8, 30), [], BOX);
    settle(simulation);
    for (let i = 0; i < simulation.nodes.length; i += 1) {
      for (let j = i + 1; j < simulation.nodes.length; j += 1) {
        const a = simulation.nodes[i];
        const b = simulation.nodes[j];
        // A pixel of tolerance: the constraint is resolved by relaxation rather
        // than solved exactly, and a pixel is not a legibility problem.
        expect(Math.hypot(a.x - b.x, a.y - b.y)).toBeGreaterThan(a.radius + b.radius - 1);
      }
    }
  });

  it('keeps every node inside the box, whole', () => {
    const simulation = new GraphSimulation(nodes(10, 20), [], BOX);
    settle(simulation);
    for (const node of simulation.nodes) {
      expect(node.x).toBeGreaterThanOrEqual(node.radius - 0.001);
      expect(node.x).toBeLessThanOrEqual(BOX.width - node.radius + 0.001);
      expect(node.y).toBeGreaterThanOrEqual(node.radius - 0.001);
      expect(node.y).toBeLessThanOrEqual(BOX.height - node.radius + 0.001);
    }
  });

  it('holds a pinned node exactly where it was put', () => {
    const simulation = new GraphSimulation(nodes(6), [{ source: 'n0', target: 'n1' }], BOX);
    const held = simulation.find('n0');
    expect(held).toBeDefined();
    held!.fx = 123;
    held!.fy = 45;
    simulation.reheat();
    for (let i = 0; i < 100; i += 1) simulation.step();
    expect(held!.x).toBe(123);
    expect(held!.y).toBe(45);
  });

  it('reheats on demand and cools again, so a drag keeps it alive', () => {
    const simulation = new GraphSimulation(nodes(4), [], BOX);
    settle(simulation);
    expect(simulation.running).toBe(false);

    simulation.reheat();
    expect(simulation.running).toBe(true);
    for (let i = 0; i < 500; i += 1) simulation.step();
    // Held warm by the target alone: a drag must not time out under the hand.
    expect(simulation.running).toBe(true);

    simulation.cool();
    expect(settle(simulation)).toBeLessThan(2000);
  });

  it('produces no NaN, whatever it is given', () => {
    // Every node on the same spot, a self-link, and a link to a node that is
    // not in the graph.
    const simulation = new GraphSimulation(
      [
        { id: 'a', radius: 5 },
        { id: 'b', radius: 5 },
        { id: 'c', radius: 5 },
      ],
      [
        { source: 'a', target: 'a' },
        { source: 'a', target: 'missing' },
        { source: 'a', target: 'b', weight: 40 },
      ],
      { width: 1, height: 1 },
    );
    settle(simulation);
    for (const node of simulation.nodes) {
      expect(Number.isFinite(node.x)).toBe(true);
      expect(Number.isFinite(node.y)).toBe(true);
    }
  });

  it('places a lone node without dividing by zero', () => {
    const simulation = new GraphSimulation([{ id: 'only', radius: 4 }], [], BOX);
    settle(simulation);
    const only = simulation.find('only');
    expect(Number.isFinite(only!.x)).toBe(true);
    expect(Number.isFinite(only!.y)).toBe(true);
  });

  it('has nothing to do with an empty graph', () => {
    const simulation = new GraphSimulation([], [], BOX);
    expect(simulation.nodes).toHaveLength(0);
    expect(settle(simulation)).toBeLessThan(2000);
  });
});

describe('seedFrom', () => {
  it('is stable for the same ids and different for different ones', () => {
    expect(seedFrom(['a', 'b'])).toBe(seedFrom(['a', 'b']));
    expect(seedFrom(['a', 'b'])).not.toBe(seedFrom(['a', 'c']));
  });
});
