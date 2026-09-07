/**
 * The graph's physics.
 *
 * A running force simulation, not a one-shot layout. The difference is the
 * whole feel of the thing: a graph solved once and painted is a picture, and a
 * graph that settles, reacts to a dragged node and comes to rest again is an
 * instrument. Obsidian's graph view is the reference, and this is the same
 * model.
 *
 * ## What it is a copy of
 *
 * `d3-force`, whose published model this reimplements with its own default
 * parameters. Velocity Verlet integration with unit time step and unit mass;
 * each tick advances `alpha` toward `alphaTarget`, lets every force write into
 * velocities, damps them, and then moves the positions:
 *
 *     alpha += (alphaTarget - alpha) * alphaDecay
 *     ...forces adjust vx, vy...
 *     vx *= 1 - velocityDecay ;  x += vx
 *
 * An `alphaDecay` of `1 - 0.001^(1/300)` is what makes a graph settle in about
 * three hundred ticks and then stop, so an idle notebook is not burning a core.
 *
 * ## Why it is written here rather than installed
 *
 * The only hard part of `d3-force` is the Barnes–Hut quadtree that takes the
 * n-body force from O(n²) to O(n log n) — and that matters at tens of thousands
 * of nodes, not at the tens or hundreds a notebook produces. Against that, this
 * repository builds and verifies offline: a new dependency has to be in the
 * build machine's npm cache and forces an SBOM regeneration. Paying that for a
 * loop over eighteen nodes is the wrong trade. If a corpus ever arrives that
 * needs the quadtree, swapping this module for the real `d3-force` is a
 * contained change, because nothing outside it knows how a tick is computed.
 *
 * ## Determinism, still
 *
 * The rule from the layout this replaces holds: the same graph must produce the
 * same picture. Starting positions are seeded from the node ids, every force is
 * a pure function of the current state, and nothing consults a clock or
 * `Math.random`. A graph left to settle twice settles identically, so a person
 * can point at a cluster and find it again.
 */
/**
 * A seed derived from the graph itself.
 *
 * So the same graph always starts from the same arrangement, and two different
 * graphs are very unlikely to share a starting one. FNV-1a over the ids.
 */
export function seedFrom(ids: readonly string[]): number {
  let hash = 2166136261;
  for (const id of ids) {
    for (let i = 0; i < id.length; i += 1) {
      hash ^= id.charCodeAt(i);
      hash = Math.imul(hash, 16777619);
    }
  }
  return hash >>> 0;
}

/** A node in the simulation. Positions and velocities are in layout units. */
export interface SimNode {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  /**
   * How much room the node takes, as a radius.
   *
   * The collide force keeps any two nodes at least the sum of their radii
   * apart. This is the field that stops eleven file names being painted on top
   * of each other: a file is drawn as a box two hundred pixels wide, and a
   * simulation that thinks it is a point will happily stack them.
   */
  radius: number;
  /** Pinned position while a pointer is holding the node. `null` when free. */
  fx: number | null;
  fy: number | null;
}

export interface SimLink {
  source: string;
  target: string;
  /** Heavier links pull harder, dampened by a log. */
  weight?: number;
}

export interface SimulationOptions {
  width: number;
  height: number;
  /** Rest length of a link, on top of the room its ends need. */
  linkDistance?: number;
  /** Many-body strength. Negative repels; d3-force's default is -30. */
  chargeStrength?: number;
  /** How hard the whole graph is held to the middle of the box. */
  centerStrength?: number;
  /** Fraction of velocity lost each tick. d3-force's default is 0.4. */
  velocityDecay?: number;
  /** Relaxation passes for the collide force. More is tidier and slower. */
  collideIterations?: number;
}

/** Alpha at which the simulation is considered at rest, as in d3-force. */
export const ALPHA_MIN = 0.001;
/** Reaches `ALPHA_MIN` from 1 in about three hundred ticks. */
export const ALPHA_DECAY = 1 - Math.pow(ALPHA_MIN, 1 / 300);
/** Alpha a drag holds the simulation at, so it stays lively under the hand. */
export const ALPHA_DRAG = 0.3;

/** A small, fast, seeded generator — mulberry32, as in the layout it replaces. */
function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class GraphSimulation {
  readonly nodes: SimNode[];
  alpha = 1;
  alphaTarget = 0;

  private readonly index = new Map<string, number>();
  private readonly links: Array<{ a: number; b: number; strength: number; bias: number }>;
  private readonly width: number;
  private readonly height: number;
  private readonly linkDistance: number;
  private readonly chargeStrength: number;
  private readonly centerStrength: number;
  private readonly velocityDecay: number;
  private readonly collideIterations: number;
  private readonly random: () => number;

  constructor(
    nodes: ReadonlyArray<{ id: string; radius: number }>,
    links: ReadonlyArray<SimLink>,
    options: SimulationOptions,
  ) {
    this.width = options.width;
    this.height = options.height;
    this.linkDistance = options.linkDistance ?? 30;
    this.chargeStrength = options.chargeStrength ?? -30;
    this.centerStrength = options.centerStrength ?? 0.05;
    this.velocityDecay = options.velocityDecay ?? 0.4;
    this.collideIterations = options.collideIterations ?? 2;

    // Sorted ids, so the arrangement does not depend on the order rows arrived
    // in — the same rule the previous layout held to.
    const ordered = [...nodes].sort((a, b) => a.id.localeCompare(b.id));
    this.random = seededRandom(seedFrom(ordered.map((node) => node.id)));

    // A ring to start on. A ring has no clumps to escape, so the graph opens
    // out instead of exploding, and it gives the settling motion that reads as
    // the graph arranging itself rather than appearing pre-arranged.
    const radius = Math.min(this.width, this.height) * 0.35;
    this.nodes = ordered.map((node, i) => {
      const angle = ordered.length === 1 ? 0 : (2 * Math.PI * i) / ordered.length;
      return {
        id: node.id,
        x: this.width / 2 + radius * Math.cos(angle) + (this.random() - 0.5) * 8,
        y: this.height / 2 + radius * Math.sin(angle) + (this.random() - 0.5) * 8,
        vx: 0,
        vy: 0,
        radius: Math.max(1, node.radius),
        fx: null,
        fy: null,
      };
    });
    this.nodes.forEach((node, i) => this.index.set(node.id, i));

    // Link strength falls off with the degree of the busier end, exactly as
    // d3-force does: without it a hub yanks its whole neighbourhood around and
    // the simulation never settles.
    const degree = new Map<string, number>();
    for (const link of links) {
      degree.set(link.source, (degree.get(link.source) ?? 0) + 1);
      degree.set(link.target, (degree.get(link.target) ?? 0) + 1);
    }
    this.links = links
      .map((link) => {
        const a = this.index.get(link.source);
        const b = this.index.get(link.target);
        if (a === undefined || b === undefined || a === b) return null;
        const da = degree.get(link.source) ?? 1;
        const db = degree.get(link.target) ?? 1;
        return {
          a,
          b,
          // Weight is dampened with a log: a pair sharing forty passages should
          // sit closer than one sharing two, but not forty times closer.
          strength: (1 / Math.min(da, db)) * (1 + Math.log(Math.max(1, link.weight ?? 1))),
          // Which end gives way. The busier end moves less.
          bias: da / (da + db),
        };
      })
      .filter(
        (link): link is { a: number; b: number; strength: number; bias: number } =>
          link !== null,
      );
  }

  /** Whether the simulation still has energy worth spending a frame on. */
  get running(): boolean {
    return this.alpha >= ALPHA_MIN || this.alphaTarget > 0;
  }

  /** Holds the simulation warm — while a node is being dragged, say. */
  reheat(target = ALPHA_DRAG): void {
    this.alphaTarget = target;
    if (this.alpha < target) this.alpha = target;
  }

  /** Lets it cool back to rest. */
  cool(): void {
    this.alphaTarget = 0;
  }

  find(id: string): SimNode | undefined {
    const at = this.index.get(id);
    return at === undefined ? undefined : this.nodes[at];
  }

  /**
   * Advances one tick. Returns whether the simulation is still running.
   *
   * The order is d3-force's: age the alpha, let every force write into
   * velocities, damp, integrate, then resolve overlaps.
   */
  step(): boolean {
    this.alpha += (this.alphaTarget - this.alpha) * ALPHA_DECAY;

    this.applyCharge();
    this.applyLinks();
    this.applyCentering();

    for (const node of this.nodes) {
      if (node.fx !== null) {
        node.x = node.fx;
        node.vx = 0;
      } else {
        node.vx *= 1 - this.velocityDecay;
        node.x += node.vx;
      }
      if (node.fy !== null) {
        node.y = node.fy;
        node.vy = 0;
      } else {
        node.vy *= 1 - this.velocityDecay;
        node.y += node.vy;
      }
    }

    this.applyCollision();
    this.clampToBox();
    return this.running;
  }

  /** Coulomb repulsion between every pair. O(n²), which is free at this size. */
  private applyCharge(): void {
    const { nodes } = this;
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        let dx = nodes[j].x - nodes[i].x;
        let dy = nodes[j].y - nodes[i].y;
        let distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < 0.01) {
          // Two nodes exactly on top of each other have no direction to
          // separate along. The nudge comes from the seeded generator, so even
          // the escape is reproducible.
          dx = (this.random() - 0.5) * 0.1;
          dy = (this.random() - 0.5) * 0.1;
          distanceSquared = dx * dx + dy * dy || 0.01;
        }
        // Inverse-square, as d3-force applies it: strength * alpha / d².
        const force = (this.chargeStrength * this.alpha) / distanceSquared;
        const distance = Math.sqrt(distanceSquared);
        const fx = (dx / distance) * force;
        const fy = (dy / distance) * force;
        nodes[i].vx += fx;
        nodes[i].vy += fy;
        nodes[j].vx -= fx;
        nodes[j].vy -= fy;
      }
    }
  }

  /** Hooke springs along the edges, pulling toward their rest length. */
  private applyLinks(): void {
    for (const link of this.links) {
      const a = this.nodes[link.a];
      const b = this.nodes[link.b];
      // A link's rest length grows with the room its ends need, or a wide file
      // box is dragged on top of the terms it is joined to.
      const rest = this.linkDistance + a.radius + b.radius;
      const dx = b.x + b.vx - (a.x + a.vx);
      const dy = b.y + b.vy - (a.y + a.vy);
      const distance = Math.hypot(dx, dy) || 0.01;
      const pull = ((distance - rest) / distance) * this.alpha * link.strength;
      const fx = dx * pull;
      const fy = dy * pull;
      b.vx -= fx * link.bias;
      b.vy -= fy * link.bias;
      a.vx += fx * (1 - link.bias);
      a.vy += fy * (1 - link.bias);
    }
  }

  /** A gentle pull to the middle, so a sparse graph does not drift to a corner. */
  private applyCentering(): void {
    const cx = this.width / 2;
    const cy = this.height / 2;
    const strength = this.centerStrength * this.alpha;
    for (const node of this.nodes) {
      node.vx += (cx - node.x) * strength;
      node.vy += (cy - node.y) * strength;
    }
  }

  /**
   * Keeps nodes from overlapping, by their radii.
   *
   * The force the previous layout did not have, and the reason a notebook of
   * eleven files drew eleven names in a heap. Positional relaxation rather than
   * a velocity nudge: an overlap is a constraint to satisfy now, not a
   * preference to express, and a name half-under another name is unreadable
   * however briefly.
   */
  private applyCollision(): void {
    const { nodes } = this;
    for (let pass = 0; pass < this.collideIterations; pass += 1) {
      for (let i = 0; i < nodes.length; i += 1) {
        for (let j = i + 1; j < nodes.length; j += 1) {
          const a = nodes[i];
          const b = nodes[j];
          const wanted = a.radius + b.radius;
          let dx = b.x - a.x;
          let dy = b.y - a.y;
          let distance = Math.hypot(dx, dy);
          if (distance >= wanted) continue;
          if (distance < 0.01) {
            dx = (this.random() - 0.5) * 0.1;
            dy = (this.random() - 0.5) * 0.1;
            distance = Math.hypot(dx, dy) || 0.01;
          }
          const push = (wanted - distance) / distance / 2;
          const sx = dx * push;
          const sy = dy * push;
          // A pinned node does not give way; the other takes the whole
          // correction, so dragging a file pushes terms aside rather than
          // sliding out from under the pointer.
          const aFixed = a.fx !== null || a.fy !== null;
          const bFixed = b.fx !== null || b.fy !== null;
          if (!bFixed) {
            b.x += aFixed ? sx * 2 : sx;
            b.y += aFixed ? sy * 2 : sy;
          }
          if (!aFixed) {
            a.x -= bFixed ? sx * 2 : sx;
            a.y -= bFixed ? sy * 2 : sy;
          }
        }
      }
    }
  }

  /** Nothing leaves the box, and nothing is drawn half off its edge. */
  private clampToBox(): void {
    for (const node of this.nodes) {
      const low = Math.min(node.radius, this.width / 2);
      const highX = Math.max(low, this.width - node.radius);
      const highY = Math.max(Math.min(node.radius, this.height / 2), this.height - node.radius);
      node.x = Math.min(highX, Math.max(low, node.x));
      node.y = Math.min(highY, Math.max(Math.min(node.radius, this.height / 2), node.y));
    }
  }
}
