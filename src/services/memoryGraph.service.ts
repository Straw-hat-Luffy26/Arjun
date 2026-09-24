/**
 * The agent-memory graph: what the agents on a task know, and how it changes.
 *
 * Thin, like every other service here. What is *not* thin is the subscription
 * contract, which is the reason this file has prose in it at all.
 *
 * ## Why the event carries a number and this file still does the fetching
 *
 * `memory-graph:moved` reaches every window in the process and carries one
 * integer. It is a doorbell, not a delivery. Everything a person is allowed to
 * see comes back from `memory_graph_changes`, which runs as the signed-in
 * session — so a second window signed in as somebody else cannot be handed a
 * fact it is not cleared for, even by accident, because the fact never left the
 * backend without a session attached to the request.
 *
 * ## Why the cursor comes from the snapshot
 *
 * `snapshot()` returns a `cursor` read in the same transaction as its contents.
 * Subscribing from it is what makes "take a snapshot, then subscribe" safe: a
 * write that lands between the two calls is after the cursor, so the first
 * batch contains it. Taking the snapshot and then asking for "changes from now"
 * would lose exactly that write, silently, which is the bug this shape exists
 * to prevent.
 */
import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { getBackendService } from './api';

/** Mirrors `knowledge::graph::runtime_memory::MemoryKind`. */
export type MemoryKind =
  | 'goal'
  | 'fact'
  | 'constraint'
  | 'correction'
  | 'decision'
  | 'plan'
  | 'openQuestion'
  | 'toolObservation'
  | 'sourceRef'
  | 'artifactRef'
  | 'preference'
  | 'procedure';

/** Mirrors `knowledge::graph::runtime_memory::EdgeKind`. */
export type MemoryEdgeKind =
  | 'supports'
  | 'contradicts'
  | 'supersedes'
  | 'derivedFrom'
  | 'cites'
  | 'partOf'
  | 'answers';

/**
 * Where an item stands. Mirrors `ItemStatus`.
 *
 * `proposed` and `admitted` are the distinction the whole view is built around:
 * a model saying something is a proposal until something outside the model
 * corroborates it, and the canvas must never draw the two the same way.
 */
export type ItemStatus = 'proposed' | 'admitted' | 'superseded' | 'rejected' | 'tombstoned';

/**
 * Who or what put an item in the graph.
 *
 * The field names inside each variant are snake_case, unlike every other type
 * on this wire. That is not a mistake to be fixed here: serde's `rename_all` on
 * an enum renames the variants, not the fields inside them, and `Provenance`
 * predates this view. Matching the wire is the job of this file.
 */
export type Provenance =
  | { kind: 'toolReceipt'; run_id: string; tool: string; event_seq: number }
  | { kind: 'model'; model_id: string; run_id: string }
  | { kind: 'operator'; user_id: string }
  | { kind: 'migrated'; legacy_store: string; legacy_id: string };

/** Which slice of the world an item belongs to. Mirrors `MemoryScope`. */
export type MemoryScope =
  | { kind: 'task'; taskId: string }
  | { kind: 'workspace'; projectId: string }
  | { kind: 'user'; userId: string };

/** Source bytes at the version they were read at. */
export interface SourceRef {
  sha256: string;
  locator: string;
  extractionRevision?: string;
}

/** A produced artifact at an exact revision. */
export interface ArtifactRef {
  artifactId: string;
  revision: number;
  sha256: string;
}

/** One thing an agent knows. Mirrors `MemoryItem`. */
export interface MemoryItem {
  itemId: string;
  revision: number;
  kind: MemoryKind;
  /** The agent this is attributed to. Never a model id — this is what colours it. */
  agentId: string;
  scope: MemoryScope;
  classification: string;
  acl: { clearedRoles: string[]; projectId: string | null; owner: string | null };
  creatorModelId?: string;
  creatorRunId?: string;
  provenance: Provenance;
  content: string;
  sources: SourceRef[];
  artifacts: ArtifactRef[];
  confidence?: number;
  status: ItemStatus;
  validFrom: string;
  validUntil?: string;
  supersedes?: string;
  conflictsWith: string[];
  causalParents: string[];
  idempotencyKey?: string;
  createdAt: string;
  updatedAt: string;
}

/** A typed link between two items. Mirrors `MemoryEdge`. */
export interface MemoryEdge {
  edgeId: string;
  fromItem: string;
  toItem: string;
  kind: MemoryEdgeKind;
  /** The agent that drew this link — a different question from who said each end. */
  agentId: string;
  scope: MemoryScope;
  createdAt: string;
}

/** One item the running turn carried. Mirrors `InContextItem`. */
export interface InContextItem {
  itemId: string;
  /** The revision the model actually read. */
  revision: number;
  reason: string;
  /** False when the item has been corrected since the turn was compiled. */
  current: boolean;
}

/** A consistent picture, and the cursor that continues it. */
export interface MemorySnapshot {
  items: MemoryItem[];
  edges: MemoryEdge[];
  cursor: number;
  inContext: InContextItem[];
  contextRevision?: number;
}

/**
 * What happened to one subject.
 *
 * A drop carries an id and nothing else — no label, no content, no kind. That
 * is the permission-revocation path, and the shape is the guarantee: there is
 * nowhere in this message for the withdrawn content to travel.
 */
export type FeedChange =
  | { change: 'itemChanged'; item: MemoryItem }
  | { change: 'itemDropped'; itemId: string }
  | { change: 'edgeChanged'; edge: MemoryEdge }
  | { change: 'edgeDropped'; edgeId: string };

/** One entry of the feed, at its position. */
export type FeedEntry = FeedChange & {
  /** Ordering is by this, never by `at` — two processes' clocks disagree. */
  revision: number;
  at: string;
};

/** What one read of the feed returned. */
export interface ChangeBatch {
  entries: FeedEntry[];
  cursor: number;
  /** Come straight back rather than waiting for the next doorbell. */
  hasMore: boolean;
  /** The cursor is below what the log still covers. Re-snapshot. */
  reset: boolean;
}

/** The doorbell's entire payload. */
export interface GraphMoved {
  revision: number;
}

/** The event name, matching `commands::memory_graph::MEMORY_GRAPH_EVENT`. */
export const MEMORY_GRAPH_EVENT = 'memory-graph:moved';

export const memoryGraphService = {
  /**
   * The graph as this person may see it, with the cursor that continues it.
   *
   * `runId` is optional: naming a run marks what that run's context actually
   * carried. Without one the graph is still the graph — worth looking at
   * between turns, not only during one.
   */
  snapshot(
    scope: MemoryScope,
    options?: { projectId?: string | null; runId?: string | null },
  ): Promise<MemorySnapshot> {
    return getBackendService().invoke<MemorySnapshot>('memory_graph_snapshot', {
      scope,
      projectId: options?.projectId ?? null,
      runId: options?.runId ?? null,
    });
  },

  /**
   * Everything after `cursor`, as this person may see it.
   *
   * `limit` is clamped by the backend. A caller asking for everything is asking
   * for a message big enough to stall the window it is drawn in.
   */
  changes(
    scope: MemoryScope,
    cursor: number,
    options?: { projectId?: string | null; limit?: number },
  ): Promise<ChangeBatch> {
    return getBackendService().invoke<ChangeBatch>('memory_graph_changes', {
      scope,
      cursor,
      projectId: options?.projectId ?? null,
      limit: options?.limit ?? null,
    });
  },

  /**
   * Rings when the graph moves. Carries a revision and nothing else.
   *
   * The caller fetches; this only says that fetching is worth doing.
   */
  onMoved(callback: (moved: GraphMoved) => void): Promise<UnlistenFn> {
    return listen<GraphMoved>(MEMORY_GRAPH_EVENT, ({ payload }) => callback(payload));
  },
};
