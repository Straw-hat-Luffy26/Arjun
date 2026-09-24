import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import type { UnlistenFn } from '@tauri-apps/api/event';
import { Brain, RotateCw, Search } from 'lucide-react';
import {
  memoryGraphService,
  type MemoryItem,
  type MemoryKind,
  type MemoryScope,
} from '../../services/memoryGraph.service';
import { applyBatch, emptyFeed, fromSnapshot, needsFetch, type FeedState } from './memoryFeed';
import {
  AGGREGATE_ABOVE,
  aggregateByAgent,
  neighbourhood,
  projectGraph,
  type MemoryGraphNode,
} from './memoryModel';
import { assignAgentColours, CANONICAL_COLOUR } from './agentColor';
import { MemoryGraphCanvas, type LayoutReport } from './MemoryGraphCanvas';
import styles from './AgentMemoryPanel.module.css';

/**
 * What a task's agents know, and how it is changing.
 *
 * ## The subscription, and the window it closes
 *
 * A snapshot first, then changes from *its* cursor. Never "snapshot, then
 * changes from now": a write landing between those two calls would be in
 * neither, and nothing afterwards could tell it had happened. The cursor comes
 * back from the same transaction that read the contents, so subscribing from it
 * misses nothing.
 *
 * The doorbell (`memory-graph:moved`) carries a revision and no payload. This
 * panel answers it by asking, as the signed-in person, what it may now see. A
 * second window signed in as somebody else gets its own answer to its own
 * question — which is the whole reason the event does not carry the change.
 *
 * ## What the header refuses to do
 *
 * It never shows a filtered count as though it were the total. "12 of 340" is
 * two numbers because a person deciding whether to trust an answer needs to
 * know both what is on screen and what exists. The six states overlap — a
 * proposal can also be in context and also in conflict — so they are listed as
 * separate counts and never as parts of a whole.
 */

export interface AgentMemoryPanelProps {
  /**
   * The task whose memory this is.
   *
   * A run id: the agent runtime scopes a task's memory by the run it belongs
   * to, so this is the same identifier `RunView` already holds.
   */
  runId: string;
  /** Narrows a workspace-scoped read. Null for a task's own memory. */
  projectId?: string | null;
}

/** Every memory kind, in the order the filter offers them. */
const KINDS: MemoryKind[] = [
  'goal',
  'fact',
  'constraint',
  'correction',
  'decision',
  'plan',
  'openQuestion',
  'toolObservation',
  'sourceRef',
  'artifactRef',
  'preference',
  'procedure',
];

/** How far a local view reaches. Three is where a neighbourhood stops being local. */
const MAX_DEPTH = 3;

/** One provenance, in a sentence an operator can act on. */
function describeProvenance(item: MemoryItem): string {
  switch (item.provenance.kind) {
    case 'operator':
      return `${item.provenance.user_id} recorded this directly.`;
    case 'toolReceipt':
      return `${item.provenance.tool} returned this in run ${item.provenance.run_id}, recorded at event ${item.provenance.event_seq}.`;
    case 'model':
      return `${item.provenance.model_id} asserted this during run ${item.provenance.run_id}. Nothing outside the model has corroborated it.`;
    case 'migrated':
      return `Brought across from ${item.provenance.legacy_store} entry ${item.provenance.legacy_id}.`;
  }
}

/** What a status means, spelled out rather than left to the outline. */
function describeStatus(item: MemoryItem): string {
  switch (item.status) {
    case 'admitted':
      return 'Established — something outside the model corroborated it.';
    case 'proposed':
      return 'Proposed — offered by a model and not yet corroborated. Not evidence.';
    case 'superseded':
      return 'Superseded — a later record replaced it. Kept, so the correction has a history.';
    case 'rejected':
      return 'Rejected — a person looked and said no. Kept, so it is not proposed again.';
    case 'tombstoned':
      return 'Deleted.';
  }
}

export const AgentMemoryPanel: React.FC<AgentMemoryPanelProps> = ({ runId, projectId = null }) => {
  const scope = useMemo<MemoryScope>(() => ({ kind: 'task', taskId: runId }), [runId]);

  const [feed, setFeed] = useState<FeedState>(emptyFeed);
  // The event listener needs the current cache without re-subscribing every
  // time it changes.
  const feedRef = useRef(feed);
  feedRef.current = feed;

  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [layout, setLayout] = useState<LayoutReport | null>(null);

  const [search, setSearch] = useState('');
  const [agents, setAgents] = useState<ReadonlySet<string>>(new Set());
  const [kinds, setKinds] = useState<ReadonlySet<MemoryKind>>(new Set());
  const [showAuthorship, setShowAuthorship] = useState(false);
  const [showCanonical, setShowCanonical] = useState(true);
  const [focus, setFocus] = useState<MemoryGraphNode | null>(null);
  /**
   * Agents whose work is shown item by item rather than as one bundle.
   *
   * Empty by default, and above {@link AGGREGATE_ABOVE} items that means every
   * agent is a bundle. Opening one is a click on it.
   */
  const [expanded, setExpanded] = useState<ReadonlySet<string>>(new Set());
  const [depth, setDepth] = useState(2);
  const [selected, setSelected] = useState<MemoryGraphNode | null>(null);

  /** Guards against two drains overlapping and applying the same batch twice. */
  const draining = useRef(false);

  const reload = useCallback(async () => {
    try {
      const snapshot = await memoryGraphService.snapshot(scope, { projectId, runId });
      const next = fromSnapshot(snapshot);
      feedRef.current = next;
      setFeed(next);
      setError(null);
    } catch (problem) {
      setError(String(problem));
    } finally {
      setLoading(false);
    }
  }, [scope, projectId, runId]);

  /**
   * Pulls changes until caught up.
   *
   * The loop is the backpressure: the backend never pushes rows, so a reader
   * that is behind simply asks again. `hasMore` is what stops a long catch-up
   * needing one round trip per write.
   *
   * A `reset` means the cursor is below what the log still covers — a gap that
   * replay cannot close — so the only correct answer is a fresh snapshot.
   */
  const drain = useCallback(async () => {
    if (draining.current) return;
    draining.current = true;
    try {
      let state = feedRef.current;
      // Bounded so a backend that always answered `hasMore` cannot spin here.
      for (let round = 0; round < 64; round += 1) {
        const batch = await memoryGraphService.changes(scope, state.cursor, { projectId });
        if (batch.reset) {
          await reload();
          return;
        }
        state = applyBatch(state, batch);
        if (!batch.hasMore) break;
      }
      feedRef.current = state;
      setFeed(state);
      setError(null);
    } catch (problem) {
      setError(String(problem));
    } finally {
      draining.current = false;
    }
  }, [scope, projectId, reload]);

  useEffect(() => {
    setLoading(true);
    void reload();
  }, [reload]);

  useEffect(() => {
    let unlisten: UnlistenFn | undefined;
    let cancelled = false;
    void memoryGraphService
      .onMoved(({ revision }) => {
        // Only when the graph is actually ahead of this cache. A revision from
        // another scope, or one this reader cannot see, is a round trip for
        // nothing.
        if (needsFetch(feedRef.current, revision)) void drain();
      })
      .then((stop) => {
        if (cancelled) stop();
        else unlisten = stop;
      })
      .catch(() => {
        // No event channel. The panel still works from its snapshot and the
        // refresh button; it simply will not update by itself.
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, [drain]);

  const colours = useMemo(
    () => assignAgentColours([...feed.items.values()].map((item) => item.agentId)),
    [feed.items],
  );

  const full = useMemo(
    () =>
      projectGraph(feed, {
        agents,
        kinds,
        search,
        includeAuthorship: showAuthorship,
        includeCanonical: showCanonical,
      }),
    [feed, agents, kinds, search, showAuthorship, showCanonical],
  );

  /**
   * What is actually drawn.
   *
   * Three narrowings, in the order that keeps each honest. The filter decides
   * what exists; focusing decides which part of it is being looked at; and
   * aggregation is a last resort applied only when what is left is still too
   * much to read.
   *
   * Aggregation is skipped entirely once a focus is set, because a local
   * neighbourhood is already small, and bundling inside it would hide the very
   * relations the reader focused in order to see.
   */
  const view = useMemo(() => {
    const local = focus ? neighbourhood(full, focus.id, depth) : full;
    const items = local.nodes.reduce(
      (total, node) => total + (node.kind === 'memory' ? 1 : 0),
      0,
    );
    if (focus || items <= AGGREGATE_ABOVE) return local;
    return aggregateByAgent(local, expanded);
  }, [full, focus, depth, expanded]);

  /** True when the picture is showing bundles rather than every item. */
  const aggregated = useMemo(
    () => view.nodes.some((node) => node.kind === 'cluster'),
    [view.nodes],
  );

  const everyAgent = useMemo(
    () => [...new Set([...feed.items.values()].map((item) => item.agentId))].sort(),
    [feed.items],
  );

  function toggled<T>(set: ReadonlySet<T>, value: T): Set<T> {
    const next = new Set(set);
    if (next.has(value)) next.delete(value);
    else next.add(value);
    return next;
  }

  const counts = view.counts;
  const totals = view.totals;
  const inContext = view.inContext.get(selected?.id ?? '');

  return (
    <section className={styles.panel}>
      <header className={styles.header}>
        <h2 className={styles.title}>
          <Brain size={15} />
          What this task knows
        </h2>
        <div className={styles.headerRight}>
          {feed.stale && (
            <span className={styles.stale} role="status">
              behind — refreshing
            </span>
          )}
          <button
            type="button"
            className={styles.refresh}
            onClick={() => void reload()}
            title="Re-read the graph from scratch"
          >
            <RotateCw size={13} />
          </button>
        </div>
      </header>

      {/* Two numbers, never one. A filtered count shown alone reads as "this
       * task only ever knew twelve things". */}
      <p className={styles.counts}>
        <strong>{counts.available}</strong> of {totals.available} shown
        <span className={styles.dot}>·</span>
        {counts.inContext} in the model&rsquo;s context
        <span className={styles.dot}>·</span>
        {counts.admitted} established
        <span className={styles.dot}>·</span>
        {counts.proposed} proposed
        <span className={styles.dot}>·</span>
        {counts.conflicted} in conflict
        <span className={styles.dot}>·</span>
        {counts.superseded} superseded
        {counts.rejected > 0 && (
          <>
            <span className={styles.dot}>·</span>
            {counts.rejected} rejected
          </>
        )}
      </p>

      <div className={styles.controls}>
        <label className={styles.searchBox}>
          <Search size={13} />
          <input
            type="search"
            value={search}
            placeholder="Search what the agents know"
            onChange={(event) => setSearch(event.target.value)}
            className={styles.searchInput}
          />
        </label>

        {everyAgent.length > 0 && (
          <div className={styles.chips}>
            {everyAgent.map((agentId) => (
              <button
                key={agentId}
                type="button"
                onClick={() => setAgents(toggled(agents, agentId))}
                className={agents.has(agentId) ? styles.chipOn : styles.chip}
                style={{ borderColor: colours.get(agentId) ?? CANONICAL_COLOUR }}
              >
                <span
                  className={styles.swatch}
                  style={{ background: colours.get(agentId) ?? CANONICAL_COLOUR }}
                />
                {agentId}
              </button>
            ))}
          </div>
        )}

        <div className={styles.chips}>
          {KINDS.map((kind) => (
            <button
              key={kind}
              type="button"
              onClick={() => setKinds(toggled(kinds, kind))}
              className={kinds.has(kind) ? styles.chipOn : styles.chip}
            >
              {kind}
            </button>
          ))}
        </div>

        <div className={styles.toggles}>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showAuthorship}
              onChange={(event) => setShowAuthorship(event.target.checked)}
            />
            authorship links
          </label>
          <label className={styles.toggle}>
            <input
              type="checkbox"
              checked={showCanonical}
              onChange={(event) => setShowCanonical(event.target.checked)}
            />
            sources and artifacts
          </label>
          {focus && (
            <span className={styles.toggle}>
              depth
              <input
                type="range"
                min={1}
                max={MAX_DEPTH}
                value={depth}
                onChange={(event) => setDepth(Number(event.target.value))}
              />
              {depth}
              <button type="button" className={styles.clear} onClick={() => setFocus(null)}>
                show all
              </button>
            </span>
          )}
        </div>
      </div>

      {error && (
        <p className={styles.error} role="alert">
          {error}
        </p>
      )}

      <div className={styles.canvasWrap}>
        {loading ? (
          <p className={styles.quiet}>Reading what this task knows…</p>
        ) : view.nodes.length === 0 ? (
          <p className={styles.quiet}>
            {totals.available === 0
              ? 'This task has not established anything yet.'
              : 'Nothing matches the current filter.'}
          </p>
        ) : (
          <MemoryGraphCanvas
            view={view}
            colours={colours}
            focusId={focus?.id ?? null}
            selectedId={selected?.id ?? null}
            onSelect={(node) => {
              // A bundle is not a thing to inspect, it is a thing to open —
              // and an opened agent is a thing to close. Both are the same
              // gesture on the same shape, which is what makes the drill-down
              // discoverable without a legend entry explaining it.
              if ((node.kind === 'cluster' || node.kind === 'agent') && node.agentId) {
                setExpanded(toggled(expanded, node.agentId));
                return;
              }
              setSelected(node);
            }}
            onFocus={(node) => {
              if ((node.kind === 'cluster' || node.kind === 'agent') && node.agentId) {
                setExpanded(toggled(expanded, node.agentId));
                return;
              }
              setFocus(node);
              setSelected(node);
            }}
            onLayout={setLayout}
          />
        )}
      </div>

      {/* The legend. Shape, outline and colour are the only things telling the
       * kinds apart on the canvas, so the key to them has to be beside it. */}
      <div className={styles.legend}>
        <span className={styles.legendItem}>● memory</span>
        <span className={styles.legendItem}>⬡ agent</span>
        <span className={styles.legendItem}>■ task</span>
        <span className={styles.legendItem}>▭ source</span>
        <span className={styles.legendItem}>◆ artifact</span>
        <span className={styles.legendSep} />
        <span className={styles.legendItem}>filled = established</span>
        <span className={styles.legendItem}>dashed = proposed</span>
        <span className={styles.legendItem}>dotted + dim = superseded</span>
        <span className={styles.legendItem}>slashed = rejected</span>
        <span className={styles.legendItem}>double ring = in conflict</span>
        <span className={styles.legendItem}>white ring = in the model&rsquo;s context</span>
        <span className={styles.legendSep} />
        <span className={styles.legendItem}>colour = the agent that wrote it</span>
        <span className={styles.legendItem} style={{ color: CANONICAL_COLOUR }}>
          slate = shared, belongs to no agent
        </span>
      </div>

      {/* What the layout could and could not do. A picture with edges drawn
       * through nodes is a fact about the picture, and belongs on screen
       * rather than in a console. */}
      {layout && (layout.unroutedEdges > 0 || layout.overlaps > 0) && (
        <p className={styles.note}>
          {layout.overlaps > 0 &&
            `${layout.overlaps} pair(s) of labels still overlap at this density. `}
          {layout.unroutedEdges > 0 &&
            `${layout.unroutedEdges} of ${layout.edges} edge(s) could not be routed clear of other nodes and are drawn straight. `}
          Focus a node, or narrow the filters, for a legible view.
        </p>
      )}

      {selected?.item && (
        <aside className={styles.inspector}>
          <h3 className={styles.inspectorTitle}>{selected.item.kind}</h3>
          <p className={styles.content}>{selected.item.content}</p>
          <dl className={styles.facts}>
            <dt>Status</dt>
            <dd>{describeStatus(selected.item)}</dd>
            <dt>Where it came from</dt>
            <dd>{describeProvenance(selected.item)}</dd>
            <dt>Agent</dt>
            <dd>
              <span
                className={styles.swatch}
                style={{ background: colours.get(selected.item.agentId) ?? CANONICAL_COLOUR }}
              />
              {selected.item.agentId}
            </dd>
            <dt>Revision</dt>
            <dd>
              {selected.item.revision} · updated {selected.item.updatedAt}
            </dd>
            {inContext && (
              <>
                <dt>In the model&rsquo;s context</dt>
                <dd>
                  {inContext.current
                    ? `Yes, at this revision (chosen as ${inContext.reason}).`
                    : `The turn carried revision ${inContext.revision}; this has been corrected since, so the model read an older version.`}
                </dd>
              </>
            )}
            {selected.item.sources.length > 0 && (
              <>
                <dt>Cites</dt>
                <dd>
                  {selected.item.sources.map((source) => (
                    <div key={`${source.sha256}:${source.locator}`}>
                      {source.sha256.slice(0, 12)}… · {source.locator}
                    </div>
                  ))}
                </dd>
              </>
            )}
            {selected.item.artifacts.length > 0 && (
              <>
                <dt>Produced</dt>
                <dd>
                  {selected.item.artifacts.map((artifact) => (
                    <div key={`${artifact.artifactId}@${artifact.revision}`}>
                      {artifact.artifactId} revision {artifact.revision}
                    </div>
                  ))}
                </dd>
              </>
            )}
            {selected.item.conflictsWith.length > 0 && (
              <>
                <dt>Disagrees with</dt>
                <dd>{selected.item.conflictsWith.join(', ')}</dd>
              </>
            )}
          </dl>
        </aside>
      )}
    </section>
  );
};

export default AgentMemoryPanel;
