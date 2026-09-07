import React, { useCallback, useEffect, useState } from 'react';
import { Network, ArrowLeft, GitBranch } from 'lucide-react';
import { Button, Spinner } from '../ui';
import { GraphCanvas } from './GraphCanvas';
import {
  notebookGraphService,
  type RelationOutcome,
  type BuildOutcome,
  type EvidenceRow,
  type GraphEdge,
  type GraphNode,
  type GraphView,
  type TypingOutcome,
} from '../../services/notebook.service';
import { describeEdge } from './edgeLabel';
import { FlowCanvas } from './FlowCanvas';
import { buildFlow, describeFlow } from './flow';
import {
  describeCoverage,
  describeRun,
  describeVerification,
  yieldIsPoor,
} from './typingOutcome';
import styles from './NotebookGraphPanel.module.css';

/**
 * The graph half of the Notebooks screen.
 *
 * ## Local first
 *
 * The default view is one node and its neighbours, not the whole graph. A global
 * force layout is legible up to a couple of hundred nodes and becomes an
 * undifferentiated tangle after that — the failure mode every large Obsidian
 * vault demonstrates. The local view stays readable at any size, so centring on
 * a term is one double-click away and the header always says how much of the
 * graph is on screen.
 *
 * ## Counts are shown, not implied
 *
 * A filtered view that quietly showed forty of nine hundred nodes would read as
 * "my documents contained forty things", which is the kind of wrong number this
 * codebase has rules against.
 */

export interface NotebookGraphPanelProps {
  notebookId: string;
  documentCount: number;
  /**
   * Draw one file's graph rather than the whole notebook's.
   *
   * Null means the notebook. Passing a sha narrows the counts too, so the
   * header describes the file rather than the library it sits in.
   */
  documentSha256?: string | null;
  /** Hands a chosen subgraph to the caller, which knows how to reach a chat. */
  onImport?: (nodes: GraphNode[]) => void;
  importing?: boolean;
}

/** One sentence describing what a build did. Every number came from the backend. */
export function describeBuild(outcome: BuildOutcome): string {
  const parts: string[] = [];
  if (outcome.documentsBuilt > 0) parts.push(`${outcome.documentsBuilt} read`);
  if (outcome.documentsSkipped > 0) parts.push(`${outcome.documentsSkipped} already built`);
  if (outcome.documentsUnreadable > 0) parts.push(`${outcome.documentsUnreadable} unreadable`);
  const head = parts.length > 0 ? parts.join(' · ') : 'Nothing to build.';
  if (outcome.documentsBuilt === 0) return head;
  return `${head} — ${outcome.candidatesFound} candidate terms, ${outcome.nodesKept} kept, ${outcome.droppedRare} too rare, ${outcome.droppedGeneric} too common.`;
}

export const NotebookGraphPanel: React.FC<NotebookGraphPanelProps> = ({
  notebookId,
  documentCount,
  documentSha256 = null,
  onImport,
  importing,
}) => {
  const [view, setView] = useState<GraphView | null>(null);
  const [focus, setFocus] = useState<GraphNode | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const [evidence, setEvidence] = useState<EvidenceRow[] | null>(null);
  const [picked, setPicked] = useState<Set<string>>(new Set());
  const [outcome, setOutcome] = useState<BuildOutcome | null>(null);
  const [typing, setTyping] = useState<TypingOutcome | null>(null);
  const [busy, setBusy] = useState(false);
  const [typingBusy, setTypingBusy] = useState(false);
  const [relations, setRelations] = useState<RelationOutcome | null>(null);
  const [relationBusy, setRelationBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  // Files are in the graph by default. Which document said this is the relation
  // a person opens this screen for, and a cloud of terms with no route back to a
  // source is what this panel was reported as being.
  const [showFiles, setShowFiles] = useState(true);
  // The flow view is entered on request, not on the second click. The force
  // graph rearranging itself the moment a second file is picked would be the
  // picture changing shape without being asked for.
  const [tracing, setTracing] = useState(false);

  const load = useCallback(
    async (focusId: string | null, files: boolean) => {
      setError(null);
      try {
        setView(
          await notebookGraphService.graph(
            notebookId,
            documentSha256,
            focusId,
            focusId ? 2 : 1,
            1,
            files,
          ),
        );
      } catch (err) {
        setError(String(err));
      } finally {
        setLoading(false);
      }
    },
    [notebookId, documentSha256],
  );

  useEffect(() => {
    setFocus(null);
    setSelected(null);
    setPicked(new Set());
    setLoading(true);
    void load(null, showFiles);
    // Deliberately not re-run on `showFiles`: its own button reloads the view
    // in place, keeping the focus rather than resetting the whole panel.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [notebookId, load]);

  /**
   * Runs the statistical pass.
   *
   * The button is a rebuild once a graph exists, and it has to mean it. Asking
   * for the resumable build there skips every document already recorded at this
   * extractor version and reports "11 already built" — which is truthful about
   * the work and useless as an answer to "this graph looks wrong, build it
   * again". So a graph that already has nodes forces the rebuild, and only the
   * first build (or one over an empty graph) resumes.
   *
   * Forcing is cheap here: no model is involved, the pass reads passages that
   * were already extracted, and the outcome it returns is the only place the
   * candidate/kept/dropped counts are ever shown.
   */
  const build = useCallback(async () => {
    setBusy(true);
    setError(null);
    const force = (view?.totalNodes ?? 0) > 0;
    try {
      const result = await notebookGraphService.build(notebookId, force);
      setOutcome(result);
      await load(focus?.id ?? null, showFiles);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }, [notebookId, focus, load, view, showFiles]);

  /**
   * Runs the typing pass.
   *
   * Slow, and honest about it: minutes per document on a workstation model. The
   * result is reported in full, including how many of the model's claims the
   * verification gate rejected — a low yield is a finding about the model, not
   * something to hide behind the count that survived.
   */
  const runTyping = useCallback(async () => {
    setTypingBusy(true);
    setError(null);
    try {
      const result = await notebookGraphService.type(notebookId);
      setTyping(result);
      await load(focus?.id ?? null, showFiles);
    } catch (err) {
      // No fallback. If the model is unreachable the graph stays statistical,
      // and saying so is better than typing it by keyword and presenting the
      // guesses in the same shape as model output.
      setError(String(err));
    } finally {
      setTypingBusy(false);
    }
  }, [notebookId, focus, load, showFiles]);

  /**
   * Names the edges with Babelscape/rebel-large.
   *
   * Separate from `runTyping` rather than folded into it, because the two do
   * different work with different models and fail differently. Typing needs a
   * warm chat model and refuses when there is none; this starts a 400M CPU
   * sidecar of its own and needs nothing warm. Merging them behind one button
   * would mean one failure message for two unrelated causes.
   */
  const runRelations = useCallback(async () => {
    setRelationBusy(true);
    setError(null);
    try {
      const result = await notebookGraphService.extractRelations(notebookId);
      setRelations(result);
      await load(focus?.id ?? null, showFiles);
    } catch (err) {
      // No fallback. If the extractor is not installed the edges stay unnamed,
      // and saying so is better than labelling them by keyword and presenting
      // the guesses in the same shape as model output.
      setError(String(err));
    } finally {
      setRelationBusy(false);
    }
  }, [notebookId, focus, load, showFiles]);

  const inspect = useCallback(
    async (node: GraphNode) => {
      setSelected(node);
      setEvidence(null);
      try {
        setEvidence(await notebookGraphService.nodeEvidence(notebookId, node.id));
      } catch (err) {
        setError(String(err));
      }
    },
    [notebookId, documentSha256],
  );

  const focusOn = useCallback(
    (node: GraphNode) => {
      setFocus(node);
      void load(node.id, showFiles);
    },
    [load, showFiles],
  );

  const togglePicked = useCallback((id: string) => {
    setPicked((current) => {
      const next = new Set(current);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  if (loading) {
    return (
      <div className={styles.centered}>
        <Spinner />
      </div>
    );
  }

  // Emptiness is about terms, not nodes. A notebook whose files all produced
  // nothing still has file nodes, and calling that a built graph would hide the
  // finding that the extractor got nothing out of any of them.
  const empty = !view || view.totalTerms === 0;
  const plural = (count: number, one: string, many: string) =>
    `${count} ${count === 1 ? one : many}`;

  /**
   * What the selected node is joined to, and by what.
   *
   * Every row carries its edge so the inspector can state the claim rather than
   * leave a line on a canvas to be interpreted. Files sort first: for a term,
   * which document it came from outranks what it happened to sit beside.
   */
  const links: Array<{ node: GraphNode; edge: GraphEdge }> = (() => {
    if (!selected || !view) return [];
    const byId = new Map(view.nodes.map((node) => [node.id, node]));
    const rows: Array<{ node: GraphNode; edge: GraphEdge }> = [];
    for (const edge of view.edges) {
      const otherId =
        edge.source === selected.id
          ? edge.target
          : edge.target === selected.id
            ? edge.source
            : null;
      if (!otherId) continue;
      const node = byId.get(otherId);
      if (node) rows.push({ node, edge });
    }
    return rows.sort(
      (a, b) =>
        Number(b.node.kind === 'document') - Number(a.node.kind === 'document') ||
        b.edge.weight - a.edge.weight ||
        a.node.label.localeCompare(b.node.label),
    );
  })();

  /**
   * The files the reader has picked, in the order they picked them.
   *
   * A `Set` iterates in insertion order, and that order is the whole basis of
   * the flow view's left-to-right sequence. Nothing in the extraction says one
   * document comes before another — the sequence is the reader's assertion, and
   * this is where it comes from.
   */
  const pickedFiles = [...picked]
    .map((id) => view?.nodes.find((node) => node.id === id))
    .filter((node): node is GraphNode => node?.kind === 'document');

  const flow = tracing ? buildFlow(pickedFiles, view?.nodes ?? [], view?.edges ?? []) : null;

  /** Content address to file name, for whatever files this view is showing. */
  const fileNames = new Map(
    (view?.nodes ?? [])
      .filter((node) => node.kind === 'document' && node.documentSha256)
      .map((node) => [node.documentSha256 as string, node.label] as const),
  );

  /**
   * How an evidence row cites itself.
   *
   * With files in the view a passage can name its document. Without them the
   * page number is all that is honestly available — "page 3" of eleven files
   * cites nothing, but naming a file the view cannot confirm would be worse.
   */
  const citation = (row: EvidenceRow): string => {
    const name = fileNames.get(row.documentSha256);
    return name ? `${name} · page ${row.page}` : `page ${row.page}`;
  };

  return (
    <div className={styles.panel}>
      <header className={styles.head}>
        <div className={styles.headText}>
          <span className={styles.title}>{tracing ? 'Flow' : 'Graph'}</span>
          {flow && (
            // The chain's own sentence, which leads with the gaps: two files
            // that share nothing is the finding, not a caveat on one.
            <span className={styles.counts}>{describeFlow(flow)}</span>
          )}
          {!tracing && view && !empty && (
            <span className={styles.counts}>
              {focus
                ? `${view.nodes.length} of ${view.totalNodes} nodes, around ${focus.label}`
                : [
                    plural(view.totalTerms, 'term', 'terms'),
                    showFiles ? plural(view.totalDocuments, 'file', 'files') : null,
                    `${view.edges.length} of ${view.totalEdges} links`,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
            </span>
          )}
        </div>
        <div className={styles.actions}>
          {tracing ? (
            <Button size="sm" variant="ghost" onClick={() => setTracing(false)}>
              <ArrowLeft size={14} /> Back to graph
            </Button>
          ) : (
            <>
              {focus && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => {
                    setFocus(null);
                    void load(null, showFiles);
                  }}
                >
                  <ArrowLeft size={14} /> Whole graph
                </Button>
              )}
              {pickedFiles.length >= 2 && (
                <Button size="sm" onClick={() => setTracing(true)}>
                  <GitBranch size={14} /> Trace flow ({pickedFiles.length} files)
                </Button>
              )}
            </>
          )}
          {!empty && !tracing && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => {
                const next = !showFiles;
                setShowFiles(next);
                // A file node cannot survive its own disappearance: drop the
                // focus and the selection if they were files.
                const wasFile = focus?.kind === 'document';
                if (!next && wasFile) setFocus(null);
                if (!next && selected?.kind === 'document') setSelected(null);
                void load(!next && wasFile ? null : (focus?.id ?? null), next);
              }}
              title="Draws the notebook's files as nodes, joined to the terms found in them."
            >
              {showFiles ? 'Hide files' : 'Show files'}
            </Button>
          )}
          {picked.size > 0 && onImport && view && !tracing && (
            <Button
              size="sm"
              loading={importing}
              disabled={importing}
              // Terms only. A file is not a term, and the renderer that turns a
              // selection into a chat attachment works over the term graph.
              onClick={() =>
                onImport(
                  view.nodes.filter(
                    (node) => picked.has(node.id) && node.kind === 'term',
                  ),
                )
              }
            >
              Ask about {picked.size} selected
            </Button>
          )}
          {!empty && !tracing && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void runTyping()}
              loading={typingBusy}
              disabled={typingBusy || busy}
              title="Asks the loaded model to type terms and name relations. Minutes per document."
            >
              Type with model
            </Button>
          )}
          {!empty && !tracing && (
            <Button
              size="sm"
              variant="ghost"
              onClick={() => void runRelations()}
              loading={relationBusy}
              disabled={relationBusy || typingBusy || busy}
              title="Names edges with REBEL, in its own CPU sidecar. Does not need a loaded chat model."
            >
              Name relations
            </Button>
          )}
          {!tracing && (
            <Button
              size="sm"
              variant="secondary"
              onClick={() => void build()}
              loading={busy}
              disabled={busy || typingBusy}
            >
              {empty ? 'Build graph' : 'Rebuild'}
            </Button>
          )}
        </div>
      </header>

      {error && <p className={styles.error}>{error}</p>}
      {outcome && <p className={styles.outcome}>{describeBuild(outcome)}</p>}

      {relations && !relationBusy && (
        <p className={styles.outcome}>
          {relations.kept} link{relations.kept === 1 ? '' : 's'} named from{' '}
          {relations.proposed} proposed
          {relations.droppedOffdomain > 0 &&
            ` \u00b7 ${relations.droppedOffdomain} off-domain`}
          {relations.droppedMisquoted > 0 &&
            ` \u00b7 ${relations.droppedMisquoted} not in the cited passage`}
          {relations.droppedUnknownTerm > 0 &&
            ` \u00b7 ${relations.droppedUnknownTerm} unknown term`}
          {relations.droppedUnknownEdge > 0 &&
            ` \u00b7 ${relations.droppedUnknownEdge} not an existing link`}
          {relations.documentsFailed > 0 &&
            ` \u00b7 ${relations.documentsFailed} failed`}
        </p>
      )}
      {relations?.problems.map((problem) => (
        <p key={problem} className={styles.typingDetail}>
          {problem}
        </p>
      ))}
      {typingBusy && (
        <p className={styles.outcome}>
          Reading passages with the loaded model. This takes minutes per document, and
          stops safely — running it again picks up where it left off.
        </p>
      )}

      {typing && (
        <div className={styles.outcome}>
          <p className={styles.typingHead}>
            {describeCoverage(typing)} · {describeRun(typing)}
          </p>
          {describeVerification(typing) && (
            <p className={styles.typingDetail}>{describeVerification(typing)}</p>
          )}
          {yieldIsPoor(typing) && (
            <p className={styles.typingWarn}>
              Most of what the model proposed could not be verified against the passages it
              cited. That usually means the loaded model is too small for this, rather than
              that the documents are empty.
            </p>
          )}
          {typing.problems.map((problem) => (
            <p key={problem} className={styles.typingDetail}>
              {problem}
            </p>
          ))}
        </div>
      )}

      {empty ? (
        <div className={styles.placeholder}>
          <Network size={26} />
          <p>
            {documentCount === 0
              ? 'Add documents first, then build the graph.'
              : 'No graph yet. Building reads the passages already extracted, and takes seconds — no model is involved.'}
          </p>
        </div>
      ) : (
        <div className={styles.body}>
          <div className={styles.canvasWrap}>
            {flow ? (
              <FlowCanvas
                flow={flow}
                selectedId={selected?.id ?? null}
                onSelect={(id) => {
                  const node = view?.nodes.find((candidate) => candidate.id === id);
                  if (node) void inspect(node);
                }}
              />
            ) : (
              <GraphCanvas
                nodes={view!.nodes}
                edges={view!.edges}
                focusId={focus?.id ?? null}
                selectedIds={picked}
                onSelect={(node) => void inspect(node)}
                onFocus={focusOn}
              />
            )}
            <p className={styles.hint}>
              {flow
                ? 'Left to right in the order you picked the files · each middle column is only what the files on either side share'
                : 'Drag a node to pull it about · click to inspect · double-click to centre · drag the background to pan · ctrl+scroll to zoom'}
            </p>
          </div>

          <aside className={styles.inspector}>
            {!selected ? (
              <p className={styles.muted}>
                Select a term to see where it came from, or a file to see what was found
                in it.
              </p>
            ) : selected.kind === 'document' ? (
              <>
                <h3 className={styles.nodeName}>{selected.label}</h3>
                <p className={styles.nodeMeta}>
                  file · {plural(links.length, 'term', 'terms')} found in it
                </p>
                <div className={styles.nodeActions}>
                  {!tracing && (
                    <Button size="sm" variant="ghost" onClick={() => focusOn(selected)}>
                      Centre here
                    </Button>
                  )}
                  {/* Picking files is what the flow view is built from, and the
                      order they are picked in is the order the chain reads. */}
                  <Button size="sm" variant="ghost" onClick={() => togglePicked(selected.id)}>
                    {picked.has(selected.id) ? 'Remove from flow' : 'Add to flow'}
                  </Button>
                </div>

                <p className={styles.evidenceTitle}>Found in this file</p>
                {links.length === 0 ? (
                  // Not an empty state to be tidied away. The extractor looks for
                  // title-case phrases and plant tags, and a file of code or a
                  // scan it could not read contains neither.
                  <p className={styles.muted}>
                    Nothing was extracted from this file, so it is not linked to anything.
                  </p>
                ) : (
                  <ul className={styles.evidence}>
                    {links.map(({ node, edge }) => (
                      <li key={node.id} className={styles.evidenceRow}>
                        <button
                          type="button"
                          className={styles.linkRow}
                          onClick={() => void inspect(node)}
                        >
                          {node.label}
                        </button>
                        <span className={styles.evidencePage}>{describeEdge(edge, selected.id)}</span>
                      </li>
                    ))}
                  </ul>
                )}
              </>
            ) : (
              <>
                <h3 className={styles.nodeName}>{selected.label}</h3>
                <p className={styles.nodeMeta}>
                  {/* "not yet typed" is a real state. The typing pass has not run,
                      and inventing a type here would be a guess shown as a fact. */}
                  {selected.nodeType ?? 'not yet typed'} · in {selected.occurrences}{' '}
                  {selected.occurrences === 1 ? 'passage' : 'passages'} ·{' '}
                  {plural(selected.documentCount, 'file', 'files')}
                </p>
                <div className={styles.nodeActions}>
                  <Button size="sm" variant="ghost" onClick={() => focusOn(selected)}>
                    Centre here
                  </Button>
                  <Button size="sm" variant="ghost" onClick={() => togglePicked(selected.id)}>
                    {picked.has(selected.id) ? 'Remove from selection' : 'Add to selection'}
                  </Button>
                </div>

                {links.length > 0 && (
                  <>
                    <p className={styles.evidenceTitle}>Linked to</p>
                    <ul className={styles.evidence}>
                      {links.map(({ node, edge }) => (
                        <li key={node.id} className={styles.evidenceRow}>
                          <button
                            type="button"
                            className={styles.linkRow}
                            onClick={() => void inspect(node)}
                          >
                            {node.label}
                          </button>
                          {/* Every link says what it claims. An unlabelled line
                              between two terms invites the reader to supply a
                              relation the extractor never observed. */}
                          <span className={styles.evidencePage}>{describeEdge(edge, selected.id)}</span>
                        </li>
                      ))}
                    </ul>
                  </>
                )}

                <p className={styles.evidenceTitle}>Where this came from</p>
                {evidence === null ? (
                  <p className={styles.muted}>Reading…</p>
                ) : evidence.length === 0 ? (
                  <p className={styles.muted}>No passages recorded.</p>
                ) : (
                  <ul className={styles.evidence}>
                    {evidence.slice(0, 12).map((row) => (
                      <li key={row.chunkId} className={styles.evidenceRow}>
                        {/* The file name, not the bare page number it used to
                            show. "page 3" of eleven documents cites nothing. */}
                        <span className={styles.evidencePage}>{citation(row)}</span>
                        {row.quote && <span className={styles.quote}>“{row.quote}”</span>}
                      </li>
                    ))}
                  </ul>
                )}
              </>
            )}
          </aside>
        </div>
      )}
    </div>
  );
};

export default NotebookGraphPanel;
