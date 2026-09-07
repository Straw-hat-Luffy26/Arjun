/**
 * Notebooks: named libraries of documents that outlive a conversation.
 *
 * Thin, like every other service here — the object literal is the whole layer.
 * No try/catch: a rejection carries the backend's own sentence, and the screen
 * that called is the only place that knows how to show it.
 */
import { getBackendService } from './api';
import type { ComposerAttachment } from './agent.service';

/** A named library of documents. Mirrors `knowledge::graph::store::Notebook`. */
export interface Notebook {
  id: string;
  name: string;
  /** RFC 3339 UTC. */
  createdAt: string;
  updatedAt: string;
  documentCount: number;
}

/** One document's membership of a notebook. */
export interface NotebookDocument {
  notebookId: string;
  documentSha256: string;
  documentName: string;
  addedAt: string;
}

/**
 * What became of one file the caller asked to add.
 *
 * Every requested name comes back, including the ones that failed. `added` and
 * `problem` are separate facts: a document the notebook already had is
 * `added: false` with a problem that says so, which is not the same as a file
 * that could not be read.
 */
export interface AddedDocument {
  name: string;
  sha256: string | null;
  pages: number;
  added: boolean;
  problem: string | null;
}

export const notebookService = {
  list(): Promise<Notebook[]> {
    return getBackendService().invoke<Notebook[]>('notebook_list');
  },

  create(name: string): Promise<Notebook> {
    return getBackendService().invoke<Notebook>('notebook_create', { name });
  },

  /** Renames a notebook, keeping its documents and its graph. */
  rename(notebookId: string, name: string): Promise<Notebook> {
    return getBackendService().invoke<Notebook>('notebook_rename', { notebookId, name });
  },

  /**
   * Deletes a notebook and the graph built over it.
   *
   * The documents are not deleted. A notebook is a way of grouping files, and
   * removing the grouping must not remove the source material.
   */
  delete(notebookId: string): Promise<void> {
    return getBackendService().invoke<void>('notebook_delete', { notebookId });
  },

  /** Takes one document out, with the graph evidence that came from it. */
  removeDocument(notebookId: string, documentSha256: string): Promise<void> {
    return getBackendService().invoke<void>('notebook_remove_document', {
      notebookId,
      documentSha256,
    });
  },

  documents(notebookId: string): Promise<NotebookDocument[]> {
    return getBackendService().invoke<NotebookDocument[]>('notebook_documents', { notebookId });
  },

  /**
   * Reads files and puts them in a notebook.
   *
   * Slow by nature — a scanned page goes through OCR — so callers should show
   * that work is happening rather than awaiting it silently.
   */
  addDocuments(notebookId: string, attachments: ComposerAttachment[]): Promise<AddedDocument[]> {
    return getBackendService().invoke<AddedDocument[]>('notebook_add_documents', {
      request: { notebookId, attachments },
    });
  },
};

/**
 * What a node stands for.
 *
 * A notebook's graph holds two populations: terms the extractor found, and the
 * notebook's own files. Keeping them apart in the type is what lets the canvas
 * draw them differently and the inspector say different things about them — a
 * file is not a claim about anything, it is where claims came from.
 */
export type NodeKind = 'term' | 'document';

/**
 * What a link means.
 *
 * `cooccurrence` is all the statistical pass can observe: the two terms were in
 * the same passage. `appearsIn` joins a term to the file it was found in, and is
 * the one edge in the graph that is a fact of the corpus rather than an
 * inference — it needs no model and is never typed.
 */
export type EdgeKind = 'cooccurrence' | 'appearsIn';

/** A node as the graph view returns it. Mirrors `knowledge::graph::persist`. */
export interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  /** `null` until the typing pass has run — a real state, not a missing value. */
  nodeType: string | null;
  occurrences: number;
  degree: number;
  /** Set on a file node only: the content address it stands for. */
  documentSha256: string | null;
  /** For a term, how many of the notebook's files it was found in. */
  documentCount: number;
}

export interface GraphEdge {
  source: string;
  target: string;
  kind: EdgeKind;
  /** Passages containing both. A count, not a score. */
  weight: number;
  relation: string | null;
}


/**
 * One view of a notebook's graph.
 *
 * `totalNodes` describes the whole graph, not this view — both numbers are
 * needed to say "showing 40 of 912" rather than implying the documents held
 * only forty things.
 */
export interface GraphView {
  nodes: GraphNode[];
  edges: GraphEdge[];
  totalNodes: number;
  totalEdges: number;
  /** The two populations behind `totalNodes`, so a header can say "7 terms · 11 files". */
  totalTerms: number;
  totalDocuments: number;
}

/** What one build actually did. Every number is counted, none estimated. */
export interface BuildOutcome {
  documentsTotal: number;
  documentsBuilt: number;
  documentsSkipped: number;
  documentsUnreadable: number;
  chunks: number;
  candidatesFound: number;
  nodesKept: number;
  droppedRare: number;
  droppedGeneric: number;
  problems: string[];
}

/**
 * What one run of the relation pass did.
 *
 * The drop counts mean different things and are reported separately.
 * `droppedOffdomain` runs high on a healthy pass over industrial documents -
 * REBEL proposes Wikipedia relations and the allowlist refuses them, which is
 * the allowlist working. `droppedMisquoted` is the one to watch: it counts
 * relations between things the cited passage does not contain.
 */
export interface RelationOutcome {
  documentsTotal: number;
  documentsNamed: number;
  documentsSkipped: number;
  documentsFailed: number;
  proposed: number;
  kept: number;
  droppedUncited: number;
  droppedMisquoted: number;
  droppedUnknownTerm: number;
  droppedUnknownEdge: number;
  droppedOffdomain: number;
  problems: string[];
}

/** One passage behind a node. */
export interface EvidenceRow {
  chunkId: string;
  documentSha256: string;
  page: number;
  quote: string | null;
}

/**
 * What one run of the typing pass did.
 *
 * The drop counts are part of the result, not a log line. A pass that kept nine
 * of two hundred proposals is reporting a problem — wrong model, scanned noise —
 * and a screen showing only the nine would read as a small graph.
 */
export interface TypingOutcome {
  documentsTotal: number;
  documentsTyped: number;
  documentsSkipped: number;
  documentsFailed: number;
  proposedNodes: number;
  proposedEdges: number;
  keptNodes: number;
  keptEdges: number;
  droppedUnknownType: number;
  droppedUnknownTerm: number;
  droppedUncited: number;
  /** Claims whose quote was not in the passage they cited: fabrication, caught. */
  droppedMisquoted: number;
  typedTerms: number;
  totalTerms: number;
  problems: string[];
}

export const notebookGraphService = {
  /** Runs the statistical pass. Resumable: already-built documents are skipped. */
  build(notebookId: string, rebuild = false): Promise<BuildOutcome> {
    return getBackendService().invoke<BuildOutcome>('notebook_build_graph', {
      notebookId,
      rebuild,
    });
  },

  /**
   * A view of the graph.
   *
   * Pass a `focus` for the local view. The global view (no focus) is an
   * unreadable tangle past a few hundred nodes, so callers should default to
   * local and make global an explicit choice.
   *
   * `withDocuments` puts the notebook's files in as nodes joined to the terms
   * found in them. On by default: which file a term came from is the relation a
   * person opens a notebook graph to see, and without it the graph is a cloud of
   * terms with no way back to a source.
   *
   * `documentSha256` narrows the view, and its counts, to one file. Different
   * from focusing that file's node: a focus walks outward and reaches terms
   * other files contributed, which answers "what does this connect to". Scoping
   * answers "what is in this", and must not include anything else.
   */
  graph(
    notebookId: string,
    documentSha256?: string | null,
    focus?: string | null,
    depth = 1,
    minWeight = 1,
    withDocuments = true,
  ): Promise<GraphView> {
    return getBackendService().invoke<GraphView>('notebook_graph', {
      notebookId,
      documentSha256: documentSha256 ?? null,
      focus: focus ?? null,
      depth,
      minWeight,
      withDocuments,
    });
  },

  /**
   * Asks the already-loaded local model to type the graph.
   *
   * Slow — minutes per document on a workstation model — and resumable, so a
   * caller that gives up can run it again and pick up where it stopped.
   */
  type(notebookId: string): Promise<TypingOutcome> {
    return getBackendService().invoke<TypingOutcome>('notebook_type_graph', { notebookId });
  },

  /**
   * Names the graph's edges with Babelscape/rebel-large.
   *
   * Unlike `type`, this does not need a warm chat model: REBEL is a 400M
   * sequence-to-sequence extractor that runs on CPU in its own sidecar, so the
   * pass starts on demand and evicts nothing from VRAM.
   *
   * Slow - seconds per passage - and resumable per document, so a caller that
   * gives up can run it again and pick up where it stopped.
   */
  extractRelations(notebookId: string): Promise<RelationOutcome> {
    return getBackendService().invoke<RelationOutcome>('notebook_extract_relations', {
      notebookId,
    });
  },

  nodeEvidence(notebookId: string, nodeId: string): Promise<EvidenceRow[]> {
    return getBackendService().invoke<EvidenceRow[]>('notebook_node_evidence', {
      notebookId,
      nodeId,
    });
  },
};

/** A subgraph rendered as text, ready to attach to a chat turn. */
export interface RenderedSubgraph {
  name: string;
  markdown: string;
  nodeCount: number;
  edgeCount: number;
}

/**
 * Renders a chosen subgraph for a chat turn.
 *
 * The caller attaches the result as an ordinary document. That is the whole
 * design: the existing attachment path already gives it a content address,
 * chunking, a token budget, a row in the context meter, and pinning by that
 * address — so importing a graph needed no new context plumbing at all.
 */
export function renderSubgraph(notebookId: string, nodeIds: string[]): Promise<RenderedSubgraph> {
  return getBackendService().invoke<RenderedSubgraph>('notebook_render_subgraph', {
    notebookId,
    nodeIds,
  });
}
