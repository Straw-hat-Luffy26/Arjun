/**
 * The notebook research workspace: sources, citations, notes and review.
 *
 * Thin, like every other service here — the object literal is the whole layer.
 * No try/catch: a rejection carries the backend's own sentence, and the screen
 * that called is the only place that knows how to show it.
 *
 * Everything here takes and returns identifiers and backend-resolved values.
 * Nothing in this file constructs evidence, a citation label or a passage; all
 * of that is resolved in Rust against the store for the signed-in owner. See
 * `src-tauri/src/commands/notebook_research.rs`.
 */
import { getBackendService } from './api';
import type { ResearchScope, SourceSelection } from './agent.service';

export type { ResearchScope, SourceSelection };

// ── Source readiness ─────────────────────────────────────────────────────

/**
 * How far along one source is, from added to answerable.
 *
 * Mirrors `knowledge::source_readiness::SourceState`. The screen shows one
 * badge per source; `ready` and `partiallyReady` are the only two a question
 * may be answered from.
 */
export type SourceState =
  | 'queued'
  | 'reading'
  | 'extracting'
  | 'needsVision'
  | 'indexing'
  | 'ready'
  | 'partiallyReady'
  | 'failed'
  | 'unavailable';

/** Whether a question may draw evidence from a source in this state. */
export function usableAsEvidence(state: SourceState): boolean {
  return state === 'ready' || state === 'partiallyReady';
}

/** Whether the source is still being worked on, so the screen should poll. */
export function inProgress(state: SourceState): boolean {
  return (
    state === 'queued' ||
    state === 'reading' ||
    state === 'extracting' ||
    state === 'indexing'
  );
}

/**
 * Precisely what is wrong with a source, in a form the screen can act on.
 *
 * Mirrors `ReadinessReason`, tagged by `kind`. One variant per distinct repair:
 * "missing", "corrupt" and "this notebook was never given access to it" used to
 * be the same word on screen, and they have three different answers.
 */
export type ReadinessReason =
  | { kind: 'notAssociated' }
  | { kind: 'missingExtraction' }
  | { kind: 'storeUnavailable'; problem: string }
  | { kind: 'incompatibleExtraction'; problem: string }
  | { kind: 'missingOriginal' }
  | { kind: 'passwordProtected' }
  | { kind: 'missingParser'; needed: string }
  | { kind: 'conversionFailure'; tool: string; problem: string }
  | { kind: 'noTextExtracted' }
  | { kind: 'requiresVision' }
  | { kind: 'visionUnavailable' }
  | { kind: 'unreadablePages'; pages: number[]; total: number }
  | { kind: 'partialExtraction'; read: number; total: number }
  | { kind: 'notIndexed'; stored: number; expected: number };

/** What the screen may offer for a reason. */
export type Repair = 'associate' | 'reprocess' | 'addAgain' | 'retry';

/** One source, and everything known about whether it can be read. */
export interface SourceReadiness {
  documentSha256: string;
  documentName: string;
  state: SourceState;
  reasons: ReadinessReason[];
  extractionKind: string | null;
  sourceRevision: string | null;
  pagesTotal: number;
  pagesWithText: number;
  passages: number;
  repair: Repair | null;
}

/**
 * The counts a screen needs to stop contradicting itself.
 *
 * Selected and readable are different facts. A preview carrying only the first
 * is how "1 of 1 will be read" came to sit above a warning that the one source
 * could not be read.
 */
export interface ReadinessCounts {
  total: number;
  selected: number;
  readable: number;
  partial: number;
  blocked: number;
  inProgress: number;
}

/** The notebook and selection a conversation was last left on. */
export interface ConversationScope {
  notebookId: string;
  notebookName: string;
  /**
   * `null` for a thread bound before selections were recorded. Shown as
   * "choose sources" — never silently treated as all of them.
   */
  selection: SourceSelection | null;
}

// ── Conversations bound to a notebook ────────────────────────────────────

/** One thread started from a notebook. */
export interface NotebookThread {
  conversationId: string;
  title: string;
  lastActivityAt: string;
  /** Turns somebody actually had — the system welcome is not counted. */
  messageCount: number;
}

// ── Evidence ─────────────────────────────────────────────────────────────

/** How the passages for a turn were found. Reported, never guessed. */
export type RetrievalMode = 'keyword' | 'semantic' | 'hybrid';

/** One passage an answer was built on, as it can be found again. */
export interface EvidenceEntry {
  /** The number the answer cites: `[E1]` is marker 1. */
  marker: number;
  chunkId: string;
  documentSha256: string;
  documentName: string;
  page: number;
  sectionPath: string[];
  /** The extraction this passage was read from. */
  sourceRevision: string;
  quote: string;
}

/** Everything needed to reconstruct what an answer used. */
export interface EvidenceManifest {
  runId: string;
  notebookId: string;
  conversationId: string;
  messageId: string;
  createdAt: string;
  scope: ResearchScope;
  retrievalMode: RetrievalMode;
  entries: EvidenceEntry[];
  /** What the turn could not do, in sentences meant for a person. */
  limitations: string[];
  graphRevision: string | null;
  /** Selected sources that contributed no passage, by name. */
  sourcesUnused: string[];
}

/**
 * Why a citation cannot be opened, when it cannot.
 *
 * Four states rather than a boolean, because "this was never real",
 * "the document has been removed since" and "the document has been re-read
 * since" call for completely different reactions.
 */
export type CitationState = 'available' | 'sourceRemoved' | 'sourceChanged' | 'unavailable';

/** One citation, resolved for the reader. */
export interface ResolvedCitation {
  marker: number;
  state: CitationState;
  documentSha256: string;
  documentName: string;
  page: number;
  sectionPath: string[];
  /** The passage as the answer used it. Always present. */
  quote: string;
  /** The page as it reads now, when it can still be read. */
  pageText: string | null;
  /** Where the quote begins in `pageText`, in characters. */
  highlightStart: number | null;
  highlightLength: number | null;
  problem: string | null;
}

/** One page of a source, for the reader. */
export interface SourcePage {
  documentSha256: string;
  documentName: string;
  page: number;
  totalPages: number;
  text: string;
  /** The reader stopped early when this file was first read. */
  sourceTruncated: boolean;
  /** `pdf-text`, `pdf-scan`, `image`, `docx` — so a reader knows what they see. */
  extractionKind: string;
  extractedAt: string;
}

// ── Notes ────────────────────────────────────────────────────────────────

/** Where a note's text came from. */
export type NoteKind = 'written' | 'answer' | 'summary' | 'comparison';

export interface Note {
  id: string;
  notebookId: string;
  title: string;
  body: string;
  kind: NoteKind;
  sourceConversationId: string | null;
  sourceMessageId: string | null;
  /** The manifest the text was built on, as JSON. */
  evidenceJson: string | null;
  /**
   * True when the body no longer matches what was saved.
   *
   * A cited answer somebody has since typed into is no longer wholly a cited
   * answer, and the interface must stop presenting it as one.
   */
  editedSinceSaved: boolean;
  createdAt: string;
  updatedAt: string;
}

// ── Relationships ────────────────────────────────────────────────────────

export type AssertionProvenance = 'model' | 'user';
export type AssertionStatus = 'proposed' | 'accepted' | 'rejected';

export interface AssertionEvidence {
  chunkId: string;
  documentSha256: string;
  page: number;
  quote: string | null;
}

/** A directed claim about two terms. */
export interface Assertion {
  id: string;
  notebookId: string;
  documentSha256: string | null;
  subject: string;
  subjectLabel: string;
  predicate: string;
  object: string;
  objectLabel: string;
  provenance: AssertionProvenance;
  status: AssertionStatus;
  /**
   * False for a claim recovered from the old undirected storage, where the
   * direction was destroyed before it could be preserved. Shown as "direction
   * unverified" rather than guessed.
   */
  directionCertain: boolean;
  extractor: string;
  extractorVersion: number;
  sourceRevision: string | null;
  /** The supporting evidence no longer stands. */
  stale: boolean;
  note: string | null;
  evidence: AssertionEvidence[];
  createdAt: string;
  updatedAt: string;
  reviewedAt: string | null;
}

// ── Scope preview ────────────────────────────────────────────────────────

/** What the next question would use, before it is asked. */
export interface ScopePreview {
  notebookName: string;
  sourcesSelected: number;
  sourcesTotal: number;
  /** Selected sources whose text cannot be read, by name. */
  unreadable: string[];
  /**
   * Selected, readable, partial, blocked and in-progress, counted apart.
   *
   * Nothing may say "will be read" from anything but `counts.readable`.
   */
  counts: ReadinessCounts;
  /** Every selected source with its state and reasons. */
  sources: SourceReadiness[];
  focusLabels: string[];
  focusRelationships: string[];
  retrievalMode: RetrievalMode;
  retrievalExplanation: string;
  graphBuilt: boolean;
}

export const notebookResearchService = {
  // ── Threads ────────────────────────────────────────────────────────────

  /** Records that a conversation belongs to a notebook. */
  bindConversation(notebookId: string, conversationId: string): Promise<void> {
    return getBackendService().invoke<void>('notebook_bind_conversation', {
      notebookId,
      conversationId,
    });
  },

  threads(notebookId: string): Promise<NotebookThread[]> {
    return getBackendService().invoke<NotebookThread[]>('notebook_threads', { notebookId });
  },

  // ── Evidence and the reader ────────────────────────────────────────────

  /** The manifest for one answer. `null` for a turn that was not scoped. */
  turnEvidence(conversationId: string, messageId: string): Promise<EvidenceManifest | null> {
    return getBackendService().invoke<EvidenceManifest | null>('notebook_turn_evidence', {
      conversationId,
      messageId,
    });
  },

  /** Opens one citation at the passage it points at. */
  openCitation(
    conversationId: string,
    messageId: string,
    marker: number,
  ): Promise<ResolvedCitation> {
    return getBackendService().invoke<ResolvedCitation>('notebook_open_citation', {
      conversationId,
      messageId,
      marker,
    });
  },

  sourcePage(notebookId: string, documentSha256: string, page: number): Promise<SourcePage> {
    return getBackendService().invoke<SourcePage>('notebook_source_page', {
      notebookId,
      documentSha256,
      page,
    });
  },

  // ── Notes ──────────────────────────────────────────────────────────────

  notes(notebookId: string): Promise<Note[]> {
    return getBackendService().invoke<Note[]>('notebook_notes', { notebookId });
  },

  createNote(notebookId: string, title: string, body?: string): Promise<Note> {
    return getBackendService().invoke<Note>('notebook_create_note', {
      notebookId,
      title,
      body: body ?? '',
    });
  },

  /**
   * Changes a note's title, its body, or both.
   *
   * Editing the text of a cited answer does not re-verify it, and the note
   * comes back reporting that it has been edited since.
   */
  updateNote(
    notebookId: string,
    noteId: string,
    changes: { title?: string; body?: string },
  ): Promise<Note> {
    return getBackendService().invoke<Note>('notebook_update_note', {
      notebookId,
      noteId,
      title: changes.title ?? null,
      body: changes.body ?? null,
    });
  },

  deleteNote(notebookId: string, noteId: string): Promise<void> {
    return getBackendService().invoke<void>('notebook_delete_note', { notebookId, noteId });
  },

  /**
   * Saves an assistant answer into the notebook with the evidence it used.
   *
   * The body is read out of the conversation store by the backend — it is not
   * sent from here, because a caller that could supply it could save anything
   * as a cited answer.
   */
  saveAnswer(request: {
    notebookId: string;
    conversationId: string;
    messageId: string;
    title: string;
    kind?: Exclude<NoteKind, 'written'>;
  }): Promise<Note> {
    return getBackendService().invoke<Note>('notebook_save_answer', { request });
  },

  // ── Relationship review ────────────────────────────────────────────────

  assertions(
    notebookId: string,
    options?: { documentSha256?: string | null; includeRejected?: boolean },
  ): Promise<Assertion[]> {
    return getBackendService().invoke<Assertion[]>('notebook_assertions', {
      notebookId,
      documentSha256: options?.documentSha256 ?? null,
      includeRejected: options?.includeRejected ?? false,
    });
  },

  reviewAssertion(
    notebookId: string,
    assertionId: string,
    status: AssertionStatus,
    note?: string,
  ): Promise<Assertion> {
    return getBackendService().invoke<Assertion>('notebook_review_assertion', {
      notebookId,
      assertionId,
      status,
      note: note ?? null,
    });
  },

  /** Corrects a relationship's wording or which way round it runs. */
  correctAssertion(request: {
    notebookId: string;
    assertionId: string;
    subject: string;
    predicate: string;
    object: string;
    note?: string;
  }): Promise<Assertion> {
    return getBackendService().invoke<Assertion>('notebook_correct_assertion', { request });
  },

  /** Records a relationship a person is asserting themselves, with no evidence. */
  createAssertion(request: {
    notebookId: string;
    subject: string;
    predicate: string;
    object: string;
    note?: string;
  }): Promise<Assertion> {
    return getBackendService().invoke<Assertion>('notebook_create_assertion', { request });
  },

  deleteAssertion(notebookId: string, assertionId: string): Promise<void> {
    return getBackendService().invoke<void>('notebook_delete_assertion', {
      notebookId,
      assertionId,
    });
  },

  // ── Scope ──────────────────────────────────────────────────────────────

  /** What the next question would use, before it is asked. */
  scopePreview(scope: ResearchScope): Promise<ScopePreview> {
    return getBackendService().invoke<ScopePreview>('notebook_scope_preview', { scope });
  },

  // ── Readiness and repair ───────────────────────────────────────────────

  /**
   * Whether each source in a notebook can actually be read.
   *
   * Distinct from `notebookService.documents`, which answers "what is in this
   * notebook" — a membership question that says nothing about whether any of it
   * has readable text.
   */
  sourceStatus(notebookId: string): Promise<SourceReadiness[]> {
    return getBackendService().invoke<SourceReadiness[]>('notebook_source_status', {
      notebookId,
    });
  },

  /**
   * Repairs one source that is readable on disk and unreachable from here.
   *
   * Records the access the notebook is missing. Re-reads nothing and runs no
   * model, because the text has been on disk the whole time. Returns the source
   * as it stands afterwards, repaired or not — a source whose extraction is
   * genuinely gone comes back still broken rather than pretending.
   */
  repairSource(notebookId: string, documentSha256: string): Promise<SourceReadiness> {
    return getBackendService().invoke<SourceReadiness>('notebook_repair_source', {
      notebookId,
      documentSha256,
    });
  },

  // ── The notebook a conversation is asking ──────────────────────────────

  /**
   * Remembers which notebook this conversation is asking, and of what.
   *
   * The preference only. Each turn's scope is frozen onto its own manifest when
   * the message is submitted, so changing this changes the next question and no
   * answer already given.
   */
  setConversationScope(
    conversationId: string,
    notebookId: string,
    selection: SourceSelection,
  ): Promise<void> {
    return getBackendService().invoke<void>('notebook_set_conversation_scope', {
      conversationId,
      notebookId,
      selection,
    });
  },

  /** The notebook and selection a conversation was left on, if any. */
  conversationScope(conversationId: string): Promise<ConversationScope | null> {
    return getBackendService().invoke<ConversationScope | null>(
      'notebook_conversation_scope',
      { conversationId },
    );
  },

  /**
   * Takes the notebook off a conversation and leaves its evidence alone.
   *
   * Deliberately not the unbind call, which also deletes every manifest in the
   * thread: clearing the chip must not stop the citations in answers already
   * given from resolving.
   */
  clearConversationScope(conversationId: string): Promise<void> {
    return getBackendService().invoke<void>('notebook_clear_conversation_scope', {
      conversationId,
    });
  },
};
