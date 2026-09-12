import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { AlertTriangle } from 'lucide-react';
import { Button } from '../ui';
import type { GraphNode, Notebook, NotebookDocument } from '../../services/notebook.service';
import {
  notebookResearchService,
  type Assertion,
  type ResolvedCitation,
} from '../../services/notebookResearch.service';
import { NotebookGraphPanel } from '../graph/NotebookGraphPanel';
import { NotebookChat } from './NotebookChat';
import { NotesPanel } from './NotesPanel';
import { RelationshipReview } from './RelationshipReview';
import { SourceList } from './SourceList';
import { SourceReader } from './SourceReader';
import styles from './workspace.module.css';

/**
 * One notebook, as a place to work rather than a list of files.
 *
 * ## The layout, and why the sources never leave the screen
 *
 * Sources on the left, always; everything else in tabs on the right. What the
 * next question will read is the single fact a person has to be able to see at
 * any moment, and putting it behind a tab means it can be wrong without being
 * visible. The tabs are the things you do *with* those sources — ask, keep,
 * explore, review — and only one of those is useful at a time.
 *
 * ## Selection state belongs to the notebook
 *
 * Source selection, the graph focus and the open reader are all cleared when the
 * notebook changes. Carrying a selection across would scope one notebook's
 * question with another notebook's ids — which the backend refuses, but the
 * refusal would arrive as a confusing error rather than as something that
 * cannot happen.
 */
type Tab = 'chat' | 'notes' | 'graph' | 'review';

interface NotebookWorkspaceProps {
  notebook: Notebook;
  sources: NotebookDocument[];
  /** Narrows the graph and the review list to one file. */
  scopedDocument: string | null;
  onScopeDocument: (documentSha256: string | null) => void;
  onRemoveSource: (source: NotebookDocument) => void;
  onAddDocuments: () => void;
  adding: boolean;
  /** Re-reads the notebook list and its documents. */
  onRefresh: () => void;
}

const TABS: { id: Tab; label: string }[] = [
  { id: 'chat', label: 'Ask' },
  { id: 'notes', label: 'Notes' },
  { id: 'graph', label: 'Graph' },
  { id: 'review', label: 'Review' },
];

export const NotebookWorkspace: React.FC<NotebookWorkspaceProps> = ({
  notebook,
  sources,
  scopedDocument,
  onScopeDocument,
  onRemoveSource,
  onAddDocuments,
  adding,
  onRefresh,
}) => {
  const [tab, setTab] = useState<Tab>('chat');
  const [selected, setSelected] = useState<Set<string>>(() => new Set());
  const [graphSelection, setGraphSelection] = useState<GraphNode[]>([]);
  const [assertionSelection, setAssertionSelection] = useState<
    { id: string; label: string }[]
  >([]);
  const [reader, setReader] = useState<{
    documentSha256: string;
    documentName: string;
    citation: ResolvedCitation | null;
  } | null>(null);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);
  const [pendingSaveKind, setPendingSaveKind] = useState<'summary' | 'comparison' | null>(
    null,
  );
  const [notesToken, setNotesToken] = useState(0);
  const [graphToken, setGraphToken] = useState(0);
  const [unreadable, setUnreadable] = useState<Set<string>>(() => new Set());
  const [error, setError] = useState<string | null>(null);

  // Everything that describes "what I am looking at" belongs to the notebook.
  useEffect(() => {
    setSelected(new Set());
    setGraphSelection([]);
    setAssertionSelection([]);
    setReader(null);
    setPendingPrompt(null);
    setPendingSaveKind(null);
    setError(null);
    setTab('chat');
  }, [notebook.id]);

  // A source that has left the notebook cannot stay selected: the backend
  // refuses a scope naming it, and the refusal would be the person's first
  // sign that anything had changed.
  useEffect(() => {
    const present = new Set(sources.map((source) => source.documentSha256));
    setSelected((current) => {
      const next = new Set([...current].filter((sha) => present.has(sha)));
      return next.size === current.size ? current : next;
    });
    setReader((current) =>
      current && !present.has(current.documentSha256) ? null : current,
    );
  }, [sources]);

  // Which sources cannot be read, so the list can say so before a question is
  // asked rather than after one is answered.
  useEffect(() => {
    let live = true;
    notebookResearchService
      .scopePreview({ notebookId: notebook.id, sourceSha256s: [] })
      .then((preview) => {
        if (!live) return;
        setUnreadable(
          new Set(
            sources
              .filter((source) => preview.unreadable.includes(source.documentName))
              .map((source) => source.documentSha256),
          ),
        );
      })
      .catch(() => {
        if (live) setUnreadable(new Set());
      });
    return () => {
      live = false;
    };
  }, [notebook.id, sources]);

  const selectedList = useMemo(() => [...selected], [selected]);
  const selectedNames = useMemo(
    () =>
      (selected.size === 0
        ? sources
        : sources.filter((s) => selected.has(s.documentSha256))
      ).map((source) => source.documentName),
    [selected, sources],
  );

  /** Opens the reader at a page, without claiming a passage was matched. */
  const openReaderAt = useCallback(
    (documentSha256: string, page: number) => {
      const source = sources.find((row) => row.documentSha256 === documentSha256);
      setReader({
        documentSha256,
        documentName: source?.documentName ?? 'this source',
        citation: {
          marker: 0,
          state: 'available',
          documentSha256,
          documentName: source?.documentName ?? '',
          page,
          sectionPath: [],
          // No quote and no offsets: this is a jump to a page, so the reader
          // shows the page plainly rather than highlighting something it was
          // not asked to find.
          quote: '',
          pageText: null,
          highlightStart: null,
          highlightLength: null,
          problem: null,
        },
      });
    },
    [sources],
  );

  const onOpenCitation = useCallback((citation: ResolvedCitation) => {
    setReader({
      documentSha256: citation.documentSha256,
      documentName: citation.documentName,
      citation,
    });
  }, []);

  const askAboutSelection = useCallback((nodes: GraphNode[]) => {
    if (nodes.length === 0) return;
    setGraphSelection(nodes);
    setAssertionSelection([]);
    const named =
      nodes.length <= 4
        ? nodes.map((n) => n.label).join(', ')
        : `${nodes
            .slice(0, 4)
            .map((n) => n.label)
            .join(', ')} and ${nodes.length - 4} more`;
    setPendingPrompt(
      `What do my sources say about ${named}? Set out how they connect, and cite the ` +
        `passage behind each point.`,
    );
    setTab('chat');
  }, []);

  const askAboutAssertion = useCallback((assertion: Assertion) => {
    setAssertionSelection([
      {
        id: assertion.id,
        label: `${assertion.subjectLabel} → ${assertion.predicate} → ${assertion.objectLabel}`,
      },
    ]);
    setGraphSelection([]);
    setPendingPrompt(
      `Do my sources support this: ${assertion.subjectLabel} ${assertion.predicate} ` +
        `${assertion.objectLabel}? Quote what they actually say, and say plainly if they ` +
        `do not support it.`,
    );
    setTab('chat');
  }, []);

  const saveAnswer = useCallback(
    async (conversationId: string, messageId: string, suggestedTitle: string) => {
      const title = window.prompt('Save this answer as:', suggestedTitle || 'Saved answer');
      if (title === null || !title.trim()) return;
      setError(null);
      try {
        await notebookResearchService.saveAnswer({
          notebookId: notebook.id,
          conversationId,
          messageId,
          title: title.trim(),
          kind: pendingSaveKind ?? 'answer',
        });
        setPendingSaveKind(null);
        setNotesToken((n) => n + 1);
        setTab('notes');
      } catch (err) {
        setError(String(err));
      }
    },
    [notebook.id, pendingSaveKind],
  );

  return (
    <div className={styles.workspace} data-reading={reader ? 'true' : undefined}>
      <aside className={styles.sourcesPane}>
        <header className={styles.paneHead}>
          <div className={styles.paneTitleBlock}>
            <h2 className={styles.notebookName}>{notebook.name}</h2>
            <p className={styles.notebookMeta}>
              {notebook.documentCount}{' '}
              {notebook.documentCount === 1 ? 'source' : 'sources'}
            </p>
          </div>
          <Button size="sm" onClick={onAddDocuments} loading={adding} disabled={adding}>
            Add
          </Button>
        </header>

        <SourceList
          sources={sources}
          selected={selected}
          unreadable={unreadable}
          openSha={reader?.documentSha256 ?? null}
          busy={adding}
          onToggle={(sha) =>
            setSelected((current) => {
              const next = new Set(current);
              // An empty set means "all", so the first tick has to mean "only
              // this one" rather than "all of them except this one".
              if (next.size === 0) {
                next.add(sha);
                return next;
              }
              if (next.has(sha)) next.delete(sha);
              else next.add(sha);
              return next;
            })
          }
          onSelectAll={() => setSelected(new Set())}
          onSelectNone={() => setSelected(new Set())}
          onOpen={(source) => {
            onScopeDocument(source.documentSha256);
            setReader({
              documentSha256: source.documentSha256,
              documentName: source.documentName,
              citation: null,
            });
          }}
          onRemove={onRemoveSource}
        />
      </aside>

      <section className={styles.mainPane}>
        <nav className={styles.tabs} role="tablist" aria-label="Notebook views">
          {TABS.map((entry) => (
            <button
              key={entry.id}
              type="button"
              role="tab"
              aria-selected={tab === entry.id}
              className={styles.tab}
              data-active={tab === entry.id || undefined}
              onClick={() => setTab(entry.id)}
            >
              {entry.label}
            </button>
          ))}
          <span className={styles.tabSpacer} />
          <button type="button" className={styles.linkAction} onClick={onRefresh}>
            Refresh
          </button>
        </nav>

        {error && (
          <p className={styles.error}>
            <AlertTriangle size={13} /> {error}
          </p>
        )}

        <div className={styles.tabBody}>
          {/* The chat stays mounted across tabs: unmounting it would drop an
              answer mid-stream the moment somebody looked at their notes. */}
          <div className={styles.tabPanel} hidden={tab !== 'chat'}>
            <NotebookChat
              notebookId={notebook.id}
              notebookName={notebook.name}
              selectedSources={selectedList}
              graphSelection={graphSelection}
              assertionSelection={assertionSelection}
              onClearGraphSelection={() => {
                setGraphSelection([]);
                setAssertionSelection([]);
              }}
              onOpenCitation={onOpenCitation}
              onSaveAnswer={(conversationId, messageId, suggested) =>
                void saveAnswer(conversationId, messageId, suggested)
              }
              pendingPrompt={pendingPrompt}
              onPendingPromptConsumed={() => setPendingPrompt(null)}
            />
          </div>

          {tab === 'notes' && (
            <NotesPanel
              notebookId={notebook.id}
              reloadToken={notesToken}
              selectedSourceNames={selectedNames}
              onOpenEvidence={openReaderAt}
              onGenerateReport={(prompt, kind) => {
                setPendingSaveKind(kind);
                setPendingPrompt(prompt);
                setTab('chat');
              }}
            />
          )}

          {tab === 'graph' && (
            <NotebookGraphPanel
              // Re-keyed after a review, so an accepted or corrected
              // relationship shows on the canvas without a manual refresh.
              key={`${notebook.id}:${graphToken}`}
              notebookId={notebook.id}
              documentCount={notebook.documentCount}
              documentSha256={scopedDocument}
              onImport={askAboutSelection}
              importing={false}
            />
          )}

          {tab === 'review' && (
            <RelationshipReview
              notebookId={notebook.id}
              documentSha256={scopedDocument}
              onAskAbout={askAboutAssertion}
              onOpenEvidence={openReaderAt}
              onChanged={() => setGraphToken((n) => n + 1)}
            />
          )}
        </div>
      </section>

      {reader && (
        <SourceReader
          notebookId={notebook.id}
          documentSha256={reader.documentSha256}
          documentName={reader.documentName}
          citation={reader.citation}
          onClose={() => setReader(null)}
        />
      )}
    </div>
  );
};
