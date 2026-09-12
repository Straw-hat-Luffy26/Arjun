import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, BookmarkPlus, Send, X } from 'lucide-react';
import { Button, Spinner } from '../ui';
import { useConversation } from '../../contexts/ConversationContext';
import {
  notebookResearchService,
  type EvidenceManifest,
  type ResolvedCitation,
  type ScopePreview,
} from '../../services/notebookResearch.service';
import type { ResearchScope } from '../../services/agent.service';
import type { GraphNode } from '../../services/notebook.service';
import { Markdown } from '../chat/Markdown';
import styles from './workspace.module.css';

/**
 * Asking the notebook a question, and getting an answer you can check.
 *
 * ## The scope is visible before the question is sent
 *
 * Every turn carries which sources it may read and which graph selection is
 * narrowing it, and the bar above the composer says so in words the answer will
 * repeat. That is the whole point of showing it *before*: discovering after a
 * two-minute answer that three of your eight sources were unreadable is finding
 * out too late to have asked differently.
 *
 * ## Citations are buttons, not decoration
 *
 * The backend records a manifest for every notebook turn — which chunk, which
 * document, which page, at which extraction revision. `[E1]` in the answer is
 * rendered as a control that resolves against that manifest and opens the
 * reader on the passage. A marker with no manifest entry says so rather than
 * doing nothing.
 */
interface NotebookChatProps {
  notebookId: string;
  notebookName: string;
  /** Sources the next question may use. Empty means every source. */
  selectedSources: string[];
  /** Graph nodes narrowing the question, with their labels for the chip. */
  graphSelection: GraphNode[];
  /** Directed claims narrowing the question. */
  assertionSelection: { id: string; label: string }[];
  onClearGraphSelection: () => void;
  /** Opens the reader at a resolved citation. */
  onOpenCitation: (citation: ResolvedCitation) => void;
  /**
   * Saves an answer into the notebook's notes.
   *
   * The conversation id is passed out rather than looked up by the parent: the
   * thread a notebook is on is this component's state, and a parent reading it
   * from anywhere else would eventually read a stale one.
   */
  onSaveAnswer: (
    conversationId: string,
    messageId: string,
    suggestedTitle: string,
  ) => void;
  /** A prompt pushed in from elsewhere — "Ask about selection", or a report. */
  pendingPrompt: string | null;
  onPendingPromptConsumed: () => void;
}

/** Splits an answer into text and the `[En]` markers inside it. */
function withCitations(
  text: string,
  onMarker: (marker: number) => void,
): React.ReactNode[] {
  const parts: React.ReactNode[] = [];
  const pattern = /\[E(\d+)\]/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let key = 0;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > last) {
      parts.push(<Markdown key={`t${key++}`} content={text.slice(last, match.index)} />);
    }
    const marker = Number(match[1]);
    parts.push(
      <button
        key={`c${key++}`}
        type="button"
        className={styles.citation}
        title={`Open the passage behind [E${marker}]`}
        onClick={() => onMarker(marker)}
      >
        E{marker}
      </button>,
    );
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    parts.push(<Markdown key={`t${key++}`} content={text.slice(last)} />);
  }
  return parts;
}

export const NotebookChat: React.FC<NotebookChatProps> = ({
  notebookId,
  notebookName,
  selectedSources,
  graphSelection,
  assertionSelection,
  onClearGraphSelection,
  onOpenCitation,
  onSaveAnswer,
  pendingPrompt,
  onPendingPromptConsumed,
}) => {
  const {
    conversation,
    isStreaming,
    streamingContents,
    send,
    openForNotebook,
    newNotebookConversation,
  } = useConversation();

  const [prompt, setPrompt] = useState('');
  const [preview, setPreview] = useState<ScopePreview | null>(null);
  const [previewError, setPreviewError] = useState<string | null>(null);
  const [manifests, setManifests] = useState<Map<string, EvidenceManifest | null>>(
    () => new Map(),
  );
  const [citationError, setCitationError] = useState<string | null>(null);
  const [opening, setOpening] = useState(false);
  const [ready, setReady] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  const nodeIds = useMemo(
    () => graphSelection.map((node) => node.id),
    [graphSelection],
  );
  const assertionIds = useMemo(
    () => assertionSelection.map((row) => row.id),
    [assertionSelection],
  );

  const scope = useMemo<ResearchScope>(
    () => ({
      notebookId,
      sourceSha256s: selectedSources,
      nodeIds,
      assertionIds,
    }),
    [notebookId, selectedSources, nodeIds, assertionIds],
  );

  // Open this notebook's own thread when the notebook changes. A notebook that
  // has never been asked anything gets a thread on its first question, not
  // before — an empty conversation per notebook visit would fill the sidebar.
  useEffect(() => {
    let live = true;
    setReady(false);
    setManifests(new Map());
    void openForNotebook(notebookId).finally(() => {
      if (live) setReady(true);
    });
    return () => {
      live = false;
    };
  }, [notebookId, openForNotebook]);

  // What the next question would use. Re-read whenever the scope moves.
  useEffect(() => {
    let live = true;
    setPreviewError(null);
    notebookResearchService
      .scopePreview(scope)
      .then((next) => {
        if (live) setPreview(next);
      })
      .catch((err) => {
        if (live) {
          setPreview(null);
          setPreviewError(String(err));
        }
      });
    return () => {
      live = false;
    };
  }, [scope]);

  const messages = useMemo(
    () => (conversation?.messages ?? []).filter((m) => m.role !== 'system'),
    [conversation],
  );

  useEffect(() => {
    bottom.current?.scrollIntoView({ block: 'end' });
  }, [messages.length, streamingContents]);

  // Manifests for the answers on screen, so a citation can be resolved.
  useEffect(() => {
    if (!conversation) return;
    let live = true;
    const wanted = messages
      .filter((m) => m.role === 'assistant' && m.status === 'done')
      .map((m) => m.id)
      .filter((id) => !manifests.has(id));
    if (wanted.length === 0) return;
    void Promise.all(
      wanted.map((id) =>
        notebookResearchService
          .turnEvidence(conversation.id, id)
          .then((manifest) => [id, manifest] as const)
          .catch(() => [id, null] as const),
      ),
    ).then((rows) => {
      if (!live) return;
      setManifests((current) => {
        const next = new Map(current);
        for (const [id, manifest] of rows) next.set(id, manifest);
        return next;
      });
    });
    return () => {
      live = false;
    };
  }, [conversation, messages, manifests]);

  const ask = useCallback(
    async (text: string) => {
      const question = text.trim();
      if (!question || isStreaming) return;
      setPrompt('');
      setCitationError(null);
      // A notebook without a thread gets one now, bound to it, so the answer
      // and everything after it belong to this notebook rather than to
      // whatever the general chat was last doing.
      if (!conversation) {
        await newNotebookConversation(notebookId, `${notebookName}: ${question}`);
      }
      await send(question, undefined, { research: scope });
    },
    [
      conversation,
      isStreaming,
      newNotebookConversation,
      notebookId,
      notebookName,
      scope,
      send,
    ],
  );

  useEffect(() => {
    if (!pendingPrompt || !ready) return;
    onPendingPromptConsumed();
    void ask(pendingPrompt);
  }, [pendingPrompt, ready, ask, onPendingPromptConsumed]);

  const openCitation = useCallback(
    async (messageId: string, marker: number) => {
      if (!conversation) return;
      setCitationError(null);
      setOpening(true);
      try {
        onOpenCitation(
          await notebookResearchService.openCitation(conversation.id, messageId, marker),
        );
      } catch (err) {
        setCitationError(String(err));
      } finally {
        setOpening(false);
      }
    },
    [conversation, onOpenCitation],
  );

  const narrowed = graphSelection.length > 0 || assertionSelection.length > 0;

  return (
    <section className={styles.chat} aria-label={`Ask ${notebookName}`}>
      <div className={styles.scopeBar}>
        {previewError ? (
          <span className={styles.scopeProblem}>
            <AlertTriangle size={12} /> {previewError}
          </span>
        ) : preview ? (
          <>
            <span className={styles.scopeChip}>
              {preview.sourcesSelected} of {preview.sourcesTotal}{' '}
              {preview.sourcesTotal === 1 ? 'source' : 'sources'}
            </span>
            <span className={styles.scopeChip} title={preview.retrievalExplanation}>
              {preview.retrievalMode} search
            </span>
            {preview.unreadable.length > 0 && (
              <span className={styles.scopeProblem}>
                <AlertTriangle size={12} /> {preview.unreadable.join(', ')} cannot be read
                and will contribute nothing
              </span>
            )}
            {!preview.graphBuilt && narrowed && (
              <span className={styles.scopeProblem}>
                <AlertTriangle size={12} /> no graph has been built for this notebook
              </span>
            )}
          </>
        ) : (
          <span className={styles.scopeChip}>checking scope…</span>
        )}
      </div>

      {narrowed && (
        <div className={styles.focusBar}>
          <span className={styles.sectionLabel}>Narrowed to</span>
          {graphSelection.map((node) => (
            <span key={node.id} className={styles.focusChip}>
              {node.label}
            </span>
          ))}
          {assertionSelection.map((row) => (
            <span key={row.id} className={styles.focusChip} data-relationship="true">
              {row.label}
            </span>
          ))}
          <button type="button" className={styles.linkAction} onClick={onClearGraphSelection}>
            <X size={11} /> clear
          </button>
        </div>
      )}

      <div className={styles.transcript}>
        {!ready && (
          <div className={styles.readerLoading}>
            <Spinner />
          </div>
        )}

        {ready && messages.length === 0 && (
          <div className={styles.empty}>
            <p>Nothing asked yet.</p>
            <p className={styles.emptyHint}>
              Ask a question and it will be answered from the sources ticked on the left,
              with every claim cited back to the page it came from.
            </p>
          </div>
        )}

        {messages.map((message) => {
          const streaming = streamingContents.get(message.id);
          const body =
            streaming !== undefined && streaming !== '' ? streaming : message.content;
          const manifest = manifests.get(message.id);
          if (message.role === 'user') {
            return (
              <div key={message.id} className={styles.userTurn}>
                {message.content}
              </div>
            );
          }
          return (
            <div key={message.id} className={styles.assistantTurn}>
              <div className={styles.answerBody}>
                {body
                  ? withCitations(body, (marker) => void openCitation(message.id, marker))
                  : message.status === 'streaming' && <Spinner />}
              </div>

              {message.error && (
                <p className={styles.error}>
                  <AlertTriangle size={13} /> {message.error}
                </p>
              )}

              {manifest && manifest.limitations.length > 0 && (
                <ul className={styles.limitations}>
                  {manifest.limitations.map((line) => (
                    <li key={line}>{line}</li>
                  ))}
                </ul>
              )}

              {manifest && manifest.sourcesUnused.length > 0 && (
                <p className={styles.limitationNote}>
                  Not used for this answer: {manifest.sourcesUnused.join(', ')}
                </p>
              )}

              {message.status === 'done' && body && (
                <div className={styles.answerActions}>
                  {manifest ? (
                    <span className={styles.answerMeta}>
                      {manifest.entries.length}{' '}
                      {manifest.entries.length === 1 ? 'passage' : 'passages'} ·{' '}
                      {manifest.retrievalMode} search
                    </span>
                  ) : (
                    <span className={styles.answerMeta}>no recorded evidence</span>
                  )}
                  <button
                    type="button"
                    className={styles.linkAction}
                    onClick={() =>
                      conversation &&
                      onSaveAnswer(
                        conversation.id,
                        message.id,
                        body.slice(0, 60).replace(/\s+/g, ' ').trim(),
                      )
                    }
                  >
                    <BookmarkPlus size={12} /> Save as note
                  </button>
                </div>
              )}
            </div>
          );
        })}
        <div ref={bottom} />
      </div>

      {citationError && (
        <p className={styles.error}>
          <AlertTriangle size={13} /> {citationError}
        </p>
      )}

      <form
        className={styles.composer}
        onSubmit={(event) => {
          event.preventDefault();
          void ask(prompt);
        }}
      >
        <textarea
          className={styles.composerInput}
          value={prompt}
          rows={2}
          placeholder={`Ask ${notebookName}…`}
          onChange={(event) => setPrompt(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter' && !event.shiftKey) {
              event.preventDefault();
              void ask(prompt);
            }
          }}
          disabled={isStreaming || !ready}
        />
        <Button
          type="submit"
          size="sm"
          icon
          disabled={isStreaming || !ready || !prompt.trim()}
          loading={isStreaming || opening}
          aria-label="Ask"
        >
          <Send size={14} />
        </Button>
      </form>
    </section>
  );
};
