import React, { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, MessagesSquare } from 'lucide-react';
import { Button } from '../ui';
import { handOffNotebook } from '../chat/notebookHandoff';
import styles from './workspace.module.css';

/**
 * The Notebooks page's hand-off to the main chat.
 *
 * This replaces the notebook's own chat thread and composer. There was a
 * second, complete chat implementation living here — its own transcript, its
 * own send path, its own citation rendering, its own scope bar — and keeping
 * it meant every improvement to asking questions had to be made twice, with
 * the two drifting until answers behaved differently depending on which screen
 * you asked from.
 *
 * The capabilities it had are not lost; they moved:
 *
 * - **Asking** is `/notebook` in the main composer.
 * - **Scope** is the notebook chip, which shows which sources are included and
 *   which of them can actually be read.
 * - **Citations** are `EvidenceCitations`, now rendered under every answer in
 *   main chat rather than only inside this page.
 * - **Notes, the graph and relationship review** stay here, because they are
 *   about the notebook rather than about a conversation.
 *
 * ## How the hand-off actually travels
 *
 * Not as an event. The chat surface is a route, and while this page is on
 * screen it is not mounted — so it has no listener, and an event dispatched
 * here would reach nothing. `handOffNotebook` leaves the notebook where the
 * surface will find it when it mounts; see `chat/notebookHandoff`. The
 * navigation is an ordinary router call.
 */
export interface AskInMainChatProps {
  notebookId: string;
  notebookName: string;
  /**
   * The sources currently ticked on this page, carried into the chip.
   *
   * Empty means the person has not narrowed the selection here, which hands
   * over as "all of this notebook" — stated explicitly on the receiving side
   * rather than inferred from an empty list.
   */
  selectedSources: string[];
  /** A question to place in the composer, when one has been composed here. */
  pendingPrompt: string | null;
  onPendingPromptConsumed: () => void;
}

export const AskInMainChat: React.FC<AskInMainChatProps> = ({
  notebookId,
  notebookName,
  selectedSources,
  pendingPrompt,
  onPendingPromptConsumed,
}) => {
  const navigate = useNavigate();

  const hand = useCallback(
    (prompt: string | null) => {
      handOffNotebook({
        notebookId,
        notebookName,
        sourceSha256s: selectedSources.length > 0 ? [...selectedSources] : null,
        prompt,
      });
      if (prompt) onPendingPromptConsumed();
      // The workbench is the chat. Navigating after the hand-off is written,
      // so the surface finds it the moment it mounts.
      navigate('/');
    },
    [notebookId, notebookName, selectedSources, onPendingPromptConsumed, navigate],
  );

  return (
    <section className={styles.askHandoff} aria-label={`Ask ${notebookName}`}>
      <MessagesSquare size={22} aria-hidden="true" />
      <h3 className={styles.askHandoffTitle}>Questions happen in the main chat</h3>
      <p className={styles.askHandoffBody}>
        Type <code>/notebook</code> in the composer and choose{' '}
        <strong>{notebookName}</strong>. The chip above the composer shows which
        sources the answer may use and which of them can be read, and every citation
        opens the passage behind it.
      </p>

      {pendingPrompt && (
        <div className={styles.askHandoffPending}>
          <p className={styles.askHandoffPrompt}>{pendingPrompt}</p>
          <Button onClick={() => hand(pendingPrompt)}>
            Ask this in the main chat <ArrowRight size={14} />
          </Button>
          {/* Dismissing leaves the selection alone. The prompt was composed
            * from a graph selection, and throwing that away because somebody
            * did not want to ask right now would be a surprise. */}
          <button
            type="button"
            className={styles.linkAction}
            onClick={onPendingPromptConsumed}
          >
            Discard the question
          </button>
        </div>
      )}

      {!pendingPrompt && (
        <Button onClick={() => hand(null)}>
          Open the chat with {notebookName} attached <ArrowRight size={14} />
        </Button>
      )}

      <p className={styles.askHandoffFootnote}>
        {selectedSources.length > 0
          ? `${selectedSources.length} ${
              selectedSources.length === 1 ? 'source is' : 'sources are'
            } ticked here, and the chat will start with that selection.`
          : 'No source is ticked here, so the chat will start with all of them selected.'}
      </p>
    </section>
  );
};
