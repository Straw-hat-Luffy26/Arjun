import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, ArrowUp, Plus, ScanText, Square, X } from 'lucide-react';
import { sovereigntyService } from '../../services/sovereignty.service';
import { toComposerAttachment, type ComposerAttachment } from '../../services/agent.service';
import { agentService } from '../../services/agent.service';
import {
  previewAttachmentRouting,
  type AttachmentPlan,
} from '../../services/ocr.service';
import { RoutingPreview } from '../routing/RoutingPreview';
import { OcrQualitySlider } from './OcrQualitySlider';
import type { OcrPreference } from './useOcrPreference';
import { ContextChip } from './ContextChip';
import { NotebookChip, NotebookChooser } from './NotebookScopeControls';
import {
  caretAfterConsume,
  consumeCommand,
  findNotebookCommand,
  moveHighlight,
  type CommandSpan,
  type NotebookScope,
} from './notebookScope';
import { useConversation } from '../run/useConversation';
import styles from './ChatSurface.module.css';

/**
 * The bottom-of-chat composer.
 *
 * Two visible layers:
 *  1. **Routing hint** — a single quiet line above the composer that
 *     shows which model will take the next message. This was previously
 *     below the composer; the redesign lifts it up so the user can see
 *     the model before they press Enter.
 *  2. **Composer** — the input row. Same Enter-to-send and auto-grow
 *     behaviour as before; the right side now has a context chip and a
 *     single send/stop button (the send button morphs into a stop
 *     button while a run is streaming).
 *
 * The composer is always present in a chat surface — the user must be
 * able to send a follow-up after every completed response, including
 * during a run that is still streaming. While a run is in flight, the
 * input itself is *not* disabled (the user can type a follow-up that
 * will queue once the run finishes); only the send button morphs into
 * a stop.
 */
const MAX_COMPOSER_HEIGHT = 220;

/**
 * How long a Stop waits to be acknowledged before it says it cannot be.
 *
 * The same fifteen seconds the run inspector allows, and for the same reason:
 * a turn stopped mid-tool finishes the tool first, and that is legitimately
 * slow. Past this the button gives up waiting and says so, rather than sitting
 * disabled for ever on a turn that never reported an ending.
 */
const STOP_ACKNOWLEDGEMENT_TIMEOUT_MS = 15_000;

export interface ChatComposerProps {
  /** True while a run is in flight; the send button becomes stop. */
  streaming?: boolean;
  /** When streaming, the active run's id (used to call abort). */
  activeRunId?: string | null;
  /** Optional placeholder override. */
  placeholder?: string;
  /** Messages already typed and waiting for the current run to finish. */
  queued?: string[];
  /** Drop a queued message before it is sent. */
  onCancelQueued?: (index: number) => void;
  onSubmit: (prompt: string, attachments: ComposerAttachment[]) => Promise<void> | void;
  /**
   * The accuracy-to-speed setting for reading attachments.
   *
   * Owned by the surface because the turn is sent from there, rendered here
   * because this is where the person is looking when they attach a file.
   */
  ocrPreference?: OcrPreference;
  /**
   * The notebook the next question is scoped to, if any.
   *
   * Owned by the surface rather than here, because the surface is what freezes
   * a turn's scope at submit time and what restores the preference when a
   * conversation is reopened. The composer only offers the controls.
   */
  notebookScope?: NotebookScope | null;
  onNotebookScopeChange?: (scope: NotebookScope | null) => void;
  /** Opens the Notebooks screen, for anything the chip cannot do inline. */
  onOpenNotebook?: (notebookId: string) => void;
  /**
   * A question composed elsewhere and handed to this composer.
   *
   * Placed in the draft and never sent. Somebody who asked the Notebooks page
   * to look at a graph selection gets the question in front of them to read
   * and edit, which is the same promise `/notebook` makes about their own
   * half-written draft.
   */
  handedPrompt?: string | null;
  onHandedPromptConsumed?: () => void;
}

export function ChatComposer({
  streaming,
  activeRunId,
  placeholder,
  queued = [],
  onCancelQueued,
  onSubmit,
  ocrPreference,
  notebookScope = null,
  onNotebookScopeChange,
  onOpenNotebook,
  handedPrompt = null,
  onHandedPromptConsumed,
}: ChatComposerProps) {
  const { conversation } = useConversation();
  const [prompt, setPrompt] = useState('');
  const [attachments, setAttachments] = useState<ComposerAttachment[]>([]);
  // What the backend says it will do with the attached files. Asked for
  // rather than worked out here: the composer guessing at the routing is how
  // a hint that says "OCR" ends up above a run that used none.
  const [plans, setPlans] = useState<AttachmentPlan[]>([]);
  // Whether the routing preview has answered for the current attachments.
  // An empty `plans` means "nothing needs OCR" only once this is true;
  // before that it means "not asked yet", and the two decide opposite
  // things about whether the reading control belongs on screen.
  const [plansResolved, setPlansResolved] = useState(false);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [stopping, setStopping] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // ── The `/notebook` chooser ──────────────────────────────────────────
  //
  // `command` is where the command sits in the draft, recomputed from the
  // text and the caret rather than from a flag. A boolean "is the chooser
  // open" would drift: deleting the slash, moving the caret away or pressing
  // Enter each have to close it, and each would be a separate place to
  // remember to clear the flag.
  //
  // `dismissed` is the one piece of state that cannot be derived — Escape
  // closes the chooser while the command text is still in the draft, and it
  // stays closed until the command is edited again.
  const [command, setCommand] = useState<CommandSpan | null>(null);
  const [dismissed, setDismissed] = useState<string | null>(null);
  const [highlight, setHighlight] = useState(0);
  const [choiceCount, setChoiceCount] = useState(0);
  // Set by the chooser, so Enter resolves against the list actually on screen
  // rather than against a second copy of the filtering rules kept here.
  const chooseRef = useRef<((index: number) => void) | null>(null);
  // Stable, because it is a dependency of the chooser's registration effect.
  // An inline literal here was a new function on every render, so the effect
  // re-fired on every keystroke -- harmless, and exactly the churn registering
  // through a ref was meant to avoid.
  const registerChooser = useCallback((choose: (index: number) => void) => {
    chooseRef.current = choose;
  }, []);

  /** Recomputes the command span from the live text and caret. */
  const syncCommand = useCallback((text: string, caret: number) => {
    const span = findNotebookCommand(text, caret);
    setCommand(span);
    setHighlight(0);
    // Editing the command after dismissing it reopens the chooser; leaving it
    // exactly as it was does not.
    setDismissed(previous => {
      const token = span ? text.slice(span.start, span.end) : null;
      return previous !== null && previous === token ? previous : null;
    });
  }, []);

  const commandToken = command ? prompt.slice(command.start, command.end) : null;
  const chooserOpen = command !== null && dismissed !== commandToken;

  const chooseNotebook = useCallback(
    (scope: NotebookScope) => {
      if (!command) return;
      // The command token is consumed rather than sent. The rest of the draft
      // — the question somebody was half-way through — is left exactly as it
      // was, and the caret goes back where the command started.
      const next = consumeCommand(prompt, command);
      setPrompt(next);
      setCommand(null);
      setDismissed(null);
      onNotebookScopeChange?.(scope);
      requestAnimationFrame(() => {
        const el = textareaRef.current;
        if (!el) return;
        el.focus();
        const caret = Math.min(caretAfterConsume(command), next.length);
        el.setSelectionRange(caret, caret);
      });
    },
    [command, prompt, onNotebookScopeChange],
  );

  const resize = useCallback(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = 'auto';
    el.style.height = `${Math.min(el.scrollHeight, MAX_COMPOSER_HEIGHT)}px`;
  }, []);

  useEffect(() => {
    resize();
  }, [prompt, resize]);

  // Keep the textarea at a sensible height when the chat's content
  // changes around it (e.g. a new assistant message that pushes the
  // composer down).
  useEffect(() => {
    const t = window.setTimeout(resize, 0);
    return () => window.clearTimeout(t);
  }, [conversation?.messages.length, resize]);

  useEffect(() => {
    if (attachments.length === 0) {
      setPlans([]);
      setPlansResolved(false);
      return;
    }
    let live = true;
    setPlansResolved(false);
    void previewAttachmentRouting(
      attachments.map(a => ({ name: a.name, mime: a.mime })),
    )
      .then(next => {
        if (live) {
          setPlans(next);
          setPlansResolved(true);
        }
      })
      .catch(() => {
        // No hint is better than a wrong one. The run itself makes the same
        // decision from the same code, so nothing is lost but the preview.
        //
        // Deliberately left unresolved rather than resolved-and-empty: the two
        // are the same empty list and very different facts, and treating a
        // failed preview as "nothing needs reading" would hide the reading
        // control on exactly the files whose routing could not be worked out.
        if (live) setPlans([]);
      });
    return () => {
      live = false;
    };
  }, [attachments]);

  // A handed-over question joins whatever is already drafted rather than
  // replacing it. Overwriting somebody's half-written sentence with a
  // generated one would be the same mistake `/notebook` was careful not to
  // make at the other end.
  useEffect(() => {
    if (!handedPrompt) return;
    setPrompt(current => (current.trim() ? `${current.trim()}

${handedPrompt}` : handedPrompt));
    onHandedPromptConsumed?.();
    requestAnimationFrame(() => textareaRef.current?.focus());
  }, [handedPrompt, onHandedPromptConsumed]);

  const hasContent = prompt.trim().length > 0 || attachments.length > 0;
  const canSubmit = hasContent;
  const ocrPlans = plans.filter(p => p.needsOcr);

  // Whether the accuracy-to-speed control is shown at all.
  //
  // It used to be permanent, on the reasoning that a camera's quality setting
  // exists before the shot. In a chat composer that reads as a control with
  // nothing to control: it sat under every message in a conversation that had
  // no attachment in it, and moving it appeared to do nothing because there was
  // nothing for it to do.
  //
  // So it appears with the file it governs. The `attachments.length > 0` half
  // matters when the routing preview has not answered yet — a control that
  // blinked in a beat after the file did would be worse than one that arrives
  // with it — and `plansResolved` keeps it from lingering once the preview
  // comes back saying nothing here needs reading, which is the answer for a
  // .txt or a .csv.
  const showOcrSlider =
    attachments.length > 0 && (!plansResolved || ocrPlans.length > 0);

  const submit = useCallback(async () => {
    if (!canSubmit) return;
    const text = prompt.trim();
    const atts = [...attachments];
    setPrompt('');
    setAttachments([]);
    setPlans([]);
    setRefusal(null);
    try {
      await onSubmit(text, atts);
    } catch (error) {
      setRefusal(error instanceof Error ? error.message : String(error));
    }
  }, [canSubmit, prompt, attachments, onSubmit]);

  /**
   * Ask the turn to stop, and wait for it to actually stop.
   *
   * ## Requested is not terminated
   *
   * `agent_abort_run` resolving means the request reached something, not that
   * the turn has ended. A turn stopped mid-tool finishes the tool first; a
   * turn stopped during OCR unwinds through several awaits. Clearing the
   * button the instant the call returned therefore reported a stop while the
   * machine was still working — the same lie, one layer up, as an abort that
   * never reached anything.
   *
   * So the button stays in its stopping state until the turn's *own* state
   * says it is over. The first effect below is that acknowledgement; the
   * second bounds the wait so it cannot sit there for ever.
   */
  const stop = useCallback(async () => {
    if (!activeRunId || stopping) return;
    setStopping(true);
    setRefusal(null);
    try {
      const outcome = await agentService.abort(activeRunId);
      if (!outcome.requested) {
        // The id named nothing: the turn ended between the render and the
        // click. An ordinary race — nothing left to wait for, and nothing
        // worth reporting.
        setStopping(false);
      }
    } catch (err) {
      setStopping(false);
      setRefusal(
        `The stop could not be sent: ${
          err instanceof Error ? err.message : String(err)
        }. The run is still going.`,
      );
    }
  }, [activeRunId, stopping]);

  // The acknowledgement: the turn itself reporting that it is over.
  //
  // `streaming` goes false when the run reaches a terminal state, which is the
  // only thing that can honestly clear a Stop. Watched rather than assumed.
  useEffect(() => {
    if (!streaming) setStopping(false);
  }, [streaming]);

  // A stop that is never acknowledged is reported, not waited on for ever.
  //
  // Otherwise the button sits disabled reading 'Stopping' with no way back,
  // which tells a person less than saying plainly that the request went out
  // and nothing came back.
  useEffect(() => {
    if (!stopping) return;
    const timer = window.setTimeout(() => {
      setStopping(false);
      setRefusal(
        'Stop was sent, but the turn has not reported that it ended. It may ' +
          'still be finishing the step it was on. Nothing here can confirm it ' +
          'has stopped.',
      );
    }, STOP_ACKNOWLEDGEMENT_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [stopping]);

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // The chooser takes the navigation keys while it is open, and nothing
    // else. Typing continues to reach the textarea underneath, so the search
    // is just the draft — no second input, no focus change, no lost caret.
    if (chooserOpen) {
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
        e.preventDefault();
        setHighlight(current =>
          moveHighlight(current, e.key === 'ArrowDown' ? 1 : -1, choiceCount),
        );
        return;
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        // Dismissed, with the draft untouched. Escape closes a chooser; it
        // does not delete what somebody typed.
        setDismissed(commandToken);
        return;
      }
      if (e.key === 'Enter' && !e.shiftKey && command?.complete && choiceCount > 0) {
        // Selecting must not send. This is the key that would otherwise have
        // submitted a message whose only content was `/notebook`.
        e.preventDefault();
        chooseRef.current?.(highlight);
        return;
      }
      if (e.key === 'Tab' && command?.complete && choiceCount > 0) {
        e.preventDefault();
        chooseRef.current?.(highlight);
        return;
      }
    }

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      // Mid-run this queues instead of starting a second run; the
      // surface sends it when the one in flight finishes.
      void submit();
    }
  };

  const onFilesPicked = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? []);
    e.target.value = '';
    if (!files.length) return;
    try {
      await sovereigntyService.assertConfidentialAllowed('attaching a document');
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
      return;
    }
    setRefusal(null);
    // Read here, at the moment of picking. The File handle is only valid
    // while the input holds it, and the backend cannot open a path the
    // webview names — so the bytes have to be carried, and this is the only
    // place they exist.
    try {
      const read = await Promise.all(files.map(toComposerAttachment));
      setAttachments(prev => [...prev, ...read]);
    } catch (err) {
      setRefusal(err instanceof Error ? err.message : String(err));
    }
  };

  return (
    <div className={styles.composerOuter}>
      <div className={styles.composerHintRow}>
        <RoutingPreview prompt={prompt} />
      </div>

      {/* What the attached files will be routed to, before anything is sent.
        * The reasoning model named above answers the question; these lines
        * name the model that has to read the page first. Showing only the
        * second one is what made an attached scan look like it was being
        * handled by a text model that had never seen it. */}
      {plans.length > 0 && (
        <ul className={styles.attachmentPlans}>
          {plans.map((plan, i) => (
            <li
              key={`${plan.name}-${i}`}
              className={styles.attachmentPlan}
              data-route={plan.route}
            >
              {plan.refusal ? (
                <AlertTriangle size={13} aria-hidden="true" />
              ) : (
                <ScanText size={13} aria-hidden="true" />
              )}
              <span>{plan.explanation}</span>
            </li>
          ))}
        </ul>
      )}

      {chooserOpen && (
        <NotebookChooser
          query={command?.query ?? ''}
          highlight={highlight}
          onHighlightChange={setHighlight}
          onCountChange={setChoiceCount}
          onRegisterChooser={registerChooser}
          onChoose={chooseNotebook}
          onDismiss={() => setDismissed(commandToken)}
        />
      )}

      {notebookScope && (
        <NotebookChip
          scope={notebookScope}
          onChange={scope => onNotebookScopeChange?.(scope)}
          onClear={() => onNotebookScopeChange?.(null)}
          onOpenNotebook={id => onOpenNotebook?.(id)}
        />
      )}

      <div className={styles.composerWrap}>
        <div className={styles.composer} data-streaming={streaming || undefined}>
          {queued.length > 0 && (
            <ul className={styles.queuedList}>
              {queued.map((text, i) => (
                <li key={`${i}-${text.slice(0, 24)}`} className={styles.queuedChip}>
                  <span className={styles.queuedBadge}>Queued</span>
                  <span className={styles.queuedText}>{text}</span>
                  {onCancelQueued && (
                    <button
                      type="button"
                      className={styles.attachmentRemove}
                      aria-label={`Remove queued message ${i + 1}`}
                      onClick={() => onCancelQueued(i)}
                    >
                      <X size={12} />
                    </button>
                  )}
                </li>
              ))}
            </ul>
          )}

          {attachments.length > 0 && (
            <ul className={styles.attachmentList}>
              {attachments.map((att, i) => (
                <li key={`${att.name}-${i}`} className={styles.attachmentChip}>
                  <span className={styles.attachmentName}>{att.name}</span>
                  <button
                    type="button"
                    className={styles.attachmentRemove}
                    aria-label={`Remove ${att.name}`}
                    onClick={() =>
                      setAttachments(prev => prev.filter((_, j) => j !== i))
                    }
                  >
                    <X size={12} />
                  </button>
                </li>
              ))}
            </ul>
          )}

          <textarea
            ref={textareaRef}
            className={styles.composerInput}
            placeholder={
              placeholder ??
              (streaming
                ? 'Keep asking, messages will be queued…'
                : 'Ask Arjun — text, images, tables')
            }
            value={prompt}
            rows={1}
            onChange={e => {
              setPrompt(e.target.value);
              syncCommand(e.target.value, e.target.selectionStart ?? e.target.value.length);
            }}
            onKeyDown={onKeyDown}
            // Clicking or arrowing away from the command closes the chooser,
            // which is why the span is recomputed from the caret here too.
            onSelect={e => {
              const el = e.currentTarget;
              syncCommand(el.value, el.selectionStart ?? el.value.length);
            }}
            onBlur={() => setCommand(null)}
            aria-label="Message"
          />

          <div className={styles.composerControls}>
            <button
              type="button"
              className={styles.iconBtn}
              onClick={() => fileInputRef.current?.click()}
              aria-label="Attach a document, drawing or photograph"
            >
              <Plus size={18} />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept=".png,.jpg,.jpeg,.webp,.pdf,.txt,.md,.markdown,.csv,.json,.log,.tsv,.docx,.xlsx,.pptx"
              multiple
              hidden
              onChange={onFilesPicked}
            />
            <div className={styles.composerRight}>
              <ContextChip />
              {/* Stop is shown for the whole of a run, draft or no draft.
                *
                * It used to be `streaming && !hasContent`, so typing a
                * follow-up *replaced* Stop with Send — and typing a follow-up
                * is exactly what somebody does while waiting for a turn they
                * are about to give up on. The only way back to Stop was to
                * delete what they had written.
                *
                * Both are shown now: Stop ends the turn in flight, Send
                * queues the draft for when it is over. Two buttons for two
                * actions, rather than one button that silently changes which
                * of them it is. */}
              {streaming && (
                <button
                  type="button"
                  className={styles.stopBtn}
                  onClick={() => void stop()}
                  disabled={stopping}
                  aria-label={stopping ? 'Stopping' : 'Stop generating'}
                  title={stopping ? 'Stopping…' : 'Stop generating'}
                >
                  <Square size={12} />
                  <span>{stopping ? 'Stopping…' : 'Stop'}</span>
                </button>
              )}
              {(!streaming || hasContent) && (
                <button
                  type="button"
                  className={styles.sendBtn}
                  onClick={() => void submit()}
                  disabled={!canSubmit}
                  aria-label={streaming ? 'Queue this message' : 'Send'}
                  title={streaming ? 'Queue this message (Enter)' : 'Send (Enter)'}
                >
                  <ArrowUp size={16} />
                </button>
              )}
            </div>
          </div>

          {refusal && (
            <p className={styles.refusalLine} role="alert">
              <span>{refusal}</span>
            </p>
          )}
        </div>

        {ocrPreference && showOcrSlider && (
          <OcrQualitySlider
            preference={ocrPreference}
            disabled={streaming}
            engaged={ocrPlans.length > 0}
          />
        )}
      </div>
    </div>
  );
}
