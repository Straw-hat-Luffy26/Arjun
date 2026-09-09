import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Loader2 } from 'lucide-react';

import { getBackendService } from '../../services/api';
import {
  HOST_SOURCE,
  MAX_WIDGET_HEIGHT,
  WIDGET_THEME_TOKENS,
  buildWidgetDocument,
  readWidgetMessage,
  themeBlock,
  widgetUrl,
  type WidgetTheme,
} from './widgetDocument';
import styles from './ChatSurface.module.css';

/**
 * A `widget` fence: HTML, CSS and JS the model wrote, running in the message.
 *
 * ## Why it is a fence of its own rather than `html`
 *
 * A ```html fence is source a reader asked to see. Running it because it
 * happens to be HTML would take away the only way to be shown markup without
 * it executing, and would run code the model never meant to run. `widget` is
 * the model saying "this is meant to be alive", which is a different statement.
 *
 * ## The two failure modes this is built around
 *
 * The build spec separates generation hanging from generation succeeding and
 * never appearing, and asks for both to be handled here.
 *
 * *Hanging*: a frame that never loads is given `LOAD_TIMEOUT_MS` and then shown
 * as failed. The timer is cleared by the frame's own `ready` message and not by
 * `onLoad`, which fires for a document that then throws before drawing
 * anything — the difference between "the browser fetched it" and "the reader
 * can see it".
 *
 * *Never appearing*: the frame is not shown until it says it is ready, and
 * every stage between the fence arriving and the widget drawing is named on
 * screen. If it stops, it stops somewhere with a name.
 */

/**
 * How long a widget gets to load before it is called failed.
 *
 * The spec asks for about eight seconds. Nothing here touches the network or
 * the disk — the document is already in memory, served from a local scheme — so
 * a widget that has not drawn in eight seconds is not slow, it is broken.
 */
const LOAD_TIMEOUT_MS = 8000;

/**
 * How long to wait after the fence stops changing before building anything.
 *
 * A widget arrives a character at a time while the model writes it. Preparing
 * on every keystroke would build hundreds of documents and show a flickering
 * half-written page; waiting for quiet costs a beat and builds one.
 */
const SETTLE_MS = 250;

/** Where a widget is between arriving and being on screen. */
type Stage = 'writing' | 'preparing' | 'loading' | 'ready' | 'failed';

const STAGE_LABEL: Record<Stage, string> = {
  writing: 'Writing the widget…',
  preparing: 'Preparing the sandbox…',
  loading: 'Loading…',
  ready: '',
  failed: '',
};

/** Reads the live values of the tokens a widget is allowed to see. */
function readTheme(): WidgetTheme {
  if (typeof window === 'undefined') return {};
  const computed = window.getComputedStyle(document.documentElement);
  const theme: WidgetTheme = {};
  for (const token of WIDGET_THEME_TOKENS) {
    const value = computed.getPropertyValue(token);
    if (value) theme[token] = value.trim();
  }
  return theme;
}

export function WidgetFrame({
  html,
  complete,
  onPrompt,
}: {
  html: string;
  /** False while the fence is still being written. */
  complete: boolean;
  /** Raised when something inside the widget calls `sendPrompt`. */
  onPrompt?: (text: string) => void;
}) {
  const frame = useRef<HTMLIFrameElement | null>(null);
  const [stage, setStage] = useState<Stage>('writing');
  const [problem, setProblem] = useState<string | null>(null);
  const [url, setUrl] = useState<string | null>(null);
  const [height, setHeight] = useState(120);
  const [attempt, setAttempt] = useState(0);

  // Prepare once the fence has closed and stopped changing.
  useEffect(() => {
    if (!complete) {
      setStage('writing');
      return;
    }

    let live = true;
    const timer = window.setTimeout(() => {
      setStage('preparing');
      setProblem(null);

      getBackendService()
        .invoke<string>('widget_prepare', { document: buildWidgetDocument(html, readTheme()) })
        .then(id => {
          if (!live) return;
          setUrl(widgetUrl(id, navigator.userAgent));
          setStage('loading');
        })
        .catch((error: unknown) => {
          if (!live) return;
          // Named, not swallowed. A widget that could not be prepared is a
          // widget the reader should be told about.
          setProblem(error instanceof Error ? error.message : String(error));
          setStage('failed');
        });
    }, SETTLE_MS);

    return () => {
      live = false;
      window.clearTimeout(timer);
    };
  }, [html, complete, attempt]);

  // The hard timeout. Cleared by the frame's own `ready`, never by `onLoad`.
  useEffect(() => {
    if (stage !== 'loading') return;
    const timer = window.setTimeout(() => {
      setProblem('the widget did not finish loading');
      setStage('failed');
    }, LOAD_TIMEOUT_MS);
    return () => window.clearTimeout(timer);
  }, [stage, url]);

  // What the widget says back.
  useEffect(() => {
    const listen = (event: MessageEvent) => {
      // The frame runs at an opaque origin, which reports itself as "null", so
      // the origin cannot identify it. The source can, and this is the check
      // that keeps every other frame's messages out.
      if (!frame.current || event.source !== frame.current.contentWindow) return;

      const message = readWidgetMessage(event.data);
      if (!message) return;

      if (message.type === 'ready') {
        setStage('ready');
        return;
      }
      if (message.type === 'height') {
        setHeight(message.height);
        return;
      }
      if (message.type === 'prompt') {
        onPrompt?.(message.text);
      }
    };

    window.addEventListener('message', listen);
    return () => window.removeEventListener('message', listen);
  }, [onPrompt]);

  // Follow the application's theme after loading, so switching light and dark
  // re-themes widgets already on screen rather than leaving them in the colours
  // they were built with.
  useEffect(() => {
    if (stage !== 'ready') return;
    frame.current?.contentWindow?.postMessage(
      { source: HOST_SOURCE, type: 'theme', css: themeBlock(readTheme()) },
      '*',
    );
  }, [stage]);

  const retry = useCallback(() => {
    setUrl(null);
    setProblem(null);
    setStage('writing');
    setAttempt(n => n + 1);
  }, []);

  if (stage === 'failed') {
    return (
      <div className={styles.mdWidget}>
        <p className={styles.mdWidgetError} role="alert">
          <AlertTriangle size={12} />
          {/* The type, the reason, and a way to try again — never a blank box. */}
          <span>Interactive widget — {problem ?? 'it could not be displayed'}.</span>
          <button type="button" className={styles.mdWidgetRetry} onClick={retry}>
            Try again
          </button>
        </p>
      </div>
    );
  }

  return (
    <div className={styles.mdWidget}>
      {stage !== 'ready' && (
        <p className={styles.mdWidgetStage} aria-busy="true">
          <Loader2 size={12} className={styles.spin} />
          <span>{STAGE_LABEL[stage]}</span>
        </p>
      )}
      {url && (
        <iframe
          ref={frame}
          src={url}
          title="Interactive widget"
          /*
           * `allow-scripts` and nothing else. Without `allow-same-origin` the
           * document runs at an opaque origin and cannot reach this one;
           * without `allow-popups`, `allow-forms` or `allow-top-navigation` it
           * cannot open, submit or steer anything. The served policy
           * (`default-src 'none'`) is the other half of the boundary.
           */
          sandbox="allow-scripts"
          className={styles.mdWidgetFrame}
          style={{
            height: `${Math.min(height, MAX_WIDGET_HEIGHT)}px`,
            // Hidden rather than unmounted while loading: unmounting would
            // restart the load, and the timeout with it.
            visibility: stage === 'ready' ? 'visible' : 'hidden',
            position: stage === 'ready' ? 'static' : 'absolute',
          }}
        />
      )}
    </div>
  );
}
