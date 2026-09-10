import React from 'react';
import { AlertTriangle, ChevronLeft, ChevronRight, Loader2 } from 'lucide-react';

import styles from './ChatSurface.module.css';

/**
 * A produced PDF, drawn in the chat rather than described.
 *
 * ## The gap this closes
 *
 * `artifact_preview.rs` answers a PDF with its kind, its size, and no body at
 * all — deliberately, and it says why: there is no PDF reader in the Rust
 * process, and starting the Python sidecar to fill a preview pane would trade a
 * fast local read for the class of hang the rest of that work exists to remove.
 *
 * That was honest and still left the one lane that produces a *report* as the
 * one lane whose output could not be read without leaving the application. The
 * preview pane said "use the folder button to open it", which is a correct
 * sentence and not a preview.
 *
 * So the bytes come over IPC through `artifact_bytes` and are rendered here by
 * `pdf.js` — the same renderer Firefox ships. Nothing is fetched: the library is
 * bundled at build time, the worker is a same-origin asset, and the document is
 * a byte array that came from a file this application wrote.
 *
 * ## One page at a time, rather than a capped stack
 *
 * The obvious alternative is to render every page into its own canvas and let
 * the pane scroll. That needs a cap — a forty-page report is forty canvases at
 * screen resolution — and a cap means telling the reader that some of their
 * document is not being shown.
 *
 * Paging has no such edge. Every page is reachable, one canvas exists at a
 * time, and the control says which page of how many, so nothing about the
 * document's extent is hidden.
 */

/** The pdf.js module, loaded on first use and kept. */
let pdfjs: Promise<typeof import('pdfjs-dist')> | null = null;

function loadPdfjs() {
  if (!pdfjs) {
    pdfjs = import('pdfjs-dist').then(module => {
      /*
        pdf.js parses and rasterises on a worker so a large document cannot
        freeze the interface. The URL is resolved against this module so the
        bundler emits the worker as an asset of the application itself — which
        is also what keeps it inside `script-src 'self'`. A CDN URL, which is
        what the library's own documentation suggests, would be refused by the
        content security policy, and would be an egress this product does not
        permit in any case.
      */
      module.GlobalWorkerOptions.workerSrc = new URL(
        'pdfjs-dist/build/pdf.worker.min.mjs',
        import.meta.url,
      ).href;
      return module;
    });
  }
  return pdfjs;
}

/**
 * base64 to bytes.
 *
 * `atob` yields a string of char codes rather than bytes, so the copy into a
 * typed array is not avoidable. Done once per document, not per page.
 */
function decode(base64: string): Uint8Array {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

interface Loaded {
  document: import('pdfjs-dist').PDFDocumentProxy;
  pages: number;
}

type State =
  | { status: 'loading' }
  | { status: 'ready'; loaded: Loaded }
  | { status: 'failed'; reason: string };

export function PdfView({ base64, name }: { base64: string; name: string }) {
  const [state, setState] = React.useState<State>({ status: 'loading' });
  const [page, setPage] = React.useState(1);
  const canvasRef = React.useRef<HTMLCanvasElement | null>(null);
  const frameRef = React.useRef<HTMLDivElement | null>(null);

  /**
   * The width to lay the page out against, observed rather than measured once.
   *
   * Measuring `clientWidth` inside the render effect is the obvious approach
   * and it is wrong: the effect runs before the browser has laid the frame out,
   * so the first read comes back at or near zero. The page was drawn at 20×28
   * pixels — a legible-looking thumbnail of nothing, with no error anywhere,
   * because rendering a page into a tiny canvas succeeds.
   *
   * An observer answers the same question at a time when there is an answer,
   * and keeps answering it: the page is redrawn when the pane is resized, which
   * a single measurement could never do.
   */
  const [width, setWidth] = React.useState(0);

  React.useEffect(() => {
    const frame = frameRef.current;
    if (!frame) return;

    const observer = new ResizeObserver(entries => {
      const measured = entries[0]?.contentRect.width ?? 0;
      // Rounded, so a sub-pixel reflow does not re-rasterise every page.
      setWidth(previous => (Math.abs(previous - measured) > 1 ? measured : previous));
    });
    observer.observe(frame);
    return () => observer.disconnect();
  }, [state.status]);

  // Open the document.
  React.useEffect(() => {
    let live = true;
    // The *loading task* rather than the document: `destroy` lives there, and
    // it is also the only handle that can tear down a load still in flight.
    // `PDFDocumentProxy` offers `cleanup()`, which frees page resources and
    // leaves the worker's copy of the file in place.
    let task: import('pdfjs-dist').PDFDocumentLoadingTask | null = null;
    setState({ status: 'loading' });
    setPage(1);

    void (async () => {
      try {
        const module = await loadPdfjs();
        if (!live) return;

        const loading = module.getDocument({ data: decode(base64) });
        task = loading;
        const document = await loading.promise;
        if (!live) return;

        setState({ status: 'ready', loaded: { document, pages: document.numPages } });
      } catch (error) {
        if (!live) return;
        setState({
          status: 'failed',
          reason: error instanceof Error ? error.message : String(error),
        });
      }
    })();

    return () => {
      live = false;
      // Releases the worker's copy of the document. Without this, a message
      // carrying several PDFs holds every one of them open for the session.
      void task?.destroy();
    };
  }, [base64]);

  // Draw the current page.
  React.useEffect(() => {
    if (state.status !== 'ready') return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    // Nothing to lay out against yet. The observer above will run again with a
    // real width and bring this effect with it.
    if (width <= 0) return;

    let live = true;
    let task: { cancel: () => void } | null = null;

    void (async () => {
      try {
        const proxy = await state.loaded.document.getPage(page);
        if (!live) return;

        const context = canvas.getContext('2d');
        if (!context) return;

        /*
          Scaled to the pane's width, then multiplied by the device pixel ratio
          and scaled back down in CSS. A canvas drawn at CSS pixels on a
          high-density screen renders a report's body text soft enough to be
          unpleasant to read, which for a document viewer is the whole job.

          The ratio is capped at 2: past that the memory cost climbs with no
          visible gain.
        */
        const unscaled = proxy.getViewport({ scale: 1 });
        const scale = width / unscaled.width;
        const ratio = Math.min(window.devicePixelRatio || 1, 2);
        const viewport = proxy.getViewport({ scale: scale * ratio });

        canvas.width = Math.floor(viewport.width);
        canvas.height = Math.floor(viewport.height);
        canvas.style.width = `${Math.floor(viewport.width / ratio)}px`;
        canvas.style.height = `${Math.floor(viewport.height / ratio)}px`;

        const render = proxy.render({ canvas, canvasContext: context, viewport });
        task = render;
        await render.promise;
      } catch (error) {
        // A cancelled render is the expected outcome of paging quickly, not a
        // failure worth showing anybody.
        if (!live) return;
        const message = error instanceof Error ? error.message : String(error);
        if (/cancel/i.test(message)) return;
        setState({ status: 'failed', reason: message });
      }
    })();

    return () => {
      live = false;
      task?.cancel();
    };
  }, [state, page, width]);

  if (state.status === 'loading') {
    return (
      <div className={styles.previewPane} aria-busy="true">
        <Loader2 size={12} className={styles.spin} />
        <span>Opening {name}…</span>
      </div>
    );
  }

  if (state.status === 'failed') {
    return (
      <div className={styles.previewPane}>
        <AlertTriangle size={12} />
        <span>
          {name} could not be opened here: {state.reason}. Use Save as… to write it
          somewhere and open it there.
        </span>
      </div>
    );
  }

  const { pages } = state.loaded;

  return (
    <div className={styles.pdfView}>
      <div className={styles.pdfFrame} ref={frameRef}>
        <canvas
          ref={canvasRef}
          className={styles.pdfCanvas}
          aria-label={`${name}, page ${page} of ${pages}`}
        />
      </div>
      {/*
        Shown even for a one-page document. The count is the honest statement of
        how much document there is, and hiding it on a single page would leave a
        reader unsure whether there was more below.
      */}
      <div className={styles.pdfPager}>
        <button
          type="button"
          className={styles.iconBtn}
          onClick={() => setPage(current => Math.max(1, current - 1))}
          disabled={page <= 1}
          aria-label="Previous page"
          title="Previous page"
        >
          <ChevronLeft size={12} />
        </button>
        <span className={styles.pdfPageCount}>
          Page {page} of {pages}
        </span>
        <button
          type="button"
          className={styles.iconBtn}
          onClick={() => setPage(current => Math.min(pages, current + 1))}
          disabled={page >= pages}
          aria-label="Next page"
          title="Next page"
        >
          <ChevronRight size={12} />
        </button>
      </div>
    </div>
  );
}
