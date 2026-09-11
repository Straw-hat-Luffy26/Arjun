import { invoke } from '@tauri-apps/api/core';
import { listen, Event } from '@tauri-apps/api/event';

/**
 * The document-scan stream.
 *
 * The OCR model does not return a page and then its layout — it interleaves
 * them, committing to one region at a time. The backend parses that into
 * discrete events (see `ai_engine::ocr_spans`) and emits them here, which is
 * what lets the scan view draw boxes as the model finds them rather than
 * showing a spinner and then a finished page.
 */

/** A box exactly as the model wrote it, in whatever space the build uses. */
export interface RawBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

/** A box mapped onto the displayed page. */
export interface PageBox extends RawBox {
  /**
   * False when the box fell outside the page.
   *
   * Reported rather than clamped: an out-of-page box means the coordinate
   * space is wrong, and quietly pulling it inside would hide the evidence.
   * The scan view draws these in the error colour instead of dropping them.
   */
  inBounds: boolean;
}

export interface OcrRegionEvent {
  event: 'region';
  index: number;
  /** `title`, `text`, `table`, `figure` — the model's own label. */
  label: string;
  bbox: RawBox | null;
  /**
   * Null until the build's coordinate convention has been calibrated.
   *
   * The same numbers mean different things depending on whether the model
   * reports normalised 0-999 or input pixels, and the two are only
   * distinguishable by running one page at two input sizes. Until that has
   * been done the backend sends null, and the scan view shows the text
   * without the overlay rather than drawing boxes that are plausibly placed
   * and actually wrong.
   */
  pageBox: PageBox | null;
}

export interface OcrTextEvent {
  event: 'text';
  /** Null for an ungrounded reply, where the whole response is prose. */
  index: number | null;
  delta: string;
}

export type OcrEvent = OcrRegionEvent | OcrTextEvent;

export type OcrState = 'reading' | 'done' | 'failed';

export interface OcrStatusPayload {
  page: number;
  state: OcrState;
  tokens: number;
  elapsedMs: number;
  /** Present only once measured; never estimated. */
  tokensPerSecond: number | null;
}

export interface OcrErrorPayload {
  page: number;
  reason: string;
}

/** The four slider stops, fastest first. Mirrors `ai_engine::ocr_profile`. */
export type OcrDetent = 'fastest' | 'fast' | 'detailed' | 'maximum';

export interface OcrDetentInfo {
  detent: OcrDetent;
  label: string;
  tier: 'high' | 'fast';
  tierLabel: string;
  /**
   * The DeepEncoder resolution mode this stop was designed around.
   *
   * Carried because the backend reports it and the number is the right one,
   * but **not shown**: it is a `llama-server` launch argument that nothing
   * currently passes, and the server is reused across stops within a tier, so
   * changing the slider cannot change it. See the header of
   * `ai_engine/ocr_profile.rs`. What a stop actually changes is the weight file
   * and `maxDecodeTokens`, and those are what the UI reports.
   */
  maxImageTokens: number;
  maxDecodeTokens: number;
}

/** The stops in slider order, fastest first. Mirrors `OcrDetent::ALL`. */
export const OCR_DETENT_ORDER: OcrDetent[] = [
  'fastest',
  'fast',
  'detailed',
  'maximum',
];

/**
 * The stops and what each costs, read from the backend rather than duplicated
 * here — a slider whose labels disagree with the profiles that actually run
 * is worse than no slider.
 */
export async function getOcrDetents(): Promise<OcrDetentInfo[]> {
  return invoke<OcrDetentInfo[]>('get_ocr_detents');
}

/**
 * What will happen to a file, decided before the turn is sent.
 *
 * Mirrors `commands::ocr::AttachmentPlan`. The composer shows this so that
 * "an OCR model will read this" is visible while the person is still typing,
 * rather than being discovered afterwards from a progress line.
 */
export interface AttachmentPlan {
  name: string;
  /** `image` | `document` | `rejected`. */
  route: 'image' | 'document' | 'rejected';
  /** True when a vision model has to look at the page. */
  needsOcr: boolean;
  /** Why, in the person's terms. Show verbatim; do not re-word. */
  explanation: string;
  /** Set when the file cannot be read at all. */
  refusal: string | null;
}

/** Asks the backend what it would do with these files. */
export async function previewAttachmentRouting(
  files: { name: string; mime: string }[]
): Promise<AttachmentPlan[]> {
  return invoke<AttachmentPlan[]>('preview_attachment_routing', { files });
}

/**
 * What the OCR model is doing to a chat attachment right now.
 *
 * Mirrors `commands::ocr::AttachmentOcrEvent`. A phase label says a model is
 * busy; these carry its actual output, so the reading is visible as it
 * happens rather than only in the finished answer.
 */
export type AttachmentOcrEvent =
  | {
      event: 'region';
      name: string;
      page: number;
      index: number;
      /** `title`, `text`, `table`, `figure`, `footer` — the model's label. */
      label: string;
    }
  | {
      event: 'text';
      name: string;
      page: number;
      index: number | null;
      delta: string;
    }
  | {
      event: 'page';
      name: string;
      page: number;
      pages: number;
      modelId: string;
      detent: OcrDetent;
      characters: number;
      elapsedMs: number;
      /**
       * True when the read stopped because it ran out of decode budget rather
       * than because the model finished — the signature of a page the model
       * looped on. What came back is as much as fitted, not the page.
       */
      hitDecodeCap: boolean;
      /**
       * True when the read degenerated into repetition and was cut there.
       *
       * The commoner of the two now that the backend stops a loop as soon as
       * it recognises one: a page cut this way never reaches the decode cap,
       * so `hitDecodeCap` stays false and this is the only signal that the
       * text below stops short of the page. `characters` counts what survived
       * the cut.
       */
      looped: boolean;
    };

export async function listenAttachmentOcr(
  callback: (payload: AttachmentOcrEvent) => void
) {
  return listen<AttachmentOcrEvent>('attachment:ocr', (e) => callback(e.payload));
}

export interface PageImage {
  /** A `data:` URI — the page lives in app data, out of the webview's reach. */
  dataUrl: string;
  /** The overlay's coordinate space, read from the file rather than assumed. */
  width: number;
  height: number;
}

/** Loads a rendered page for display. */
export async function getPageImage(
  documentSha256: string,
  page: number
): Promise<PageImage> {
  return invoke<PageImage>('get_page_image', { documentSha256, page });
}

/** Starts a scan of one page. Progress arrives on the listeners below. */
export async function scanPage(
  documentSha256: string,
  page: number,
  detent: OcrDetent
): Promise<void> {
  return invoke('scan_page', { documentSha256, page, detent });
}

export async function cancelScan(): Promise<void> {
  return invoke('cancel_scan');
}

export async function listenOcrSpan(callback: (payload: OcrEvent) => void) {
  return listen<OcrEvent>('ocr:span', (event: Event<OcrEvent>) => {
    callback(event.payload);
  });
}

export async function listenOcrStatus(
  callback: (payload: OcrStatusPayload) => void
) {
  return listen<OcrStatusPayload>('ocr:status', (event: Event<OcrStatusPayload>) => {
    callback(event.payload);
  });
}

export async function listenOcrError(
  callback: (payload: OcrErrorPayload) => void
) {
  return listen<OcrErrorPayload>('ocr:error', (event: Event<OcrErrorPayload>) => {
    callback(event.payload);
  });
}

/** One region, accumulated from the events that describe it. */
export interface ScannedRegion {
  index: number;
  label: string;
  pageBox: PageBox | null;
  text: string;
}

/**
 * Folds the event stream into regions.
 *
 * Kept out of the component so it can be reasoned about without a render:
 * text deltas arrive after their region and have to land in the right one,
 * and deltas for a region that never opened must not create a phantom.
 */
export function applyOcrEvent(
  regions: ScannedRegion[],
  event: OcrEvent
): ScannedRegion[] {
  if (event.event === 'region') {
    return [
      ...regions,
      {
        index: event.index,
        label: event.label,
        pageBox: event.pageBox,
        text: '',
      },
    ];
  }
  if (event.index === null) {
    return regions;
  }
  const at = regions.findIndex((r) => r.index === event.index);
  if (at === -1) {
    return regions;
  }
  const next = regions.slice();
  next[at] = { ...next[at], text: next[at].text + event.delta };
  return next;
}

/* ------------------------------------------------------------------ *
 * Chat attachments: folding the live read into something renderable.
 * ------------------------------------------------------------------ */

/** One region of one page, as the model is building it. */
export interface OcrReadRegion {
  index: number;
  /** `title`, `text`, `table`, `figure`, `footer` — the model's own label. */
  label: string;
  text: string;
}

/** One page of one attachment, as the model reads it. */
export interface OcrPageRead {
  name: string;
  page: number;
  /** Known once the page finishes; null while it is still the only one seen. */
  pages: number | null;
  regions: OcrReadRegion[];
  /**
   * Text the model emitted without opening a region.
   *
   * Kept rather than dropped: an ungrounded line is still something the model
   * read, and silently discarding it would make the readout disagree with the
   * text that actually reaches the answer.
   */
  loose: string;
  done: boolean;
  /** Facts about the finished read. Null until the page completes. */
  modelId: string | null;
  detent: OcrDetent | null;
  characters: number | null;
  elapsedMs: number | null;
  /** True when the read ran out of decode budget instead of finishing. */
  hitDecodeCap: boolean;
  /** True when the read was cut because the model began repeating itself. */
  looped: boolean;
}

function blankPage(name: string, page: number): OcrPageRead {
  return {
    name,
    page,
    pages: null,
    regions: [],
    loose: '',
    done: false,
    modelId: null,
    detent: null,
    characters: null,
    elapsedMs: null,
    hitDecodeCap: false,
    looped: false,
  };
}

/**
 * Folds `attachment:ocr` events into per-page reads.
 *
 * Kept out of the component so it can be reasoned about without a render.
 * Two properties matter and are easy to get wrong: a text delta must land in
 * the region that opened it rather than the newest one, and a delta for a
 * region that never opened must not conjure a phantom region — it goes to
 * `loose` instead. Both mirror `applyOcrEvent`, which solves the same problem
 * for the scan view.
 */
export function applyAttachmentOcrEvent(
  pages: OcrPageRead[],
  event: AttachmentOcrEvent
): OcrPageRead[] {
  const at = pages.findIndex(
    (p) => p.name === event.name && p.page === event.page
  );
  const next = pages.slice();
  const target = at === -1 ? blankPage(event.name, event.page) : { ...next[at] };
  if (at === -1) {
    next.push(target);
  } else {
    next[at] = target;
  }

  if (event.event === 'region') {
    // An index that has already been seen on this page means the page is
    // being read again, not that the model found a second region with the
    // same number. Replacing rather than appending is what stops a retry
    // from showing every region twice with the text split between them.
    const existing = target.regions.findIndex(r => r.index === event.index);
    if (existing !== -1) {
      const regions = target.regions.slice(0, existing);
      regions.push({ index: event.index, label: event.label, text: '' });
      target.regions = regions;
      return next;
    }
    target.regions = [
      ...target.regions,
      { index: event.index, label: event.label, text: '' },
    ];
    return next;
  }

  if (event.event === 'text') {
    if (event.index === null) {
      target.loose += event.delta;
      return next;
    }
    const region = target.regions.findIndex((r) => r.index === event.index);
    if (region === -1) {
      target.loose += event.delta;
      return next;
    }
    const regions = target.regions.slice();
    regions[region] = {
      ...regions[region],
      text: regions[region].text + event.delta,
    };
    target.regions = regions;
    return next;
  }

  // `page`: the read is over and every number is now measured rather than
  // in progress.
  target.pages = event.pages;
  target.done = true;
  target.modelId = event.modelId;
  target.detent = event.detent;
  target.characters = event.characters;
  target.elapsedMs = event.elapsedMs;
  target.hitDecodeCap = event.hitDecodeCap;
  // Defaulted rather than assumed present: a page event from a build that
  // predates the repetition guard carries no flag, and reading `undefined` as
  // "looped" would put a warning on every historical read.
  target.looped = event.looped ?? false;
  return next;
}
