import { useCallback, useEffect, useRef, useState } from 'react';
import {
  OCR_DETENT_ORDER,
  getOcrDetents,
  type OcrDetent,
  type OcrDetentInfo,
} from '../../services/ocr.service';

/**
 * Where the accuracy-to-speed slider is set for chat attachments.
 *
 * The chat path used to read every attachment at the `detailed` stop with no
 * way to change it — the slider existed only on the document-scan screen, so
 * a person who attached a photograph in chat could not trade accuracy for
 * speed at all. The preference lives here, is remembered between sessions,
 * and is sent with the turn.
 *
 * The stops themselves are read from the backend rather than listed here.
 * `OCR_DETENT_ORDER` is only the slider's *order*; what each stop costs — the
 * weight file, the vision token budget — is whatever `ocr_profile` says it
 * is, so the labels cannot drift from the profiles that run.
 */
const STORAGE_KEY = 'arjun.ocr.detent';

/** The stop used when nobody has moved the slider. Matches the backend. */
export const DEFAULT_OCR_DETENT: OcrDetent = 'detailed';

function remembered(): OcrDetent {
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved && (OCR_DETENT_ORDER as string[]).includes(saved)) {
      return saved as OcrDetent;
    }
  } catch {
    // A webview with storage disabled is not a reason to lose the slider.
  }
  return DEFAULT_OCR_DETENT;
}

export interface OcrPreference {
  detent: OcrDetent;
  setDetent: (next: OcrDetent) => void;
  /** Slider position, 0–3. */
  index: number;
  /** Every stop, as the backend describes it. Empty until it answers. */
  detents: OcrDetentInfo[];
  /** The stop currently selected, or null before the backend has answered. */
  active: OcrDetentInfo | null;
  /**
   * True when moving to this stop swaps the weight file.
   *
   * Stops 1–2 share Q4_K_M and stops 3–4 share Q6_K; only the 2→3 move
   * reloads. Saying so is the difference between a slider that feels instant
   * and one that stalls without explanation.
   */
  crossesReload: boolean;
}

export function useOcrPreference(): OcrPreference {
  const [detent, setDetentState] = useState<OcrDetent>(remembered);
  const [detents, setDetents] = useState<OcrDetentInfo[]>([]);

  useEffect(() => {
    let live = true;
    getOcrDetents()
      .then((info) => {
        if (live) setDetents(info);
      })
      .catch(() => {
        // The slider still works with its four positions; only the cost
        // labels are missing. Failing closed here would remove a control
        // that governs real work for the sake of a caption.
      });
    return () => {
      live = false;
    };
  }, []);

  const setDetent = useCallback((next: OcrDetent) => {
    setDetentState(next);
    try {
      window.localStorage.setItem(STORAGE_KEY, next);
    } catch {
      // Not remembering it is a smaller failure than refusing to set it.
    }
  }, []);

  const index = Math.max(0, OCR_DETENT_ORDER.indexOf(detent));
  const active = detents[index] ?? null;

  /**
   * The tier that was in force when this surface opened.
   *
   * Held so the warning below can describe a *move* rather than a position.
   * It used to be `active.tier !== detents[1].tier`, which is just "the current
   * stop is in the High tier" — permanently true at `Detailed`, the default, so
   * the "first use reloads the model" note was on screen before anyone had
   * touched the slider. Rust has the correct predicate already
   * (`OcrDetent::reloads_from`), exported for nobody.
   */
  const openedAtTier = useRef<string | null>(null);
  if (openedAtTier.current === null && active != null) {
    openedAtTier.current = active.tier;
  }

  const crossesReload =
    active != null &&
    openedAtTier.current != null &&
    active.tier !== openedAtTier.current;

  return { detent, setDetent, index, detents, active, crossesReload };
}
