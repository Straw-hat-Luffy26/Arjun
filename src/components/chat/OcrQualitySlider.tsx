import React from 'react';
import { Gauge } from 'lucide-react';
import { OCR_DETENT_ORDER } from '../../services/ocr.service';
import type { OcrPreference } from './useOcrPreference';
import styles from './ChatSurface.module.css';

/**
 * The accuracy-to-speed control for reading attachments, docked under the
 * composer and mounted only while an attachment is on the turn.
 *
 * ## Why a slider and not a model picker
 *
 * The obvious control is "which OCR model" — and it is the wrong one. Moving
 * between weight files means a different file on disk and a model reload; a
 * control whose every notch might stall for a reload is not a control this
 * product can offer. The lever that costs nothing is the vision token budget,
 * which llama.cpp takes as a per-request argument. So the four stops span two
 * installed weight files: stops 1–2 share Q4_K_M, stops 3–4 share Q6_K, and
 * only the 2→3 move reloads. The caption says which stop is which, and the
 * reload boundary is marked rather than left to be discovered.
 *
 * Every number shown here comes from `get_ocr_detents`, which reads the same
 * `ocr_profile` the run uses. Nothing is hard-coded in this file except the
 * order of the stops.
 */
export interface OcrQualitySliderProps {
  preference: OcrPreference;
  /** True while a run is in flight; the stops are fixed for that turn. */
  disabled?: boolean;
  /**
   * True when the turn being composed carries a file that needs reading.
   *
   * The control is only mounted once an attachment is present (see
   * `ChatComposer`), so this is not "shown or hidden" — it is the difference
   * between a scanned PDF, which this setting governs, and a .txt attached
   * alongside it, which it does not. Styling goes quiet in the second case
   * rather than the control vanishing mid-compose.
   */
  engaged?: boolean;
}

export function OcrQualitySlider({
  preference,
  disabled,
  engaged,
}: OcrQualitySliderProps) {
  const { index, setDetent, active, crossesReload } = preference;

  return (
    <div
      className={styles.ocrSlider}
      data-engaged={engaged || undefined}
      data-disabled={disabled || undefined}
    >
      <Gauge size={14} className={styles.ocrSliderIcon} aria-hidden="true" />
      <label htmlFor="chat-ocr-detent" className={styles.ocrSliderLabel}>
        Document reading
      </label>

      <span className={styles.ocrSliderEnd} aria-hidden="true">
        Fast
      </span>
      <input
        id="chat-ocr-detent"
        className={styles.ocrSliderInput}
        type="range"
        min={0}
        max={OCR_DETENT_ORDER.length - 1}
        step={1}
        value={index}
        disabled={disabled}
        onChange={(e) => setDetent(OCR_DETENT_ORDER[Number(e.target.value)])}
        aria-label="How carefully attached documents are read"
        aria-valuetext={
          active
            ? `${active.label}, ${active.tierLabel} tier, up to ${active.maxDecodeTokens} tokens a page`
            : OCR_DETENT_ORDER[index]
        }
      />
      <span className={styles.ocrSliderEnd} aria-hidden="true">
        Accurate
      </span>

      {/* The caption is the honest part: it names the weight tier and the
        * vision budget, so "Accurate" is not an adjective the UI made up. */}
      <span className={styles.ocrSliderValue}>
        {active ? (
          <>
            <strong>{active.label}</strong>
            <span className={styles.ocrSliderDetail}>
              {active.tierLabel} tier · up to {active.maxDecodeTokens} tokens a page
            </span>
          </>
        ) : (
          <strong>{OCR_DETENT_ORDER[index]}</strong>
        )}
      </span>

      {crossesReload && (
        <span className={styles.ocrSliderReload} role="note">
          uses the other weight file — first use reloads the model
        </span>
      )}
    </div>
  );
}
