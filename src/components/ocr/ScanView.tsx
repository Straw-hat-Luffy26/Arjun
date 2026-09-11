import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  applyOcrEvent,
  cancelScan,
  getOcrDetents,
  listenOcrError,
  listenOcrSpan,
  listenOcrStatus,
  scanPage,
  type OcrDetent,
  type OcrDetentInfo,
  type OcrStatusPayload,
  type ScannedRegion,
} from '../../services/ocr.service';
import styles from './ScanView.module.css';

/**
 * Watching the model read a page.
 *
 * The OCR model commits to one region at a time and says where it looked, so
 * a scan is something the operator can follow rather than wait out. Each
 * region draws as soon as the model closes its detection box; the text fills
 * in after. The most recent box keeps a read-head outline, which is what
 * makes the progression down the page legible at a glance.
 *
 * Two honesty rules are load-bearing here:
 *
 * - Boxes are drawn only when the backend supplies page coordinates. Until
 *   the build's coordinate convention has been calibrated it sends none, and
 *   this view says so instead of drawing rectangles that look right on a
 *   square page and drift on every other one.
 * - Throughput is shown only when measured. There is no estimate behind the
 *   dash.
 */

const DETENT_ORDER: OcrDetent[] = ['fastest', 'fast', 'detailed', 'maximum'];

export interface ScanViewProps {
  /** Content-addressed id of the document being read. */
  documentSha256: string;
  /** 1-based page number. Pages are read one at a time. */
  page: number;
  /** Where the rendered page image can be loaded from. */
  pageImageSrc: string;
  /** Natural size of that image, which is the overlay's coordinate space. */
  pageWidth: number;
  pageHeight: number;
}

export const ScanView: React.FC<ScanViewProps> = ({
  documentSha256,
  page,
  pageImageSrc,
  pageWidth,
  pageHeight,
}) => {
  const [regions, setRegions] = useState<ScannedRegion[]>([]);
  const [status, setStatus] = useState<OcrStatusPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [detents, setDetents] = useState<OcrDetentInfo[]>([]);
  const [detentIndex, setDetentIndex] = useState(2);
  const [hovered, setHovered] = useState<number | null>(null);
  const railRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let live = true;
    getOcrDetents()
      .then((info) => {
        if (live) setDetents(info);
      })
      .catch(() => {
        // The slider degrades to its four positions without backend labels
        // rather than blocking the scan.
      });
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    const subs = Promise.all([
      listenOcrSpan((event) => setRegions((prev) => applyOcrEvent(prev, event))),
      listenOcrStatus(setStatus),
      listenOcrError((payload) => setError(payload.reason)),
    ]);
    return () => {
      subs.then((unlisten) => unlisten.forEach((fn) => fn()));
    };
  }, []);

  // Follow the model down the page: the newest region scrolls into view.
  useEffect(() => {
    const rail = railRef.current;
    if (rail) rail.scrollTop = rail.scrollHeight;
  }, [regions.length]);

  const reading = status?.state === 'reading';
  const current = regions.length > 0 ? regions[regions.length - 1].index : null;

  /**
   * Whether the backend is supplying page coordinates at all. Derived from
   * the data rather than a flag, so a build that starts or stops sending them
   * is reflected without a second source of truth.
   */
  const hasCoordinates = useMemo(
    () => regions.some((r) => r.pageBox !== null),
    [regions]
  );

  const outOfBounds = useMemo(
    () => regions.filter((r) => r.pageBox && !r.pageBox.inBounds).length,
    [regions]
  );

  const start = useCallback(() => {
    setRegions([]);
    setError(null);
    setStatus(null);
    scanPage(documentSha256, page, DETENT_ORDER[detentIndex]).catch((e) =>
      setError(String(e))
    );
  }, [documentSha256, page, detentIndex]);

  const active = detents[detentIndex];
  // Stops 1-2 share one weight file and 3-4 the other, so only the 2->3 move
  // reloads. Marking that boundary is the difference between a slider that
  // feels instant and one that mysteriously stalls.
  //
  // Compared against the tier this view opened at, not against `detents[1]`.
  // The old form reduced to "the current stop is in the High tier", which is
  // true at `Detailed` — the default — so the note was showing before anyone
  // had moved the slider, on a move that was not going to reload anything.
  const openedAtTier = useRef<string | null>(null);
  if (openedAtTier.current === null && active) {
    openedAtTier.current = active.tier;
  }
  const crossesReload = Boolean(
    active && openedAtTier.current && active.tier !== openedAtTier.current,
  );

  return (
    <div className={styles.wrap}>
      <header className={styles.head}>
        <div>
          <h2 className={styles.title}>Reading page {page}</h2>
          <p className={styles.sub}>
            {regions.length} region{regions.length === 1 ? '' : 's'}
            {status ? ` · ${(status.elapsedMs / 1000).toFixed(1)}s` : ''}
            {' · '}
            {status?.tokensPerSecond != null
              ? `${status.tokensPerSecond.toFixed(1)} tok/s`
              : '— tok/s'}
          </p>
        </div>
        <div className={styles.actions}>
          {reading ? (
            <button type="button" onClick={() => cancelScan()}>
              Stop
            </button>
          ) : (
            <button type="button" onClick={start}>
              Read page
            </button>
          )}
        </div>
      </header>

      <div className={styles.slider}>
        <label htmlFor="ocr-detent" className={styles.sliderLabel}>
          Accuracy
        </label>
        <input
          id="ocr-detent"
          type="range"
          min={0}
          max={3}
          step={1}
          value={detentIndex}
          onChange={(e) => setDetentIndex(Number(e.target.value))}
          disabled={reading}
        />
        <span className={styles.sliderValue}>
          {active
            ? `${active.label} · ${active.tierLabel} · up to ${active.maxDecodeTokens} tokens a page`
            : DETENT_ORDER[detentIndex]}
        </span>
      </div>
      {crossesReload && (
        <p className={styles.reload}>
          This stop uses the other weight file — switching to it reloads the
          model.
        </p>
      )}

      {error && <p className={styles.error}>{error}</p>}

      {regions.length > 0 && !hasCoordinates && (
        <p className={styles.notice}>
          Text only — the model returned regions without page coordinates, so
          the overlay is hidden. Boxes appear once the coordinate space has
          been calibrated.
        </p>
      )}
      {outOfBounds > 0 && (
        <p className={styles.notice}>
          {outOfBounds} box{outOfBounds === 1 ? '' : 'es'} fell outside the
          page. That means the coordinate space is wrong, not the page — they
          are drawn in red rather than hidden.
        </p>
      )}

      <div className={styles.split}>
        <div className={styles.pane}>
          <div className={styles.pageStack}>
            <img
              className={styles.page}
              src={pageImageSrc}
              alt={`Page ${page}`}
              width={pageWidth}
              height={pageHeight}
            />
            <svg
              className={styles.overlay}
              viewBox={`0 0 ${pageWidth} ${pageHeight}`}
              preserveAspectRatio="xMidYMid meet"
              aria-hidden="true"
            >
              {regions.map((region) => {
                const box = region.pageBox;
                if (!box) return null;
                const classes = [
                  styles.box,
                  styles[`label_${region.label}`] ?? styles.label_text,
                  region.index === current ? styles.readHead : '',
                  box.inBounds ? '' : styles.offPage,
                  region.index === hovered ? styles.linked : '',
                ]
                  .filter(Boolean)
                  .join(' ');
                return (
                  <rect
                    key={region.index}
                    className={classes}
                    x={box.x1}
                    y={box.y1}
                    width={Math.max(0, box.x2 - box.x1)}
                    height={Math.max(0, box.y2 - box.y1)}
                    rx={2}
                  />
                );
              })}
            </svg>
          </div>
        </div>

        <div className={styles.rail} ref={railRef}>
          {regions.length === 0 && !reading && (
            <p className={styles.empty}>
              Nothing read yet. Choose an accuracy stop and start the page.
            </p>
          )}
          {regions.map((region) => (
            <div
              key={region.index}
              className={`${styles.block} ${
                region.index === current ? styles.blockCurrent : ''
              }`}
              onMouseEnter={() => setHovered(region.index)}
              onMouseLeave={() => setHovered(null)}
            >
              <span className={styles.blockLabel}>{region.label}</span>
              <span className={styles.blockText}>{region.text}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
