import React, { useCallback, useEffect, useState } from 'react';
import { Eye, ScanText } from 'lucide-react';
import {
  documentAnalystService,
  type DocumentAnalystStatus,
} from '../services/documentAnalyst.service';
import styles from './Agents.module.css';

const short = (hash: string | null | undefined, length = 12) =>
  hash ? `${hash.slice(0, length)}…` : 'not pinned';

/**
 * What reads pages on this machine (plan P06).
 *
 * The OCR model is shown by its exact files, because "Unlimited-OCR" names a
 * family and a third-party conversion, not the weights that read a given page.
 * A model with a projector is listed as vision-ready only when its last image
 * probe on this machine read the token back; the reason is shown otherwise.
 */
export const DocumentAnalystPanel: React.FC = () => {
  const [status, setStatus] = useState<DocumentAnalystStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [modelId, setModelId] = useState('');
  const [projectorPath, setProjectorPath] = useState('');

  const load = useCallback(async () => {
    try {
      setStatus(await documentAnalystService.status());
      setError(null);
    } catch (problem) {
      setError(String(problem));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const probe = async (id: string) => {
    setBusy(id);
    setMessage(null);
    try {
      const record = await documentAnalystService.probe(id);
      setMessage(
        record.passed
          ? `${id} read the probe token ${record.probeToken} from an image: vision-ready.`
          : `${id} did not read the probe token: ${record.reason}`,
      );
    } catch (problem) {
      setMessage(String(problem));
    } finally {
      setBusy(null);
      void load();
    }
  };

  const bind = async (event: React.FormEvent) => {
    event.preventDefault();
    setBusy('bind');
    setMessage(null);
    try {
      const outcome = await documentAnalystService.bindProjector(modelId.trim(), projectorPath.trim());
      setMessage(
        `Bound ${outcome.binding.projector} to ${outcome.binding.modelId} (width ${outcome.binding.projectionDim} checked). ` +
          'It is used from the next start, and the model is not vision-ready until it passes an image probe.',
      );
      void load();
    } catch (problem) {
      setMessage(String(problem));
    } finally {
      setBusy(null);
    }
  };

  return (
    <section className={styles.dryRun} aria-label="Document and vision analyst">
      <h3 className={styles.previewTitle}>
        <ScanText size={14} aria-hidden /> What reads pages on this machine
      </h3>
      {error && <p className={styles.quiet}>{error}</p>}
      {status && (
        <dl className={styles.facts}>
          <dt>OCR</dt>
          <dd>
            {status.ocr ? (
              <>
                {status.ocr.modelId} ({status.ocr.detent}) · weights {short(status.ocr.weightsSha256)} ·
                projector {status.ocr.projectorFile ?? 'none'}
                {status.ocr.projectorBytes !== null ? ` (${status.ocr.projectorBytes} bytes)` : ''} ·{' '}
                {status.ocrTransport}
              </>
            ) : (
              <span className={styles.refused}>{status.ocrUnavailable}</span>
            )}
          </dd>
          <dt>Page analyser</dt>
          <dd>
            {status.pageAnalyser ?? <span className={styles.refused}>{status.pageAnalyserUnavailable}</span>}
          </dd>
          <dt>Interpreter</dt>
          <dd>
            {status.interpreter ?? (
              <span className={styles.refused}>
                none vision-ready — interpretation is not attempted, and nothing is guessed in its place
              </span>
            )}
          </dd>
        </dl>
      )}
      {status && status.vision.length > 0 && (
        <ul className={styles.jobs} aria-label="Models with an image projector">
          {status.vision.map((candidate) => (
            <li key={candidate.modelId} className={styles.analystRow} data-status={candidate.ready ? 'ready' : 'not-ready'}>
              <span className={styles.rowMain}>
                <span className={styles.rowName}>{candidate.name}</span>
                <span className={styles.hint}>{candidate.projector ?? 'no projector'}</span>
              </span>
              <span className={styles.rowMain}>
                <span className={candidate.ready ? styles.ready : styles.notReady}>
                  {candidate.ready ? 'vision-ready' : 'not vision-ready'}
                </span>
                {candidate.reason && <span className={styles.hint}>{candidate.reason}</span>}
                {candidate.lastProbe && (
                  <span className={styles.hint}>
                    last probe {candidate.lastProbe.at}: token {candidate.lastProbe.probeToken},{' '}
                    {candidate.lastProbe.passed ? 'read' : 'not read'} — “{candidate.lastProbe.answerExcerpt}”
                  </span>
                )}
              </span>
              <span className={styles.rowMain}>
                <button
                  type="button"
                  className={styles.ghost}
                  disabled={busy !== null}
                  onClick={() => void probe(candidate.modelId)}
                >
                  <Eye size={14} aria-hidden /> {busy === candidate.modelId ? 'Probing…' : 'Probe with an image'}
                </button>
              </span>
            </li>
          ))}
        </ul>
      )}
      <form className={styles.actions} onSubmit={(event) => void bind(event)} aria-label="Bind an image projector">
        <input
          className={styles.input}
          value={modelId}
          onChange={(event) => setModelId(event.target.value)}
          placeholder="model id, e.g. gemma-4-e4b-q4"
          aria-label="Model id"
        />
        <input
          className={styles.input}
          value={projectorPath}
          onChange={(event) => setProjectorPath(event.target.value)}
          placeholder="projector file, any name"
          aria-label="Projector file"
        />
        <button type="submit" className={styles.ghost} disabled={busy !== null || !modelId.trim() || !projectorPath.trim()}>
          Bind projector
        </button>
      </form>
      <p className={styles.hint}>
        A projector is checked against the model from both files&apos; headers before it is bound. Binding is not
        seeing: only an image probe that reads a random token back makes a model vision-ready.
      </p>
      {message && <p className={styles.quiet} role="status">{message}</p>}
    </section>
  );
};
