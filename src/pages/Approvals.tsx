import React, { useCallback, useEffect, useRef, useState } from 'react';
import { AlertTriangle, Check, FileOutput, Inbox, ShieldCheck, X } from 'lucide-react';
import { approvalsService, type ApprovalItem } from '../services/approvals.service';
import styles from './Approvals.module.css';

/** How often the queue re-reads. Local state, so this costs nothing off-machine. */
const POLL_INTERVAL_MS = 4000;

const formatTime = (iso: string) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? iso : d.toLocaleString();
};

/* The five things ARJUN design rule 26 requires be shown before a risky action. Listed
 * here as data so the card cannot quietly stop rendering one of them. */
const FACETS = [
  { key: 'target', label: 'Target' },
  { key: 'arguments', label: 'Arguments' },
  { key: 'evidence', label: 'Supporting evidence' },
  { key: 'expectedOutput', label: 'Expected output' },
  { key: 'consequences', label: 'Consequences' },
] as const;

const Card = ({
  item,
  onDecide,
  busy,
}: {
  item: ApprovalItem;
  onDecide: (id: string, approve: boolean, because?: string) => Promise<void>;
  busy: boolean;
}) => {
  const [rejecting, setRejecting] = useState(false);
  const [reason, setReason] = useState('');
  const { request, decision } = item;

  const value = (key: (typeof FACETS)[number]['key']) => {
    switch (key) {
      case 'target':
        return <code className={styles.target}>{request.target}</code>;
      case 'arguments':
        return (
          <ul className={styles.list}>
            {request.arguments.map((a) => (
              <li key={a}>
                <code>{a}</code>
              </li>
            ))}
          </ul>
        );
      case 'evidence':
        return (
          <ul className={styles.list}>
            {request.evidence.map((e) => (
              <li key={e}>{e}</li>
            ))}
          </ul>
        );
      default:
        return <p className={styles.prose}>{request[key]}</p>;
    }
  };

  return (
    <article className={`${styles.card} ${decision ? styles.settled : ''}`}>
      <header className={styles.cardHead}>
        <div>
          <h2 className={styles.tool}>{request.tool}</h2>
          <p className={styles.meta}>
            Task {request.taskId} · raised by {request.requestedBy} ·{' '}
            {formatTime(request.requestedAt)}
          </p>
        </div>
        {decision && (
          <span
            className={`${styles.badge} ${
              decision.decision === 'approved' ? styles.badgeApproved : styles.badgeRejected
            }`}
          >
            {decision.decision === 'approved' ? 'Approved' : 'Rejected'} by {decision.by}
          </span>
        )}
      </header>

      <dl className={styles.facets}>
        {FACETS.map((facet) => (
          <div className={styles.facet} key={facet.key}>
            <dt>{facet.label}</dt>
            <dd>{value(facet.key)}</dd>
          </div>
        ))}
      </dl>

      {decision?.decision === 'rejected' && (
        <p className={styles.rejectionReason}>
          <AlertTriangle size={14} /> {decision.because}
        </p>
      )}

      {!decision && (
        <footer className={styles.actions}>
          {rejecting ? (
            <div className={styles.rejectRow}>
              <input
                className={styles.reasonInput}
                placeholder="Why is this being rejected? The task needs this to do anything else."
                value={reason}
                onChange={(e) => setReason(e.target.value)}
                autoFocus
              />
              <button
                type="button"
                className={styles.reject}
                disabled={busy || reason.trim().length === 0}
                onClick={() => void onDecide(request.id, false, reason)}
              >
                Confirm rejection
              </button>
              <button
                type="button"
                className={styles.cancel}
                onClick={() => {
                  setRejecting(false);
                  setReason('');
                }}
              >
                Cancel
              </button>
            </div>
          ) : (
            <>
              <button
                type="button"
                className={styles.approve}
                disabled={busy}
                onClick={() => void onDecide(request.id, true)}
              >
                <Check size={15} /> Approve
              </button>
              <button
                type="button"
                className={styles.reject}
                disabled={busy}
                onClick={() => setRejecting(true)}
              >
                <X size={15} /> Reject
              </button>
            </>
          )}
        </footer>
      )}
    </article>
  );
};

export const Approvals = () => {
  const [items, setItems] = useState<ApprovalItem[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const mountedRef = useRef(true);

  const refresh = useCallback(async () => {
    try {
      const next = await approvalsService.list();
      if (!mountedRef.current) return;
      setItems(next);
      setError(null);
    } catch (e) {
      if (!mountedRef.current) return;
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(timer);
    };
  }, [refresh]);

  const onDecide = async (id: string, approve: boolean, because?: string) => {
    setBusy(true);
    try {
      await approvalsService.decide(id, approve, because);
      setError(null);
      await refresh();
    } catch (e) {
      // The backend's refusal is the useful text — "you are not a reviewer",
      // "this was already approved" — so it is shown verbatim.
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  const pending = items.filter((i) => !i.decision);
  const settled = items.filter((i) => i.decision);

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Approvals</h1>
        <p className={styles.subtitle}>
          Actions waiting on a person. Each shows what it would act on, with what arguments, the
          evidence behind it, what it should produce and what it would change &mdash; before
          anything is written. Only a reviewer can decide, and a decision cannot be reversed.
        </p>
      </header>

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={18} />
          <p>{error}</p>
        </div>
      )}

      {pending.length === 0 && settled.length === 0 && (
        <div className={styles.empty}>
          <Inbox size={22} />
          <div>
            <strong>Nothing is waiting on you.</strong>
            <p>
              A task pauses here before it writes a file outside its own workspace, runs code, or
              deletes a notebook. Producing a document you asked for &mdash; a note, a workbook, a
              deck &mdash; does not stop here: you already asked for it, and it is written inside
              the task&rsquo;s own folder.
            </p>
          </div>
        </div>
      )}

      {pending.length > 0 && (
        <section>
          <h2 className={styles.sectionHead}>
            <ShieldCheck size={16} /> Waiting on a decision ({pending.length})
          </h2>
          <div className={styles.stack}>
            {pending.map((item) => (
              <Card key={item.request.id} item={item} onDecide={onDecide} busy={busy} />
            ))}
          </div>
        </section>
      )}

      {settled.length > 0 && (
        <section>
          <h2 className={styles.sectionHead}>
            <FileOutput size={16} /> Decided this session ({settled.length})
          </h2>
          <div className={styles.stack}>
            {settled.map((item) => (
              <Card key={item.request.id} item={item} onDecide={onDecide} busy={busy} />
            ))}
          </div>
        </section>
      )}
    </div>
  );
};
