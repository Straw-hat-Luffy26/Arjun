import { useCallback, useEffect, useState } from 'react';
import { agentService, isTerminal, type Resumability, type TaskSnapshot } from '../../services/agent.service';
import styles from './RunView.module.css';

/** The server rechecks identity, policy, lease and effects on every resume. */
export function RecoveryControls({ runId }: { runId: string }) {
  const [snapshot, setSnapshot] = useState<TaskSnapshot | null>(null);
  const [verdict, setVerdict] = useState<Resumability | null>(null);
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    const current = await agentService.snapshot(runId);
    setSnapshot(current);
    setVerdict(current && (current.state === 'paused' || current.state === 'recovering')
      ? await agentService.resumability(runId) : null);
  }, [runId]);

  useEffect(() => {
    let live = true;
    let unsubscribe: (() => void) | undefined;
    const update = () => { if (live) void refresh().catch(error => setNotice(String(error))); };
    void agentService.subscribeDurable(event => {
      if (event.runId === runId && ['runPaused', 'runResumed', 'recoveryStarted', 'runCompleted',
        'runFailed', 'runDegraded', 'runCancelled', 'toolEffectReconciled'].includes(event.eventType)) update();
    }).then(stop => { if (live) unsubscribe = stop; else stop(); }).catch(error => { if (live) setNotice(String(error)); });
    update();
    return () => { live = false; unsubscribe?.(); };
  }, [runId, refresh]);

  const act = async (resume: boolean) => {
    setBusy(true);
    setNotice(null);
    try {
      if (resume) await agentService.resume(runId, 'Continue this saved task from the recovery controls.');
      else setNotice(await agentService.pause(runId)
        ? 'Pause requested. The current tool batch will finish before the saved pause boundary.'
        : 'This task has no active worker to pause.');
      await refresh();
    } catch (error) { setNotice(String(error)); }
    finally { setBusy(false); }
  };

  if (!snapshot) return notice ? <p role="alert">{notice}</p> : null;
  if (isTerminal(snapshot.state)) return null;
  const waiting = snapshot.state === 'paused' || snapshot.state === 'recovering';
  return <section aria-label="Task recovery" className={styles.section}>
    {waiting ? <>
      <p>{snapshot.state === 'paused' ? 'Task paused.' : 'Task interrupted.'} Resume continues the same task with its remaining budget.</p>
      {verdict && verdict.status !== 'resumable' && <p role="status">{verdict.because}</p>}
      <button className={styles.newBtn} disabled={busy || verdict?.status !== 'resumable'} onClick={() => void act(true)}>
        {busy ? 'Continuing…' : 'Resume task'}
      </button>
      <button className={styles.newBtn} disabled={busy} onClick={() => void refresh().catch(error => setNotice(String(error)))}>Check recovery</button>
    </> : <button className={styles.newBtn} disabled={busy} onClick={() => void act(false)}>{busy ? 'Requesting pause…' : 'Pause task'}</button>}
    {notice && <p role="status">{notice}</p>}
  </section>;
}
