import React, { useCallback, useEffect, useState } from 'react';
import {
  HardDrive,
  Trash2,
  RefreshCw,
  Power,
  Play,
  AlertTriangle,
  Download,
  Star,
  ScanSearch,
} from 'lucide-react';
import { Button, Spinner } from '../components/ui';
import { Can } from '../components/auth/Can';
import { useToast } from '../hooks/useToast';
import {
  getInstalledModels,
  deleteInstalledModel,
  getStorageSummary,
} from '../services/download.service';
import { getInferenceStatus, loadInstalledModel, unloadActiveModel } from '../services/ai.service';
import { formatSize } from '../services/catalog.service';
import {
  registryService,
  type LibraryModel,
  type OrchestratorModelSelection,
  type OrchestratorSwapStep,
} from '../services/registry.service';
import styles from './Storage.module.css';

/** A quantisation that names the container rather than the weights. */
const isPlaceholderQuant = (quantization?: string | null) =>
  !quantization?.trim() || quantization.trim().toLowerCase() === 'gguf';

/**
 * Whether the configured orchestrator is this installed model.
 *
 * The provider and model id are the package's identity and are compared
 * strictly. Quantisation is compared only when both sides name a real one:
 * what is saved is the registry's spelling, and a package whose file name
 * declares no quantisation records the placeholder "GGUF" forever. Demanding
 * they match exactly would drop the star off the model the chat is actually
 * using, which is the confusion this whole area is being fixed for.
 */
const isSameModel = (
  chosen: OrchestratorModelSelection | null,
  installed: { providerId: string; modelId: string; quantization: string }
) => {
  if (!chosen) return false;
  if (chosen.providerId !== installed.providerId || chosen.modelId !== installed.modelId) {
    return false;
  }
  if (isPlaceholderQuant(chosen.quantization) || isPlaceholderQuant(installed.quantization)) {
    return true;
  }
  return chosen.quantization.toLowerCase() === installed.quantization.toLowerCase();
};

/**
 * Storage — manage what is on disk.
 *
 * Deliberately does *not* recommend or download models; that belongs in
 * Discover. Having both meant two screens that each did half the job and
 * shared a confusing name.
 */
export const Storage: React.FC = () => {
  const { addToast } = useToast();
  const [models, setModels] = useState<any[]>([]);
  /**
   * The manifest, which is not the same list as the installed packages
   * above. A model can be registered and running without ever having been
   * downloaded through this app, which is exactly how both Unlimited-OCR
   * weights came to be missing from this screen.
   */
  const [library, setLibrary] = useState<LibraryModel[]>([]);
  const [summary, setSummary] = useState<any>(null);
  const [loadedModelId, setLoadedModelId] = useState<string | null>(null);
  const [orchestrator, setOrchestrator] = useState<OrchestratorModelSelection | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  /** The stage the running swap is on, or null when none is running. */
  const [swap, setSwap] = useState<OrchestratorSwapStep | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [installed, sum, status, configuredOrchestrator, registered] =
        await Promise.all([
          getInstalledModels(),
          getStorageSummary().catch(() => null),
          getInferenceStatus().catch(() => null),
          registryService.getOrchestratorModel().catch(() => null),
          // Caught rather than thrown: a manifest this build cannot read is a
          // reason to show an empty Registered list, not a reason to hide the
          // packages that are plainly installed.
          registryService.listLibraryModels().catch(() => []),
        ]);
      setModels(installed ?? []);
      setLibrary(registered ?? []);
      setSummary(sum);
      setLoadedModelId(status?.model?.modelId ?? null);
      setOrchestrator(configuredOrchestrator);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Subscribed for the life of the screen rather than opened per swap: the
  // first stage is emitted before `setOrchestratorModel` resolves, so a
  // listener attached at click time would miss the release it exists to show.
  useEffect(() => {
    let unlisten: (() => void) | undefined;
    let cancelled = false;
    void registryService
      .subscribeOrchestratorSwap(step => setSwap(step))
      .then(fn => {
        if (cancelled) fn();
        else unlisten = fn;
      })
      .catch(() => {
        // No live channel. The swap still happens and the toast still
        // reports it; only the running commentary is missing.
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const handleDeleteModel = async (m: any) => {
    // Deleting frees gigabytes and cannot be undone, so it is confirmed and the
    // size is named — "are you sure?" alone does not convey what is at stake.
    const ok = window.confirm(
      `Delete ${m.modelName}?\n\nThis frees ${formatSize(m.sizeBytes)}. ` +
        `You would need to download it again to use it.`
    );
    if (!ok) return;

    setBusy(m.modelId);
    try {
      await deleteInstalledModel(m.providerId, m.modelId, m.quantization);
      addToast('success', `${m.modelName} deleted`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  // Loading reads gigabytes into VRAM and takes a while, so the button reports
  // progress and every other model's button locks — two concurrent loads would
  // fight over the same VRAM budget.
  const handleLoad = async (m: any) => {
    setBusy(m.modelId);
    try {
      await loadInstalledModel(m.providerId, m.modelId, m.quantization);
      addToast('success', `${m.modelName} loaded — the gateway can serve requests`);
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

// Detection walks the disk and writes the manifest. It is deliberately a
  // button rather than something that happens on load: it touches the file
  // that decides what this machine will run, and an operator should be the one
  // who asks for that.
  const handleDetect = async () => {
    setBusy('detect');
    try {
      const report = await registryService.detectSystemModels();
      setLibrary(report.models);
      if (report.added.length === 0) {
        addToast(
          'info',
          `Nothing new — ${report.alreadyRegistered} of ${report.filesSeen} weight files in ` +
            `${report.roots.length} folders were already registered`
        );
      } else {
        const names = report.added.map((m) => m.name).join(', ');
        addToast(
          'success',
          `Registered ${report.added.length} model${report.added.length === 1 ? '' : 's'}: ` +
            `${names}. Each is cleared for no classification until you review it` +
            (report.restartRequired ? ', and routing will use them after a restart' : '')
        );
      }
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
    }
  };

  const handleUnload = async () => {
    try {
      await unloadActiveModel();
      addToast('info', 'Model unloaded — the gateway has nothing to serve until you load one');
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    }
  };

  // Choosing an orchestrator releases whatever is serving and loads the new
  // model, which takes as long as reading several gigabytes off disk. The
  // stages are shown as they arrive rather than behind one "Saving…" label:
  // releasing and loading are the two halves of the wait, and which model each
  // is about is the thing worth knowing while it happens.
  const handleSetOrchestrator = async (m: any) => {
    const busyKey = `orchestrator:${m.id}`;
    setBusy(busyKey);
    setSwap({ phase: 'loading', modelId: m.modelId, modelName: m.modelName, detail: null });
    try {
      const change = await registryService.setOrchestratorModel({
        providerId: m.providerId,
        modelId: m.modelId,
        quantization: m.quantization,
      });
      setOrchestrator(change.selected);

      const releasedNote = change.released.length
        ? `, releasing ${change.released.join(' and ')}`
        : '';
      if (change.serving) {
        addToast(
          'success',
          `${change.modelName} is the orchestrator and is now answering${releasedNote}`
        );
      } else {
        // The choice was saved; the model just is not up. Reporting only
        // "saved" would repeat the original bug in words — the setting looking
        // like it took effect while another model does the talking.
        addToast(
          'error',
          `${change.modelName} is saved as the orchestrator but did not start: ${
            change.detail ?? 'the model server did not come up'
          }`
        );
      }
      await refresh();
    } catch (err) {
      addToast('error', String(err));
    } finally {
      setBusy(null);
      setSwap(null);
    }
  };

  if (loading) {
    return (
      <div className={styles.centered}>
        <Spinner />
        <p>Reading what is on disk…</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Storage</h1>
          <p className={styles.subtitle}>
            Models on this computer. <strong>Detect models</strong> searches the
            disk for weight files that are installed but unregistered; to find
            new ones to download, use Discover.
          </p>
        </div>
        <div className={styles.headerActions}>
          <Can permission="importModel">
            <Button
              variant="ghost"
              size="sm"
              onClick={handleDetect}
              disabled={busy !== null}
              title="Search this computer for weight files and register the ones the manifest does not list"
            >
              <ScanSearch size={14} />{' '}
              {busy === 'detect' ? 'Searching…' : 'Detect models'}
            </Button>
          </Can>
          <Button variant="ghost" size="sm" onClick={refresh} disabled={busy !== null}>
            <RefreshCw size={14} /> Refresh
          </Button>
        </div>
      </header>

      {error && (
        <div className={styles.error} role="alert">
          <AlertTriangle size={15} /> {error}
        </div>
      )}

      {summary && (
        <div className={styles.summary}>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Models</span>
            <span className={styles.statValue}>{models.length}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Used by models</span>
            <span className={styles.statValue}>{formatSize(summary.totalModelsBytes ?? 0)}</span>
          </div>
          <div className={styles.stat}>
            <span className={styles.statLabel}>Free on disk</span>
            <span className={styles.statValue}>{formatSize(summary.availableDiskSpaceBytes ?? 0)}</span>
          </div>
        </div>
      )}

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <HardDrive size={15} /> Installed
        </h2>

        {models.length === 0 && (
          <p className={styles.empty}>
            Nothing installed yet. Open <strong>Discover</strong> to find a model that fits your
            hardware.
          </p>
        )}

        {swap && (
          <p className={styles.empty} role="status" aria-live="polite">
            {swap.phase === 'releasing' && <>Releasing {swap.modelName} from memory…</>}
            {swap.phase === 'loading' && <>Loading {swap.modelName} — this reads the weights off disk…</>}
            {swap.phase === 'ready' && <>{swap.modelName} is loaded and answering.</>}
            {swap.phase === 'failed' && (
              <>
                {swap.modelName} did not start: {swap.detail ?? 'no detail was reported'}
              </>
            )}
          </p>
        )}

        {models.map((m: any) => {
          const isLoaded = loadedModelId === m.modelId;
          const isOrchestrator = isSameModel(orchestrator, m);
          return (
            <article key={`${m.modelId}-${m.quantization}`} className={styles.model}>
              <div className={styles.modelHead}>
                <div className={styles.modelInfo}>
                  <h3 className={styles.modelName}>
                    {m.modelName} <span className={styles.quant}>{m.quantization}</span>
                    {isOrchestrator && (
                      <span className={styles.orchestrator}>★ Orchestrator</span>
                    )}
                  </h3>
                  <p className={styles.modelMeta}>
                    {formatSize(m.sizeBytes)} · {m.providerId}
                  </p>
                </div>

                <div className={styles.modelActions}>
                  {!isOrchestrator && (
                    <Can permission="modifyPolicy">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleSetOrchestrator(m)}
                        disabled={busy !== null || !m.isReady}
                        title="Use this exact model variant as the startup orchestrator"
                      >
                        <Star size={13} />
                        {busy === `orchestrator:${m.id}` ? 'Saving…' : 'Set as orchestrator'}
                      </Button>
                    </Can>
                  )}
                  {isLoaded ? (
                    <>
                      <span className={styles.serving}>● Serving the gateway</span>
                      <Can permission="importModel">
                        <Button variant="ghost" size="sm" onClick={handleUnload}>
                          <Power size={13} /> Unload
                        </Button>
                      </Can>
                    </>
                  ) : (
                    <Can permission="importModel">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => handleLoad(m)}
                        disabled={busy !== null}
                        title="Load this model so the gateway can serve it"
                      >
                        <Play size={13} /> {busy === m.modelId ? 'Loading…' : 'Load'}
                      </Button>
                    </Can>
                  )}
                  <Can permission="importModel">
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => handleDeleteModel(m)}
                      disabled={busy === m.modelId}
                      title="Delete this model"
                    >
                      <Trash2 size={13} />
                    </Button>
                  </Can>
                </div>
              </div>

            </article>
          );
        })}
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <ScanSearch size={15} /> Registered
          <span className={styles.count}>{library.length}</span>
        </h2>

        <p className={styles.empty}>
          Every model the manifest lists, including ones put on disk by hand.
          Detection adds what it finds here; it never edits or removes an entry
          somebody wrote, and everything it adds is cleared for no
          classification until an administrator reviews it.
        </p>

        {library.length === 0 && (
          <p className={styles.empty}>
            The manifest is empty. Press <strong>Detect models</strong> to search
            this computer for weight files.
          </p>
        )}

        {library.map((m) => (
          <article key={m.id} className={styles.model}>
            <div className={styles.modelHead}>
              <div className={styles.modelInfo}>
                <h3 className={styles.modelName}>
                  {m.name}
                  {m.quantization && <span className={styles.quant}>{m.quantization}</span>}
                  {!m.present && (
                    <span className={styles.missing} title={m.path}>
                      file missing
                    </span>
                  )}
                </h3>
                <p className={styles.modelMeta}>
                  {formatSize(m.bytesOnDisk ?? m.weightsBytes)} · {m.runtime}
                  {m.roles.length > 0 && ` · ${m.roles.join(', ')}`}
                  {m.projector && ' · vision projector paired'}
                  {m.permittedClassifications.length === 0 && ' · cleared for nothing'}
                </p>
                <p className={styles.modelPath} title={m.path}>
                  {m.path}
                </p>
              </div>
            </div>
          </article>
        ))}
      </section>
    </div>
  );
};
