import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, FolderSync, Gauge, Trash2 } from 'lucide-react';
import { Button } from '../components/ui';
import {
  retrievalService,
  type Collection,
  type CollectionView,
  type RetrievalStatus,
} from '../services/retrieval.service';
import styles from './Knowledge.module.css';

const STATE_LABEL: Record<string, string> = {
  qualified: 'Keyword + semantic',
  unqualified: 'Keyword (semantic model not qualified)',
  unavailable: 'Keyword (no embedding model)',
};

const EMPTY: Collection = {
  id: '',
  name: '',
  kind: 'localFolder',
  root: '',
  owner: '',
  classification: 'internal',
  restrictedToRoles: [],
  retentionDays: null,
  enabled: true,
};

/**
 * The knowledge connector and the retrieval it feeds (plan P07).
 *
 * Collections are folders or shares this machine reads; a sync takes their
 * documents in through the document reader, versions them, withdraws what
 * disappeared and tells the memory graph. The retrieval line says which half of
 * search actually runs here — keyword always, semantic only once the embedding
 * model on this machine has been measured on labelled probes — and why.
 */
export const KnowledgeCollectionsPanel: React.FC<{ onChanged?: () => void }> = ({ onChanged }) => {
  const [collections, setCollections] = useState<CollectionView[]>([]);
  const [status, setStatus] = useState<RetrievalStatus | null>(null);
  const [draft, setDraft] = useState<Collection>(EMPTY);
  const [busy, setBusy] = useState<string | null>(null);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    try {
      const [list, state] = await Promise.all([
        retrievalService.collections(),
        retrievalService.status().catch(() => null),
      ]);
      setCollections(list);
      setStatus(state);
      setError(null);
    } catch (problem) {
      setError(String(problem));
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const act = async (key: string, work: () => Promise<string>) => {
    setBusy(key);
    setMessage(null);
    try {
      setMessage(await work());
      onChanged?.();
    } catch (problem) {
      setMessage(String(problem));
    } finally {
      setBusy(null);
      void load();
    }
  };

  const save = (event: React.FormEvent) => {
    event.preventDefault();
    void act('save', async () => {
      await retrievalService.save({ ...draft, id: draft.id.trim(), name: draft.name.trim(), root: draft.root.trim() });
      setDraft(EMPTY);
      return `Saved ${draft.name}. Sync it to read its documents.`;
    });
  };

  const provider = status?.provider;
  const coverage = status?.coverage;

  return (
    <>
      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <Gauge size={15} /> Retrieval on this machine
        </h2>
        {provider ? (
          <>
            <p className={styles.sectionNote}>
              <strong>{STATE_LABEL[provider.state] ?? provider.state}.</strong> {provider.detail}
            </p>
            {provider.identity && (
              <p className={styles.sectionNote}>
                Model {provider.identity.modelId} · profile {provider.identity.profile} ·{' '}
                {provider.identity.dimensions} dimensions · {provider.identity.pooling} pooling · weights{' '}
                <span title={provider.identity.weightsSha256}>{provider.identity.weightsSha256.slice(0, 12)}…</span>
              </p>
            )}
            {coverage && (
              <p className={styles.sectionNote}>
                Semantic coverage of the passages you can read: {coverage.embedded} of {coverage.total}
                {coverage.failed > 0 && ` (${coverage.failed} could not be embedded)`}
                {status?.reindexing && ' · an embedding pass is running'}.
              </p>
            )}
            <div className={styles.searchRow}>
              {provider.state !== 'unavailable' && (
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy !== null}
                  onClick={() =>
                    void act('qualify', async () => {
                      const record = await retrievalService.qualify();
                      const failed = record.checks.filter((check) => !check.passed);
                      return record.passed
                        ? `Qualified: ${record.orderedCorrectly} of ${record.probes} labelled probes ordered correctly.`
                        : `Not qualified: ${failed.map((check) => `${check.name} — ${check.detail}`).join('; ')}`;
                    })
                  }
                >
                  {busy === 'qualify' ? 'Measuring…' : 'Qualify the embedding model'}
                </Button>
              )}
              {provider.state === 'qualified' && (
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy !== null || status?.reindexing}
                  onClick={() =>
                    void act('reindex', async () => {
                      const report = await retrievalService.reindex();
                      return `Embedded ${report.embedded}, failed ${report.failed}, ${report.remaining} remaining.`;
                    })
                  }
                >
                  {busy === 'reindex' ? 'Embedding…' : 'Run the embedding pass'}
                </Button>
              )}
            </div>
          </>
        ) : (
          <p className={styles.empty}>The retrieval status could not be read.</p>
        )}
      </section>

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <FolderSync size={15} /> Collections
        </h2>
        <p className={styles.sectionNote}>
          Folders and shares this machine reads, read-only. A sync indexes new and changed files under the
          collection's classification, supersedes old versions and withdraws files that are gone.
        </p>
        {error && (
          <p className={styles.error}>
            <AlertTriangle size={14} /> {error}
          </p>
        )}
        {message && <p className={styles.note}>{message}</p>}
        {collections.length === 0 ? (
          <p className={styles.empty}>No collection is defined yet.</p>
        ) : (
          <ul className={styles.documents}>
            {collections.map((collection) => (
              <li key={collection.id} className={styles.document}>
                <div className={styles.documentBody}>
                  <span className={styles.documentName}>
                    {collection.name} {!collection.enabled && '(disabled)'}
                  </span>
                  <span className={styles.documentMeta}>
                    {collection.root} · {collection.currentSources} current source(s) · last synced{' '}
                    {collection.lastSyncedAt ?? 'never'}
                    {collection.restrictedToRoles.length > 0 && ` · only ${collection.restrictedToRoles.join(', ')}`}
                  </span>
                </div>
                <Button
                  size="sm"
                  variant="ghost"
                  disabled={busy !== null}
                  onClick={() =>
                    void act(`sync-${collection.id}`, async () => {
                      const report = await retrievalService.sync(collection.id);
                      const parts = [
                        report.plan,
                        `${report.outcome.documentsRead} read, ${report.outcome.chunksIndexed} passage(s) indexed`,
                      ];
                      if (report.outcome.supersededSha256s.length) parts.push(`${report.outcome.supersededSha256s.length} superseded`);
                      if (report.outcome.withdrawnSha256s.length) parts.push(`${report.outcome.withdrawnSha256s.length} withdrawn`);
                      if (report.outcome.failures.length) parts.push(`${report.outcome.failures.length} failed`);
                      if (report.outcome.pagesNeedingReview) parts.push(`${report.outcome.pagesNeedingReview} page(s) need review`);
                      if (report.memoryInvalidated || report.memoryStale)
                        parts.push(`${report.memoryInvalidated} memory item(s) invalidated, ${report.memoryStale} stale`);
                      return parts.join(' · ');
                    })
                  }
                >
                  {busy === `sync-${collection.id}` ? 'Syncing…' : 'Sync'}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  aria-label={`Remove ${collection.name}`}
                  disabled={busy !== null}
                  onClick={() =>
                    void act(`remove-${collection.id}`, async () => {
                      const withdrawn = await retrievalService.remove(collection.id);
                      return `Removed ${collection.name}; ${withdrawn} source(s) withdrawn and kept traceable.`;
                    })
                  }
                >
                  <Trash2 size={14} />
                </Button>
              </li>
            ))}
          </ul>
        )}

        <form className={styles.searchRow} onSubmit={save}>
          <input
            className={styles.searchInput}
            value={draft.id}
            onChange={(event) => setDraft({ ...draft, id: event.target.value })}
            placeholder="id, e.g. sops"
            aria-label="Collection id"
          />
          <input
            className={styles.searchInput}
            value={draft.name}
            onChange={(event) => setDraft({ ...draft, name: event.target.value, owner: draft.owner || event.target.value })}
            placeholder="Maintenance SOPs"
            aria-label="Collection name"
          />
          <input
            className={styles.searchInput}
            value={draft.root}
            onChange={(event) => setDraft({ ...draft, root: event.target.value })}
            placeholder={'D:\\plant\\sops or \\\\server\\share'}
            aria-label="Collection folder"
          />
          <Button type="submit" size="sm" disabled={busy !== null || !draft.id.trim() || !draft.name.trim() || !draft.root.trim()}>
            Add
          </Button>
        </form>
      </section>
    </>
  );
};
