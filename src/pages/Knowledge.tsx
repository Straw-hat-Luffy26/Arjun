import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { Library, RefreshCw, Search, AlertTriangle, FileText } from 'lucide-react';
import { Button, Spinner } from '../components/ui';
import {
  knowledgeService,
  type IndexedDocument,
  type KnowledgeHealth,
  type SearchResult,
} from '../services/knowledge.service';
import { KnowledgeCollectionsPanel } from './KnowledgeCollectionsPanel';
import { retrievalService, type RetrievalStatus } from '../services/retrieval.service';
import styles from './Knowledge.module.css';

/** Classification labels, as an operator reads them rather than as JSON spells them. */
const CLASSIFICATION_LABEL: Record<string, string> = {
  internal: 'Internal',
  processDiagram: 'Process diagram',
  financial: 'Financial',
  vendorNegotiation: 'Vendor negotiation',
  unreleasedDesign: 'Unreleased design',
  internalCorrespondence: 'Internal correspondence',
  businessStrategy: 'Business strategy',
};

const label = (value: string) => CLASSIFICATION_LABEL[value] ?? value;

/**
 * Knowledge — what this machine has indexed, and what a run would retrieve.
 *
 * Two things, deliberately together. The list says which documents are in the
 * index; the search box runs the *same* query the agent's
 * `knowledge.search_authorized` tool runs, so somebody wondering why a run said
 * "no source was found" can ask the index directly and see what the model saw.
 * A screen that approximated retrieval instead would be worse than none: it
 * would explain a run that never happened.
 *
 * Everything here is filtered by clearance inside the SQL, so what is missing
 * from this page is missing for the same reason it is missing from a run.
 */
export const Knowledge: React.FC = () => {
  const [documents, setDocuments] = useState<IndexedDocument[]>([]);
  const [health, setHealth] = useState<KnowledgeHealth | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [searching, setSearching] = useState(false);
  const [searchError, setSearchError] = useState<string | null>(null);
  const [retrieval, setRetrieval] = useState<RetrievalStatus | null>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      const [docs, state, mode] = await Promise.all([
        knowledgeService.documents(),
        knowledgeService.health().catch(() => null),
        retrievalService.status().catch(() => null),
      ]);
      setDocuments(docs);
      setHealth(state);
      setRetrieval(mode);
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runSearch = async (event: React.FormEvent) => {
    event.preventDefault();
    const trimmed = query.trim();
    if (!trimmed) {
      setResults(null);
      return;
    }
    setSearching(true);
    setSearchError(null);
    try {
      setResults(await knowledgeService.search(trimmed));
    } catch (err) {
      setSearchError(String(err));
      setResults(null);
    } finally {
      setSearching(false);
    }
  };

  // The gap between what the index holds and what this person may see. Shown
  // only when there is one: where somebody is cleared for everything the
  // sentence is noise, and where they are not it is the answer to "why is this
  // list short?".
  const withheld = useMemo(() => {
    if (!health) return 0;
    return Math.max(0, health.documents - health.visibleDocuments);
  }, [health]);

  if (loading) {
    return (
      <div className={styles.centered}>
        <Spinner />
        <p>Reading the index…</p>
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <div>
          <h1 className={styles.title}>Knowledge</h1>
          <p className={styles.subtitle}>
            The documents indexed on this machine. Nothing here has left it.
          </p>
        </div>
        <Button variant="ghost" size="sm" onClick={() => void refresh()}>
          <RefreshCw size={14} /> Refresh
        </Button>
      </header>

      {error && (
        <p className={styles.error}>
          <AlertTriangle size={14} /> {error}
        </p>
      )}

      {health && !health.readable && (
        <p className={styles.error}>
          <AlertTriangle size={14} /> The index could not be read. That is not the same as an
          empty index — the documents may be there and unreachable.
        </p>
      )}

      <div className={styles.stats}>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Documents you can search</span>
          <span className={styles.statValue}>{health?.visibleDocuments ?? documents.length}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Passages</span>
          <span className={styles.statValue}>{health?.visiblePassages ?? '—'}</span>
        </div>
        <div className={styles.stat}>
          <span className={styles.statLabel}>Retrieval</span>
          <span className={styles.statValue}>
            {retrieval?.provider.state === 'qualified' ? 'Keyword + semantic' : 'Keyword'}
          </span>
          <span className={styles.statNote}>
            {retrieval ? retrieval.provider.detail : 'The retrieval status could not be read.'}
          </span>
        </div>
      </div>

      {withheld > 0 && (
        <p className={styles.note}>
          {withheld} further {withheld === 1 ? 'document is' : 'documents are'} indexed that your
          clearance does not cover. A run of yours would not retrieve from them either.
        </p>
      )}

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <Search size={15} /> Search the index
        </h2>
        <p className={styles.sectionNote}>
          The same query an agent run makes. What comes back here is what the model would be
          given.
        </p>
        <form className={styles.searchRow} onSubmit={runSearch}>
          <input
            className={styles.searchInput}
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="A tag, a clause, a standard number — PT-2201, seal thickness, IS 15656"
            aria-label="Search the knowledge index"
          />
          <Button type="submit" size="sm" disabled={searching || !query.trim()}>
            {searching ? 'Searching…' : 'Search'}
          </Button>
        </form>

        {searchError && (
          <p className={styles.error}>
            <AlertTriangle size={14} /> {searchError}
          </p>
        )}

        {results !== null && results.length === 0 && !searching && (
          <p className={styles.empty}>
            No passage matched. Nothing in the connected collections says this — which is not the
            same as it being false. Try the specific technical term rather than a paraphrase.
          </p>
        )}

        {results !== null && results.length > 0 && (
          <ol className={styles.results}>
            {results.map((hit, index) => (
              <li key={hit.chunkId} className={styles.result}>
                <div className={styles.resultHead}>
                  <span className={styles.marker}>[E{index + 1}]</span>
                  <span className={styles.citation}>
                    {hit.documentName}
                    {hit.sectionPath.length > 0 && ` — ${hit.sectionPath.join(' › ')}`}, page{' '}
                    {hit.page}
                  </span>
                  <span className={styles.badge}>{label(hit.classification)}</span>
                </div>
                <p className={styles.passage}>{hit.text}</p>
              </li>
            ))}
          </ol>
        )}
      </section>

      <KnowledgeCollectionsPanel onChanged={() => void refresh()} />

      <section className={styles.section}>
        <h2 className={styles.sectionTitle}>
          <Library size={15} /> Indexed documents
        </h2>

        {documents.length === 0 ? (
          <p className={styles.empty}>
            Nothing is indexed yet. Documents enter the index by syncing a collection above, which
            reads them under the collection's classification — there is deliberately no way to add
            a single file from this screen, because material that skipped classification would be
            retrievable by a run that is not cleared for it.
          </p>
        ) : (
          <ul className={styles.documents}>
            {documents.map((doc) => (
              <li key={doc.documentSha256} className={styles.document}>
                <FileText size={14} className={styles.documentIcon} aria-hidden />
                <div className={styles.documentBody}>
                  <span className={styles.documentName}>{doc.documentName}</span>
                  <span className={styles.documentMeta}>
                    {doc.chunks} {doc.chunks === 1 ? 'passage' : 'passages'} · {doc.pages}{' '}
                    {doc.pages === 1 ? 'page' : 'pages'} ·{' '}
                    <span title={doc.documentSha256}>{doc.documentSha256.slice(0, 8)}</span>
                  </span>
                </div>
                <span className={styles.badge}>{label(doc.classification)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
};
