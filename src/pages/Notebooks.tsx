import React, { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { NotebookPen, Plus, FileText, AlertTriangle, RefreshCw } from 'lucide-react';
import { Button, Spinner } from '../components/ui';
import {
  notebookService,
  renderSubgraph,
  type AddedDocument,
  type GraphNode,
  type Notebook,
  type NotebookDocument,
} from '../services/notebook.service';
import { useConversation } from '../contexts/ConversationContext';
import { toComposerAttachment } from '../services/agent.service';
import { sovereigntyService } from '../services/sovereignty.service';
import { problemRows, summariseAdds } from '../components/notebook/addOutcomes';
import { NotebookGraphPanel } from '../components/graph/NotebookGraphPanel';
import styles from './Notebooks.module.css';

/**
 * Notebooks — named libraries of documents that outlive a conversation.
 *
 * A conversation's attachments vanish with it, so nothing in ARJUN could hold
 * "these forty documents belong together". A notebook is that container: it
 * stores no bytes, only the content addresses of documents read through the
 * same pipeline a chat attachment goes through.
 *
 * The screen is deliberately plain about outcomes. Adding files reports one row
 * per file requested — added, already present, or failed with the reason —
 * because a document silently missing from a library is the failure that costs
 * somebody an afternoon.
 */

/** What the file picker will offer, matching the reader's own list. */
const ACCEPT =
  '.png,.jpg,.jpeg,.webp,.pdf,.txt,.md,.markdown,.csv,.json,.log,.tsv,.docx,.xlsx,.pptx';

/**
 * Base64 for arbitrary bytes.
 *
 * `btoa` takes a Latin-1 string and throws on anything outside it, and the
 * rendered graph is full of em-dashes and curly quotes. Chunked so a large
 * selection cannot blow the argument limit of `String.fromCharCode`.
 */
function bytesToBase64(bytes: Uint8Array): string {
  let binary = '';
  const CHUNK = 0x8000;
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode(...bytes.subarray(i, i + CHUNK));
  }
  return btoa(binary);
}

/** A short, local date. Notebooks are per-machine, so the machine's zone is right. */
function shortDate(iso: string): string {
  const at = new Date(iso);
  return Number.isNaN(at.getTime()) ? iso : at.toLocaleDateString();
}

export const Notebooks: React.FC = () => {
  const [notebooks, setNotebooks] = useState<Notebook[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [documents, setDocuments] = useState<NotebookDocument[]>([]);
  /**
   * Which file the graph is narrowed to, or null for the whole notebook.
   *
   * Held here rather than inside the panel because the control that sets it is
   * the sidebar file list, which lives here.
   */
  const [scopedDocument, setScopedDocument] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [creating, setCreating] = useState(false);
  const [newName, setNewName] = useState('');

  const [adding, setAdding] = useState(false);
  const [outcomes, setOutcomes] = useState<AddedDocument[] | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    setError(null);
    try {
      setNotebooks(await notebookService.list());
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // The document list belongs to the selection, so it is re-read whenever the
  // selection changes rather than cached per notebook — a stale list here would
  // claim a document is in a notebook it was removed from.
  useEffect(() => {
    // A scope belongs to the notebook it was chosen in. Carrying it across
    // would ask the new notebook for a file it does not contain, and an empty
    // graph reads as "nothing was extracted" rather than "wrong file".
    setScopedDocument(null);
    if (!selectedId) {
      setDocuments([]);
      return;
    }
    let live = true;
    void notebookService
      .documents(selectedId)
      .then((rows) => {
        if (live) setDocuments(rows);
      })
      .catch((err) => {
        if (live) setError(String(err));
      });
    return () => {
      live = false;
    };
  }, [selectedId]);

  const create = useCallback(async () => {
    const name = newName.trim();
    if (!name) return;
    setError(null);
    try {
      const notebook = await notebookService.create(name);
      setNewName('');
      setCreating(false);
      setNotebooks((current) => [notebook, ...current]);
      setSelectedId(notebook.id);
    } catch (err) {
      setError(String(err));
    }
  }, [newName]);

  /**
   * Renames the open notebook.
   *
   * The prompt is the browser's, deliberately: this is a one-field edit on a
   * screen that has no modal of its own, and inventing one here would be a
   * component to maintain for a rename.
   */
  const rename = useCallback(async (notebook: Notebook) => {
    const next = window.prompt('Rename this notebook to:', notebook.name);
    if (next === null) return;
    const name = next.trim();
    if (!name || name === notebook.name) return;
    setError(null);
    try {
      const renamed = await notebookService.rename(notebook.id, name);
      setNotebooks((current) => current.map((row) => (row.id === renamed.id ? renamed : row)));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  /**
   * Deletes a notebook, after saying what goes with it.
   *
   * The confirmation names the graph rather than asking "are you sure": the
   * documents survive and the graph does not, and that asymmetry is the thing
   * somebody needs to know before answering.
   */
  const remove = useCallback(async (notebook: Notebook) => {
    const agreed = window.confirm(
      `Delete the notebook "${notebook.name}"?\n\n` +
        'The knowledge graph built over it is deleted too, and rebuilding it means ' +
        `running the extraction passes again. Its ${notebook.documentCount} document(s) ` +
        'are not deleted and stay where they are.',
    );
    if (!agreed) return;
    setError(null);
    try {
      await notebookService.delete(notebook.id);
      setNotebooks((current) => current.filter((row) => row.id !== notebook.id));
      setSelectedId((current) => (current === notebook.id ? null : current));
    } catch (err) {
      setError(String(err));
    }
  }, []);

  /** Takes one source out of the open notebook. */
  const removeSource = useCallback(
    async (notebookId: string, document: NotebookDocument) => {
      const agreed = window.confirm(
        `Take "${document.documentName}" out of this notebook?\n\n` +
          'The terms it was the only source for stop being cited. The file itself is ' +
          'not deleted.',
      );
      if (!agreed) return;
      setError(null);
      try {
        await notebookService.removeDocument(notebookId, document.documentSha256);
        setDocuments(await notebookService.documents(notebookId));
        setNotebooks(await notebookService.list());
        // A scope pointing at a file that is no longer here would filter the
        // graph to nothing and give no clue why.
        setScopedDocument((current) =>
          current === document.documentSha256 ? null : current,
        );
      } catch (err) {
        setError(String(err));
      }
    },
    [],
  );

  const onFilesPicked = useCallback(
    async (event: React.ChangeEvent<HTMLInputElement>) => {
      const files = Array.from(event.target.files ?? []);
      event.target.value = '';
      if (files.length === 0 || !selectedId) return;

      // The same refusal the composer makes. Reading a confidential document
      // while the machine is allowed to reach the network is the thing this
      // product exists to prevent, and a second door into the same pipeline
      // must not be a way around it.
      try {
        await sovereigntyService.assertConfidentialAllowed('adding a document to a notebook');
      } catch (err) {
        setError(String(err));
        return;
      }

      setAdding(true);
      setOutcomes(null);
      setError(null);
      try {
        const attachments = await Promise.all(files.map(toComposerAttachment));
        const result = await notebookService.addDocuments(selectedId, attachments);
        setOutcomes(result);
        setDocuments(await notebookService.documents(selectedId));
        setNotebooks(await notebookService.list());
      } catch (err) {
        setError(String(err));
      } finally {
        setAdding(false);
      }
    },
    [selectedId],
  );

  const navigate = useNavigate();
  const { send } = useConversation();
  const [importing, setImporting] = useState(false);

  /**
   * Takes the chosen terms into a chat turn.
   *
   * The subgraph is rendered to Markdown by the backend and handed to `send` as
   * an ordinary attachment. Nothing about the context pipeline needed changing:
   * the attachment path already content-addresses it, chunks it, budgets it,
   * shows it in the context meter and lets it be pinned — so the imported graph
   * behaves exactly like any other document the model was given.
   */
  const importSelection = useCallback(
    async (nodes: GraphNode[]) => {
      if (!selectedId || nodes.length === 0) return;
      setImporting(true);
      setError(null);
      try {
        const rendered = await renderSubgraph(
          selectedId,
          nodes.map((node) => node.id),
        );
        const labels = nodes.map((node) => node.label);
        const named =
          labels.length <= 4
            ? labels.join(', ')
            : `${labels.slice(0, 4).join(', ')} and ${labels.length - 4} more`;

        await send(
          `Using the attached selection from my notebook, what connects ${named}? Cite the documents.`,
          undefined,
          {
            attachments: [
              {
                name: rendered.name,
                mime: 'text/markdown',
                // The markdown is UTF-8 and may contain non-Latin-1 characters,
                // which `btoa` cannot encode. Going through the byte encoder
                // first is what stops an em-dash from throwing here.
                dataBase64: bytesToBase64(new TextEncoder().encode(rendered.markdown)),
              },
            ],
          },
        );
        navigate('/');
      } catch (err) {
        setError(String(err));
      } finally {
        setImporting(false);
      }
    },
    [selectedId, send, navigate],
  );

  const selected = notebooks.find((n) => n.id === selectedId) ?? null;

  if (loading) {
    return (
      <div className={styles.centered}>
        <Spinner />
      </div>
    );
  }

  return (
    <div className={styles.page}>
      <aside className={styles.sidebar}>
        <div className={styles.sidebarHead}>
          <span className={styles.sidebarTitle}>Notebooks</span>
          <Button size="sm" variant="ghost" icon onClick={() => setCreating((c) => !c)}>
            <Plus size={15} />
          </Button>
        </div>

        {creating && (
          <div className={styles.createRow}>
            <input
              className={styles.nameInput}
              autoFocus
              value={newName}
              placeholder="Notebook name"
              onChange={(e) => setNewName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') void create();
                if (e.key === 'Escape') setCreating(false);
              }}
            />
            <Button size="sm" onClick={() => void create()} disabled={!newName.trim()}>
              Create
            </Button>
          </div>
        )}

        {notebooks.length === 0 ? (
          <p className={styles.empty}>
            No notebooks yet. A notebook holds documents that stay together across
            conversations.
          </p>
        ) : (
          <ul className={styles.list}>
            {notebooks.map((notebook) => (
              <li key={notebook.id}>
                <button
                  type="button"
                  className={styles.listItem}
                  data-selected={notebook.id === selectedId}
                  onClick={() => setSelectedId(notebook.id)}
                >
                  <span className={styles.listName}>{notebook.name}</span>
                  <span className={styles.listMeta}>
                    {notebook.documentCount}{' '}
                    {notebook.documentCount === 1 ? 'document' : 'documents'}
                  </span>
                </button>

                {/* Only on the open notebook. A row of controls on every row
                    turns a list of names into a list of buttons, and the one
                    being worked on is the only one they apply to. */}
                {notebook.id === selectedId && (
                  <div className={styles.listActions}>
                    <button
                      type="button"
                      className={styles.listAction}
                      onClick={() => void rename(notebook)}
                    >
                      Rename
                    </button>
                    <button
                      type="button"
                      className={styles.listAction}
                      onClick={() => void remove(notebook)}
                    >
                      Delete
                    </button>
                  </div>
                )}

                {/* The files live here, under the notebook they belong to, so
                    the middle of the screen can be the graph and nothing else.
                    Only the open notebook lists them: fetching the documents of
                    notebooks nobody has opened would be a query apiece to show
                    something nobody asked to see. */}
                {notebook.id === selectedId && documents.length > 0 && (
                  <ul className={styles.sidebarDocuments}>
                    {documents.map((document) => (
                      <li key={document.documentSha256}>
                        <button
                          type="button"
                          className={
                            scopedDocument === document.documentSha256
                              ? `${styles.sidebarDocument} ${styles.sidebarDocumentActive}`
                              : styles.sidebarDocument
                          }
                          title={`${document.documentName} · added ${shortDate(document.addedAt)}`}
                          onClick={() =>
                            // Clicking the open file again clears the scope, so
                            // the way back to the whole notebook is the control
                            // that got you here rather than one somewhere else.
                            setScopedDocument((current) =>
                              current === document.documentSha256
                                ? null
                                : document.documentSha256,
                            )
                          }
                        >
                          <FileText size={13} className={styles.documentIcon} />
                          <span className={styles.sidebarDocumentName}>
                            {document.documentName}
                          </span>
                        </button>
                        <button
                          type="button"
                          className={styles.sidebarDocumentRemove}
                          title={`Take ${document.documentName} out of this notebook`}
                          aria-label={`Take ${document.documentName} out of this notebook`}
                          onClick={() => void removeSource(notebook.id, document)}
                        >
                          &times;
                        </button>
                      </li>
                    ))}
                  </ul>
                )}

                {notebook.id === selectedId && documents.length === 0 && (
                  <p className={styles.sidebarEmpty}>Nothing in it yet.</p>
                )}
              </li>
            ))}
          </ul>
        )}
      </aside>

      <section className={styles.detail}>
        {error && (
          <p className={styles.error}>
            <AlertTriangle size={14} /> {error}
          </p>
        )}

        {!selected ? (
          <div className={styles.placeholder}>
            <NotebookPen size={28} />
            <p>Select a notebook, or create one.</p>
          </div>
        ) : (
          <>
            <header className={styles.detailHead}>
              <div>
                <h2 className={styles.detailTitle}>{selected.name}</h2>
                <p className={styles.detailMeta}>
                  {selected.documentCount}{' '}
                  {selected.documentCount === 1 ? 'document' : 'documents'} · created{' '}
                  {shortDate(selected.createdAt)}
                </p>
              </div>
              <div className={styles.detailActions}>
                <Button size="sm" variant="ghost" icon onClick={() => void refresh()}>
                  <RefreshCw size={14} />
                </Button>
                <Button
                  size="sm"
                  onClick={() => fileInput.current?.click()}
                  loading={adding}
                  disabled={adding}
                >
                  Add documents
                </Button>
              </div>
            </header>

            <input
              ref={fileInput}
              type="file"
              multiple
              accept={ACCEPT}
              className={styles.hiddenInput}
              onChange={(e) => void onFilesPicked(e)}
            />

            {adding && (
              <p className={styles.working}>
                Reading the files. A scanned page goes through OCR, so this can take a
                while.
              </p>
            )}

            {outcomes && (
              <div className={styles.outcomes}>
                <p className={styles.outcomeSummary}>{summariseAdds(outcomes)}</p>
                {problemRows(outcomes).map((outcome) => (
                  <p key={outcome.name} className={styles.outcomeProblem}>
                    <span className={styles.outcomeName}>{outcome.name}</span>
                    {outcome.problem}
                  </p>
                ))}
              </div>
            )}

            {documents.length === 0 && (
              <p className={styles.empty}>
                Nothing in this notebook yet. Add documents to build it up.
              </p>
            )}

            <NotebookGraphPanel
              notebookId={selected.id}
              documentCount={selected.documentCount}
              documentSha256={scopedDocument}
              onImport={(nodes) => void importSelection(nodes)}
              importing={importing}
            />
          </>
        )}
      </section>
    </div>
  );
};

export default Notebooks;
