import React, { useCallback, useEffect, useState } from 'react';
import { AlertTriangle, FileText, Plus, Trash2 } from 'lucide-react';
import { Button, Spinner } from '../ui';
import {
  notebookResearchService,
  type EvidenceManifest,
  type Note,
} from '../../services/notebookResearch.service';
import styles from './workspace.module.css';

/**
 * What the notebook has been asked to keep: notes, and generated reports.
 *
 * ## Three kinds of text, labelled
 *
 * A note a person typed, an answer saved with its citations, and a generated
 * summary or comparison all look like prose. The badge says which, because they
 * carry different weight: only the last two were built from passages, and only
 * they have a manifest to check.
 *
 * ## Editing does not re-verify
 *
 * A saved answer whose text has been edited shows "edited since saved" and keeps
 * its original manifest. The backend decides this by comparing the body against
 * the hash recorded when it was saved — the interface does not guess, and
 * nothing here re-stamps an edited answer as verified. See
 * `knowledge::graph::notes`.
 */
interface NotesPanelProps {
  notebookId: string;
  /** Raised so the workspace can put the prompt into the notebook's chat. */
  onGenerateReport: (prompt: string, kind: 'summary' | 'comparison') => void;
  /** Names of the sources currently selected, for the comparison prompt. */
  selectedSourceNames: string[];
  /** Bumped by the workspace when a note is saved elsewhere. */
  reloadToken: number;
  /** Opens a source at a page, when a note's evidence is clicked. */
  onOpenEvidence: (documentSha256: string, page: number) => void;
}

const KIND_LABEL: Record<Note['kind'], string> = {
  written: 'Written',
  answer: 'Saved answer',
  summary: 'Summary',
  comparison: 'Comparison',
};

/** The manifest a note carries, when it carries one that can still be read. */
function manifestOf(note: Note): EvidenceManifest | null {
  if (!note.evidenceJson) return null;
  try {
    return JSON.parse(note.evidenceJson) as EvidenceManifest;
  } catch {
    // A manifest written by an older build, or corrupted. The note is still
    // shown; only its evidence list is not.
    return null;
  }
}

export const NotesPanel: React.FC<NotesPanelProps> = ({
  notebookId,
  onGenerateReport,
  selectedSourceNames,
  reloadToken,
  onOpenEvidence,
}) => {
  const [notes, setNotes] = useState<Note[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openId, setOpenId] = useState<string | null>(null);
  const [draft, setDraft] = useState<{ title: string; body: string } | null>(null);
  const [saving, setSaving] = useState(false);

  const reload = useCallback(async () => {
    setError(null);
    try {
      setNotes(await notebookResearchService.notes(notebookId));
    } catch (err) {
      setError(String(err));
    } finally {
      setLoading(false);
    }
  }, [notebookId]);

  useEffect(() => {
    setLoading(true);
    setOpenId(null);
    setDraft(null);
    void reload();
  }, [reload, reloadToken]);

  const open = notes.find((note) => note.id === openId) ?? null;

  const create = useCallback(async () => {
    setError(null);
    try {
      const note = await notebookResearchService.createNote(notebookId, 'Untitled note');
      await reload();
      setOpenId(note.id);
      setDraft({ title: note.title, body: note.body });
    } catch (err) {
      setError(String(err));
    }
  }, [notebookId, reload]);

  const save = useCallback(async () => {
    if (!open || !draft) return;
    setSaving(true);
    setError(null);
    try {
      await notebookResearchService.updateNote(notebookId, open.id, {
        title: draft.title,
        body: draft.body,
      });
      await reload();
    } catch (err) {
      setError(String(err));
    } finally {
      setSaving(false);
    }
  }, [draft, notebookId, open, reload]);

  const remove = useCallback(
    async (note: Note) => {
      if (!window.confirm(`Delete the note "${note.title}"?`)) return;
      setError(null);
      try {
        await notebookResearchService.deleteNote(notebookId, note.id);
        if (openId === note.id) {
          setOpenId(null);
          setDraft(null);
        }
        await reload();
      } catch (err) {
        setError(String(err));
      }
    },
    [notebookId, openId, reload],
  );

  return (
    <section className={styles.notes} aria-label="Notes and reports">
      <header className={styles.notesHead}>
        <span className={styles.sectionLabel}>Notes &amp; reports</span>
        <div className={styles.notesActions}>
          <Button size="sm" variant="ghost" onClick={() => void create()}>
            <Plus size={13} /> Note
          </Button>
          <Button
            size="sm"
            variant="ghost"
            onClick={() =>
              onGenerateReport(
                'Write a summary of the selected sources. Cover what each one is about ' +
                  'and what they agree and disagree on. Cite every claim to the passage ' +
                  'it came from, and say plainly where the sources are silent.',
                'summary',
              )
            }
          >
            Summarise
          </Button>
          <Button
            size="sm"
            variant="ghost"
            disabled={selectedSourceNames.length < 2}
            title={
              selectedSourceNames.length < 2
                ? 'Select at least two sources to compare them'
                : `Compare ${selectedSourceNames.join(', ')}`
            }
            onClick={() =>
              onGenerateReport(
                `Compare these sources: ${selectedSourceNames.join(', ')}. Set out what ` +
                  'they each say on the points they have in common, where they differ, ' +
                  'and what is claimed in one and absent from the others. Cite every ' +
                  'point to the passage it came from.',
                'comparison',
              )
            }
          >
            Compare
          </Button>
        </div>
      </header>

      {error && (
        <p className={styles.error}>
          <AlertTriangle size={13} /> {error}
        </p>
      )}

      {loading ? (
        <div className={styles.readerLoading}>
          <Spinner />
        </div>
      ) : notes.length === 0 ? (
        <div className={styles.empty}>
          <FileText size={22} />
          <p>Nothing kept yet.</p>
          <p className={styles.emptyHint}>
            Save an answer from the chat, write a note, or generate a summary. Saved
            answers keep the passages they were built from.
          </p>
        </div>
      ) : (
        <div className={styles.notesLayout}>
          <ul className={styles.noteList}>
            {notes.map((note) => (
              <li key={note.id}>
                <button
                  type="button"
                  className={styles.noteRow}
                  data-open={note.id === openId || undefined}
                  onClick={() => {
                    setOpenId(note.id);
                    setDraft({ title: note.title, body: note.body });
                  }}
                >
                  <span className={styles.noteTitle}>{note.title}</span>
                  <span className={styles.noteMeta}>
                    {KIND_LABEL[note.kind]}
                    {note.editedSinceSaved && ' · edited'}
                  </span>
                </button>
                <button
                  type="button"
                  className={styles.sourceRemove}
                  aria-label={`Delete ${note.title}`}
                  onClick={() => void remove(note)}
                >
                  <Trash2 size={12} />
                </button>
              </li>
            ))}
          </ul>

          {open && draft && (
            <div className={styles.noteEditor}>
              <input
                className={styles.noteTitleInput}
                value={draft.title}
                onChange={(event) =>
                  setDraft((current) =>
                    current ? { ...current, title: event.target.value } : current,
                  )
                }
                aria-label="Note title"
              />

              <div className={styles.noteBadges}>
                <span className={styles.scopeChip}>{KIND_LABEL[open.kind]}</span>
                {open.kind !== 'written' && !open.editedSinceSaved && (
                  <span className={styles.scopeChip}>as generated</span>
                )}
                {open.editedSinceSaved && (
                  <span className={styles.scopeProblem}>
                    <AlertTriangle size={12} /> edited since it was saved — the changes are
                    yours, not the sources&rsquo;
                  </span>
                )}
              </div>

              <textarea
                className={styles.noteBody}
                value={draft.body}
                onChange={(event) =>
                  setDraft((current) =>
                    current ? { ...current, body: event.target.value } : current,
                  )
                }
                aria-label="Note body"
              />

              <div className={styles.noteEditorFoot}>
                <Button size="sm" onClick={() => void save()} loading={saving}>
                  Save
                </Button>
                {open.updatedAt && (
                  <span className={styles.answerMeta}>
                    last saved {new Date(open.updatedAt).toLocaleString()}
                  </span>
                )}
              </div>

              {(() => {
                const manifest = manifestOf(open);
                if (!manifest || manifest.entries.length === 0) return null;
                return (
                  <div className={styles.noteEvidence}>
                    <span className={styles.sectionLabel}>
                      Evidence this was built on ({manifest.retrievalMode} search)
                    </span>
                    <ul className={styles.evidenceList}>
                      {manifest.entries.map((entry) => (
                        <li key={entry.chunkId}>
                          <button
                            type="button"
                            className={styles.evidenceRow}
                            onClick={() => onOpenEvidence(entry.documentSha256, entry.page)}
                          >
                            <span className={styles.citation}>E{entry.marker}</span>
                            <span className={styles.evidenceName}>
                              {entry.documentName}, page {entry.page}
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                    {manifest.limitations.length > 0 && (
                      <ul className={styles.limitations}>
                        {manifest.limitations.map((line) => (
                          <li key={line}>{line}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
    </section>
  );
};
