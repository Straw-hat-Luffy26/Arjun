import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, BookOpen, Check, Loader2, RefreshCw, X } from 'lucide-react';
import { notebookService, type Notebook } from '../../services/notebook.service';
import {
  inProgress,
  notebookResearchService,
  usableAsEvidence,
  type ReadinessReason,
  type SourceReadiness,
  type SourceState,
} from '../../services/notebookResearch.service';
import {
  allSources,
  describeSelection,
  noSources,
  someSources,
} from '../../services/agent.service';
import { filterNotebooks, type NotebookScope } from './notebookScope';
import styles from './ChatSurface.module.css';

/**
 * The two controls `/notebook` puts on the main composer.
 *
 * [`NotebookChooser`] is the searchable list that opens directly above the
 * input while the command is being typed. [`NotebookChip`] is what remains
 * afterwards: the notebook the next question is scoped to, which of its
 * sources are included, and the way back out.
 *
 * ## Why the chooser does not own focus
 *
 * It never takes focus from the textarea. Arrow keys, Enter and Escape are
 * handled by the composer and passed down as props, so the draft stays live
 * underneath and the caret never moves. A chooser with its own search box
 * would have been easier to write and would have meant tabbing away from a
 * half-written question in order to use it.
 *
 * ## Readiness is shown before the question, not after the answer
 *
 * Each row says how many of a notebook's sources can actually be read.
 * "8 sources" and "8 sources, 3 of which have no text" are different notebooks
 * to ask a question of, and the moment to learn which one you have is before
 * spending two minutes on an answer.
 */

/** What a row needs, once readiness has been asked for. */
interface Row {
  notebook: Notebook;
  /** `undefined` until readiness arrives; `null` when it could not be read. */
  readiness?: SourceReadiness[] | null;
}

/** The one sentence a state deserves on a source row. */
function describeState(state: SourceState): string {
  switch (state) {
    case 'queued':
      return 'Waiting to be read';
    case 'reading':
      return 'Being read';
    case 'extracting':
      return 'Being extracted';
    case 'needsVision':
      return 'Needs OCR or vision';
    case 'indexing':
      return 'Being indexed';
    case 'ready':
      return 'Ready';
    case 'partiallyReady':
      return 'Partly readable';
    case 'failed':
      return 'Cannot be read';
    case 'unavailable':
      return 'Not readable on this machine';
  }
}

/**
 * The backend's own words for a reason.
 *
 * Deliberately kept in step with `ReadinessReason::describe` in Rust: the same
 * sentence appears on the Notebooks screen and in a turn's limitations, and
 * three different wordings for one fact is how a person concludes there are
 * three different problems.
 */
export function describeReason(reason: ReadinessReason): string {
  switch (reason.kind) {
    case 'notAssociated':
      return 'Its text is on this machine, but this notebook was never given access to it. Repairing takes a moment and re-reads nothing.';
    case 'missingExtraction':
      return 'Nothing was ever extracted from this file. Add it again to read it.';
    case 'storeUnavailable':
      return `This machine’s document store could not be read, so whether this source has text is not known: ${reason.problem}`;
    case 'incompatibleExtraction':
      return `Its stored text cannot be read by this version of ARJUN: ${reason.problem}`;
    case 'missingOriginal':
      return 'The original file is no longer on this machine, so it cannot be read again.';
    case 'passwordProtected':
      return 'The file is password-protected, so nothing could be opened.';
    case 'missingParser':
      return `Reading this format needs ${reason.needed}, which is not installed on this machine.`;
    case 'conversionFailure':
      return `${reason.tool} could not convert this file: ${reason.problem}`;
    case 'noTextExtracted':
      return 'The reader opened it and found no text.';
    case 'requiresVision':
      return 'Its content is images, so reading it needs the local OCR/vision model.';
    case 'visionUnavailable':
      return 'Its content is images and no vision model is loaded on this machine, so there is no local way to read it.';
    case 'unreadablePages': {
      const shown = reason.pages.slice(0, 8);
      const rest = reason.pages.length - shown.length;
      const list = rest > 0 ? `${shown.join(', ')} and ${rest} more` : shown.join(', ');
      return `${reason.pages.length} of ${reason.total} pages produced no text (page ${list}).`;
    }
    case 'partialExtraction':
      return `The reader stopped after ${reason.read} of ${reason.total}, so the rest of this file is not indexed at all.`;
    case 'notIndexed':
      return `${reason.stored} of ${reason.expected} passages were stored, so part of this source cannot be retrieved.`;
  }
}

export interface NotebookChooserProps {
  /** What has been typed after the command word. */
  query: string;
  /** Index of the highlighted row, owned by the composer's keyboard handler. */
  highlight: number;
  onHighlightChange: (index: number) => void;
  /** How many rows there are, so the composer can bound its own arrow keys. */
  onCountChange: (count: number) => void;
  /**
   * The row the composer should choose when Enter is pressed.
   *
   * Registered as a callback rather than read from props, because the composer
   * owns the key handler and this component owns the filtered list — and the
   * two must not each keep their own idea of which row is which.
   */
  onRegisterChooser: (choose: (index: number) => void) => void;
  onChoose: (scope: NotebookScope) => void;
  onDismiss: () => void;
}

/**
 * The searchable notebook list, drawn above the composer.
 *
 * Loading, empty and failed are three visibly different states. An empty list
 * because the person has no notebooks and an empty list because the query
 * matched nothing are also different, and say so — the first offers the way to
 * make one, the second names what was searched.
 */
export const NotebookChooser: React.FC<NotebookChooserProps> = ({
  query,
  highlight,
  onHighlightChange,
  onCountChange,
  onRegisterChooser,
  onChoose,
  onDismiss,
}) => {
  const [notebooks, setNotebooks] = useState<Notebook[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [readiness, setReadiness] = useState<Map<string, SourceReadiness[] | null>>(
    () => new Map(),
  );
  const listRef = useRef<HTMLUListElement>(null);

  // The list, fetched once per opening. `live` guards the late arrival of a
  // request whose chooser has already closed — writing that into state would
  // repopulate a list the person dismissed.
  useEffect(() => {
    let live = true;
    setProblem(null);
    notebookService
      .list()
      .then(found => {
        if (live) setNotebooks(found);
      })
      .catch(error => {
        if (!live) return;
        // Reported, not swallowed. A chooser that is empty because the call
        // failed looks exactly like one belonging to somebody with no
        // notebooks, and the two need different reactions.
        setNotebooks([]);
        setProblem(
          error instanceof Error ? error.message : 'Your notebooks could not be listed.',
        );
      });
    return () => {
      live = false;
    };
  }, []);

  const rows: Row[] = useMemo(() => {
    const matching = filterNotebooks(
      (notebooks ?? []).map(notebook => ({
        id: notebook.id,
        name: notebook.name,
        documentCount: notebook.documentCount,
        original: notebook,
      })),
      query,
    );
    return matching.map(entry => ({
      notebook: entry.original,
      readiness: readiness.get(entry.original.id),
    }));
  }, [notebooks, query, readiness]);

  useEffect(() => {
    onCountChange(rows.length);
  }, [rows.length, onCountChange]);

  const choose = useCallback(
    (notebook: Notebook) => {
      onChoose({
        notebookId: notebook.id,
        notebookName: notebook.name,
        documentCount: notebook.documentCount,
        // Attaching a notebook selects all of it — said explicitly, rather
        // than left as an absent value that something downstream reads as
        // "all" on its own.
        selection: allSources(),
      });
    },
    [onChoose],
  );

  // Enter, from the composer's key handler, resolved against *this* list.
  useEffect(() => {
    onRegisterChooser((index: number) => {
      const row = rows[index];
      if (row) choose(row.notebook);
    });
  }, [rows, choose, onRegisterChooser]);

  // Readiness for the rows on screen, asked for once each. Per notebook rather
  // than in one call, so a notebook whose sources cannot be read does not stop
  // every other row from showing its own count.
  // Notebooks a request is already out for. `readiness` records an id only
  // once its answer has *landed*, so without this a keystroke that re-filters
  // the list while a call is in flight fires a second call for the same
  // notebook -- harmless, and growing with typing speed.
  const asking = useRef<Set<string>>(new Set());

  useEffect(() => {
    let live = true;
    for (const row of rows) {
      const id = row.notebook.id;
      if (readiness.has(id) || asking.current.has(id)) continue;
      asking.current.add(id);
      notebookResearchService
        .sourceStatus(id)
        .then(found => {
          if (live) setReadiness(prev => new Map(prev).set(id, found));
        })
        .catch(() => {
          // Recorded as "could not be read" rather than as zero readable
          // sources: a failed status call must not make a healthy notebook
          // look broken.
          if (live) setReadiness(prev => new Map(prev).set(id, null));
        })
        .finally(() => {
          asking.current.delete(id);
        });
    }
    return () => {
      live = false;
    };
  }, [rows, readiness]);

  // Keep the highlighted row in view when the arrow keys move past the edge.
  useEffect(() => {
    const list = listRef.current;
    if (!list || highlight < 0) return;
    const row = list.children[highlight] as HTMLElement | undefined;
    row?.scrollIntoView({ block: 'nearest' });
  }, [highlight]);

  return (
    <div className={styles.notebookChooser} role="listbox" aria-label="Choose a notebook">
      <div className={styles.notebookChooserHead}>
        <BookOpen size={13} aria-hidden="true" />
        <span>Ask a notebook</span>
        <button
          type="button"
          className={styles.attachmentRemove}
          onMouseDown={event => {
            event.preventDefault();
            onDismiss();
          }}
          aria-label="Close the notebook chooser"
        >
          <X size={12} />
        </button>
      </div>

      {notebooks === null && (
        <p className={styles.notebookChooserNote} role="status">
          <Loader2 size={12} className={styles.spin} aria-hidden="true" />
          Looking for your notebooks…
        </p>
      )}

      {problem && (
        <p className={styles.notebookChooserProblem} role="alert">
          <AlertTriangle size={12} aria-hidden="true" />
          {problem}
        </p>
      )}

      {notebooks !== null && !problem && rows.length === 0 && (
        <p className={styles.notebookChooserNote}>
          {notebooks.length === 0
            ? 'You have no notebooks yet. Create one on the Notebooks screen and add sources to it.'
            : `No notebook matches “${query}”.`}
        </p>
      )}

      <ul className={styles.notebookChooserList} ref={listRef}>
        {rows.map((row, index) => (
          <li key={row.notebook.id}>
            <button
              type="button"
              role="option"
              aria-selected={index === highlight}
              className={styles.notebookChooserRow}
              data-highlighted={index === highlight || undefined}
              // `onMouseDown`, not `onClick`: a click blurs the textarea
              // first, and the blur is what closes the chooser — so the row
              // would unmount before its handler ever ran.
              onMouseDown={event => {
                event.preventDefault();
                choose(row.notebook);
              }}
              onMouseEnter={() => onHighlightChange(index)}
            >
              <span className={styles.notebookChooserName}>{row.notebook.name}</span>
              <span className={styles.notebookChooserMeta}>
                <ReadinessSummary
                  total={row.notebook.documentCount}
                  readiness={row.readiness}
                />
              </span>
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
};

/**
 * "8 sources · 5 readable" — or the honest alternative when it is not known.
 *
 * Never says "8 sources will be read". How many sources a notebook has and how
 * many can be read are separate numbers, and printing the first where the
 * second belongs is how "1 of 1 will be read" came to sit above a warning that
 * the one source was unreadable.
 */
const ReadinessSummary: React.FC<{
  total: number;
  readiness?: SourceReadiness[] | null;
}> = ({ total, readiness }) => {
  const sources = total === 1 ? '1 source' : `${total} sources`;

  if (readiness === undefined) return <>{sources} · checking…</>;
  if (readiness === null) return <>{sources} · readability unknown</>;
  if (readiness.length === 0) return <>no sources yet</>;

  const readable = readiness.filter(source => usableAsEvidence(source.state)).length;
  const working = readiness.filter(source => inProgress(source.state)).length;
  const blocked = readiness.length - readable - working;

  return (
    <>
      {sources} · {readable} readable
      {working > 0 ? ` · ${working} still being read` : ''}
      {blocked > 0 ? ` · ${blocked} need attention` : ''}
    </>
  );
};

export interface NotebookChipProps {
  scope: NotebookScope;
  onChange: (scope: NotebookScope) => void;
  onClear: () => void;
  /** Opens the notebook's own screen, for sources and evidence. */
  onOpenNotebook: (notebookId: string) => void;
}

/**
 * The notebook the next question is scoped to.
 *
 * Four things have to be reachable from here, because the alternative is a
 * person who cannot tell what their question is about to read: which notebook,
 * which of its sources, the way to clear it, and the way through to the
 * notebook itself for anything more involved.
 */
export const NotebookChip: React.FC<NotebookChipProps> = ({
  scope,
  onChange,
  onClear,
  onOpenNotebook,
}) => {
  const [open, setOpen] = useState(false);
  const [readiness, setReadiness] = useState<SourceReadiness[] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [repairing, setRepairing] = useState<string | null>(null);
  // Bumped by "Check again", which is the offer made when the store itself
  // could not be read -- nothing is known to be wrong with the file, so asking
  // again is the whole remedy.
  const [reloadToken, setReloadToken] = useState(0);

  // Reloaded whenever the notebook changes and whenever the panel opens.
  // Readiness from the previous notebook must never be shown against this one:
  // that is exactly how a warning about another notebook's scan ends up beside
  // a healthy source.
  useEffect(() => {
    if (!open) return;
    let live = true;
    setReadiness(null);
    setProblem(null);
    notebookResearchService
      .sourceStatus(scope.notebookId)
      .then(found => {
        if (live) setReadiness(found);
      })
      .catch(error => {
        if (!live) return;
        // Left as `null` rather than an empty list. An empty list is a
        // notebook with no sources; this is a notebook whose sources could not
        // be read, and a selection must not be built on it.
        setReadiness(null);
        setProblem(
          error instanceof Error
            ? error.message
            : 'This notebook’s sources could not be read.',
        );
      });
    return () => {
      live = false;
    };
  }, [open, scope.notebookId, reloadToken]);

  const readable = useMemo(
    () => (readiness ?? []).filter(source => usableAsEvidence(source.state)),
    [readiness],
  );

  const included = useCallback(
    (sha: string) => {
      switch (scope.selection.mode) {
        case 'all':
          return true;
        case 'none':
          return false;
        case 'subset':
          return scope.selection.sha256s.includes(sha);
      }
    },
    [scope.selection],
  );

  const toggle = useCallback(
    (sha: string) => {
      const current =
        scope.selection.mode === 'all'
          ? (readiness ?? []).map(source => source.documentSha256)
          : scope.selection.mode === 'subset'
            ? scope.selection.sha256s
            : [];
      const next = current.includes(sha)
        ? current.filter(id => id !== sha)
        : [...current, sha];
      // Unticking the last source is "none", said as none. Not an empty
      // subset, which the backend refuses, and certainly not "all".
      onChange({
        ...scope,
        selection: next.length === 0 ? noSources() : someSources(next),
      });
    },
    [scope, readiness, onChange],
  );

  // Which notebook the panel is currently showing, readable from inside an
  // async callback that started before the answer arrived.
  const showing = useRef(scope.notebookId);
  useEffect(() => {
    showing.current = scope.notebookId;
  }, [scope.notebookId]);

  const repair = useCallback(
    async (sha: string) => {
      // Captured at call time. This chip is not remounted when somebody
      // switches notebooks -- only its `scope` prop changes -- so a repair
      // started against notebook A can resolve after the panel has moved to
      // notebook B and loaded B's readiness.
      //
      // Applying it anyway would be worse than a wasted call: sources are
      // content-addressed, so the same file can sit in both notebooks, the row
      // would be matched by `documentSha256`, and B's row would be quietly
      // overwritten with the state of A's copy.
      const requestedFor = scope.notebookId;
      setRepairing(sha);
      try {
        const repaired = await notebookResearchService.repairSource(requestedFor, sha);
        if (showing.current !== requestedFor) return;
        setReadiness(prev =>
          (prev ?? []).map(source => (source.documentSha256 === sha ? repaired : source)),
        );
      } catch (error) {
        if (showing.current !== requestedFor) return;
        setProblem(
          error instanceof Error ? error.message : 'That source could not be repaired.',
        );
      } finally {
        if (showing.current === requestedFor) setRepairing(null);
      }
    },
    [scope.notebookId],
  );

  const summary = describeSelection(scope.selection, scope.documentCount);
  const nothingReadable =
    readiness !== null && readiness.length > 0 && readable.length === 0;

  return (
    <div className={styles.notebookChipWrap}>
      <div className={styles.notebookChip}>
        <BookOpen size={12} aria-hidden="true" />
        <button
          type="button"
          className={styles.notebookChipName}
          onClick={() => setOpen(value => !value)}
          aria-expanded={open}
          title="Choose which sources this notebook contributes"
        >
          {scope.notebookName}
          <span className={styles.notebookChipCount}>{summary}</span>
        </button>
        {scope.selection.mode === 'none' && (
          <span className={styles.notebookChipWarn} title="No source will be read">
            <AlertTriangle size={11} aria-hidden="true" />
          </span>
        )}
        <button
          type="button"
          className={styles.attachmentRemove}
          onClick={onClear}
          aria-label={`Stop asking ${scope.notebookName}`}
          title="Clear the notebook from this chat"
        >
          <X size={12} />
        </button>
      </div>

      {open && (
        <div className={styles.notebookChipPanel} role="dialog" aria-label="Sources">
          <div className={styles.notebookChipPanelHead}>
            <button
              type="button"
              className={styles.notebookChipAction}
              onClick={() => onChange({ ...scope, selection: allSources() })}
            >
              Select all
            </button>
            <button
              type="button"
              className={styles.notebookChipAction}
              onClick={() => onChange({ ...scope, selection: noSources() })}
            >
              Select none
            </button>
            <button
              type="button"
              className={styles.notebookChipAction}
              onClick={() => onOpenNotebook(scope.notebookId)}
            >
              Manage sources
            </button>
          </div>

          {problem && (
            <p className={styles.notebookChooserProblem} role="alert">
              <AlertTriangle size={12} aria-hidden="true" />
              {problem}
            </p>
          )}

          {readiness === null && !problem && (
            <p className={styles.notebookChooserNote} role="status">
              <Loader2 size={12} className={styles.spin} aria-hidden="true" />
              Checking which sources can be read…
            </p>
          )}

          {nothingReadable && (
            <p className={styles.notebookChooserProblem} role="alert">
              <AlertTriangle size={12} aria-hidden="true" />
              None of this notebook’s sources can be read yet, so a question scoped to
              it has no evidence to draw on.
            </p>
          )}

          <ul className={styles.notebookChipSources}>
            {(readiness ?? []).map(source => (
              <li key={source.documentSha256} className={styles.notebookChipSource}>
                <label className={styles.notebookChipSourceLabel}>
                  <input
                    type="checkbox"
                    checked={included(source.documentSha256)}
                    disabled={!usableAsEvidence(source.state)}
                    onChange={() => toggle(source.documentSha256)}
                  />
                  <span className={styles.notebookChipSourceName}>
                    {source.documentName}
                  </span>
                  <span className={styles.notebookChipSourceState} data-state={source.state}>
                    {usableAsEvidence(source.state) && (
                      <Check size={11} aria-hidden="true" />
                    )}
                    {describeState(source.state)}
                  </span>
                </label>

                {source.reasons.length > 0 && (
                  <ul className={styles.notebookChipReasons}>
                    {source.reasons.map((reason, i) => (
                      <li key={`${reason.kind}-${i}`}>{describeReason(reason)}</li>
                    ))}
                  </ul>
                )}

                {source.repair === 'associate' && (
                  <button
                    type="button"
                    className={styles.notebookChipAction}
                    disabled={repairing === source.documentSha256}
                    onClick={() => void repair(source.documentSha256)}
                  >
                    <RefreshCw size={11} aria-hidden="true" />
                    {repairing === source.documentSha256 ? 'Repairing…' : 'Repair'}
                  </button>
                )}
                {source.repair === 'retry' && (
                  <button
                    type="button"
                    className={styles.notebookChipAction}
                    onClick={() => setReloadToken(token => token + 1)}
                  >
                    <RefreshCw size={11} aria-hidden="true" /> Check again
                  </button>
                )}
                {source.repair === 'reprocess' && (
                  <button
                    type="button"
                    className={styles.notebookChipAction}
                    onClick={() => onOpenNotebook(scope.notebookId)}
                  >
                    Read it again on the Notebooks screen
                  </button>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
