import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  AlertTriangle,
  Check,
  Copy,
  FlaskConical,
  Plus,
  RotateCw,
  Save,
  ShieldAlert,
  X,
  ArrowLeftRight,
  Play,
} from 'lucide-react';
import {
  agentRegistryService,
  type AgentDefinition,
  type AgentPreview,
  type AgentState,
  type AgentView,
  type MemoryScope,
  type OrchestratorJob,
  type OutputSchema,
  RUNNABLE_SCHEMAS,
  type SkillOption,
  type TestRunOutcome,
} from '../services/agentRegistry.service';
import {
  awaiting,
  isWaitingForSomebody,
  modelTransitionService,
  type TransitionStatus,
} from '../services/modelTransition.service';
import styles from './Agents.module.css';

/**
 * Agent administration.
 *
 * ## What a saved form is
 *
 * It is a write to the registry that the runtime reads. There is no separate
 * "apply" step and no cache to warm: the next run pins whatever version the
 * registry holds, so saving here changes what the machine does next.
 *
 * ## Why every refusal comes from the back end
 *
 * The menu hides this page from an employee, and the route is gated. Neither is
 * the control. The commands are registered and reachable by anything that can
 * make an IPC call, so the refusal that matters is the one in
 * `agents::store::AgentRegistry`, which checks the role on every mutation. This
 * page shows what that refusal said; it does not decide anything.
 *
 * ## Why "ready" is never inferred here
 *
 * Whether an agent can run depends on the model registry, the skill registry
 * and whether a worker is registered for its role — none of which this page can
 * see. It asks the back end and prints the sentences it gets back. An agent
 * whose record parsed is not an agent that runs, and this screen must never be
 * the thing that confuses the two.
 */

/** The roles an agent may hold, matching `registry::ModelRole`. */
const ROLES = ['reasoning', 'coding', 'vision', 'documentOcr', 'embedding', 'rerank'] as const;

const SCHEMAS: OutputSchema[] = [
  'extraction',
  'retrieval',
  'calculation',
  'review',
  'code',
  'document',
  'deck',
  'workbook',
];

const MEMORY_SCOPES: MemoryScope[] = ['none', 'task', 'run'];

/** RFC 3339 to something a person reads. */
function when(iso: string): string {
  const at = new Date(iso);
  return Number.isNaN(at.getTime()) ? iso : at.toLocaleString();
}

/** A colour swatch on the surface the agent graph actually uses. */
const ColorPreview: React.FC<{ color: string; label: string }> = ({ color, label }) => (
  <span className={styles.preview} aria-label={`${label} on the graph background`}>
    <span className={styles.previewDot} style={{ background: color }} />
    <span className={styles.previewEdge} style={{ background: color }} />
  </span>
);

/** One row of the list. */
const Row: React.FC<{
  view: AgentView;
  selected: boolean;
  onOpen: () => void;
}> = ({ view, selected, onOpen }) => {
  const { definition } = view;
  const model = definition.models.defaultModelId ?? 'chosen by routing';
  return (
    <li>
      <button
        type="button"
        className={selected ? styles.rowSelected : styles.row}
        onClick={onOpen}
        aria-current={selected ? 'true' : undefined}
      >
        <span className={styles.swatch} style={{ background: definition.color }} aria-hidden />
        <span className={styles.rowMain}>
          <span className={styles.rowName}>{definition.displayName}</span>
          {/* The id is shown, not hidden: it is what survives a rename, and an
            * administrator reconciling a graph attribution needs to read it. */}
          <span className={styles.rowId}>{definition.agentId}</span>
        </span>
        <span className={styles.rowMeta}>{definition.role}</span>
        <span className={styles.rowMeta}>{model}</span>
        <span className={styles.rowMeta}>
          {definition.skills.length} skill{definition.skills.length === 1 ? '' : 's'}
        </span>
        <span className={styles.rowState} data-state={definition.state}>
          {definition.state}
        </span>
        {/* Three separate facts, never merged into one badge: an agent can be
          * enabled, resolve everything it names, and still have no worker. */}
        <span
          className={view.ready ? styles.ready : styles.notReady}
          title={
            view.ready
              ? 'Resolves, enabled, and a worker is registered.'
              : [
                  ...view.unresolved,
                  ...(view.workerAvailable ? [] : ['No worker is registered for this role.']),
                ].join(' ')
          }
        >
          {view.ready ? 'ready' : view.workerAvailable ? 'blocked' : 'no worker'}
        </span>
      </button>
    </li>
  );
};

export const Agents: React.FC = () => {
  const [agents, setAgents] = useState<AgentView[]>([]);
  const [palette, setPalette] = useState<string[]>([]);
  const [catalog, setCatalog] = useState<SkillOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  const [openId, setOpenId] = useState<string | null>(null);
  /** The form. Null when nothing is open. */
  const [draft, setDraft] = useState<AgentDefinition | null>(null);
  const [preview, setPreview] = useState<AgentPreview | null>(null);
  const [testRun, setTestRun] = useState<TestRunOutcome | null>(null);
  const [busy, setBusy] = useState(false);
  /** Where the open agent stands with respect to its model. */
  const [transition, setTransition] = useState<TransitionStatus | null>(null);
  /** What the main run's coordinator delegated, as the backend recorded it. */
  const [jobs, setJobs] = useState<OrchestratorJob[] | null>(null);
  const headingRef = useRef<HTMLHeadingElement>(null);

  const load = useCallback(async () => {
    // Read beside the registry and never in its way: a deployment whose job
    // table cannot be read still lists and edits its agents, and says so.
    void agentRegistryService
      .orchestratorJobs()
      .then(setJobs)
      .catch(() => setJobs(null));
    try {
      const [found, colors] = await Promise.all([
        agentRegistryService.list(),
        agentRegistryService.palette(),
      ]);
      setAgents(found);
      setPalette(colors);
      setError(null);
    } catch (problem) {
      // The refusal from Rust, verbatim. A non-administrator reaching this page
      // by a deep link sees the reason rather than an empty table.
      setError(String(problem));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  useEffect(() => {
    // The catalogue is read once, from the cached snapshot. Re-reading it per
    // keystroke would walk the filesystem for information that changes when
    // somebody installs a skill.
    void agentRegistryService
      .skillCatalog()
      .then(setCatalog)
      .catch(() => setCatalog([]));
  }, []);

  const open = useCallback(async (agentId: string) => {
    setOpenId(agentId);
    setNotice(null);
    setPreview(null);
    setTestRun(null);
    try {
      // Re-read rather than using the row. The list may be minutes old, and
      // editing from a stale revision is the thing `expectedVersion` exists to
      // refuse — better to start from the current one.
      const fresh = await agentRegistryService.get(agentId);
      setDraft(fresh.definition);
      // Where this agent stands with its model. An agent mid-handoff must not
      // be moved again, and the screen has to say so rather than letting
      // somebody start a second one.
      setTransition(await modelTransitionService.status(agentId).catch(() => null));
      headingRef.current?.focus();
    } catch (problem) {
      setError(String(problem));
    }
  }, []);

  const runPreview = useCallback(async (agentId: string) => {
    setBusy(true);
    try {
      setPreview(await agentRegistryService.preview(agentId));
      setNotice(null);
    } catch (problem) {
      setError(String(problem));
    } finally {
      setBusy(false);
    }
  }, []);

  /**
   * Moves an agent to a different model.
   *
   * Not part of `save`, because it is not an edit. `agent_registry_update`
   * refuses a change to the model binding for exactly this reason: a model
   * change has to drain the current round, freeze what the old model was
   * working from, verify the destination can hold it, and stay resumable if the
   * process dies half way. What it does *not* touch is the agent's identity,
   * its memory or its colour — the same agent continues on a different engine.
   */
  const changeModel = useCallback(
    async (agentId: string, currentVersion: number) => {
      const target = window.prompt('Move this agent to which model?');
      if (!target) return;
      setBusy(true);
      try {
        const view = await modelTransitionService.begin({
          agentId,
          targetModelId: target.trim(),
          expectedDefinitionVersion: currentVersion,
          verify: 'loadAndVerify',
        });
        setNotice(view.summary);
        setTransition(await modelTransitionService.status(agentId));
        await load();
        setError(null);
      } catch (problem) {
        setError(String(problem));
      } finally {
        setBusy(false);
      }
    },
    [load],
  );

  /**
   * Runs the agent once, for real.
   *
   * Deliberately separate from the dry run: that one answers "would this be
   * allowed", and this one answers "does this work". A worker that registers
   * and then errors passes the first and fails the second.
   */
  const executeTestRun = useCallback(async (agentId: string) => {
    setBusy(true);
    setTestRun(null);
    try {
      const outcome = await agentRegistryService.testRun(agentId);
      setTestRun(outcome);
      setNotice(
        outcome.reused
          ? `This exact test had already been run; showing that result (${outcome.runId}).`
          : `Ran as ${outcome.runId}.`,
      );
      setError(null);
    } catch (problem) {
      setError(String(problem));
    } finally {
      setBusy(false);
    }
  }, []);

  const save = useCallback(async () => {
    if (!draft) return;
    setBusy(true);
    try {
      const mutation = await agentRegistryService.update(
        draft.agentId,
        draft.definitionVersion,
        draft,
      );
      setNotice(
        mutation.unchanged
          ? 'Nothing changed.'
          : `Saved as version ${mutation.definitionVersion}.`,
      );
      await load();
      // The version moved, so the form must move with it, or the next save is
      // refused as stale against a version this screen itself created.
      const fresh = await agentRegistryService.get(draft.agentId);
      setDraft(fresh.definition);
      setPreview(null);
      setError(null);
    } catch (problem) {
      setError(String(problem));
    } finally {
      setBusy(false);
    }
  }, [draft, load]);

  const setState = useCallback(
    async (view: AgentView, state: AgentState) => {
      // Archiving or disabling changes what happens to work already under way,
      // so what it does is said before the decision rather than after it.
      if (state !== 'enabled') {
        const verb = state === 'archived' ? 'Archive' : 'Disable';
        // What this would affect, read before the question is asked. "Are you
        // sure?" without the consequences is a question nobody can answer.
        let affected = '';
        try {
          const dependents = await agentRegistryService.dependents(view.definition.agentId);
          if (dependents.activeRuns.length > 0) {
            affected =
              `\n\nRunning right now (${dependents.activeRuns.length}):\n` +
              dependents.activeRuns
                .map((run) => `  - ${run.runId} - ${run.actor} - ${run.children.length} child(ren)`)
                .join('\n') +
              '\n\nThese keep the version they pinned and will finish. No new work goes to it.';
          } else {
            affected = '\n\nNothing is running under this agent right now.';
          }
          if (dependents.blocking.length > 0) {
            affected += `\n\n${dependents.blocking.join('\n')}`;
          }
        } catch (problem) {
          // The dependants could not be read. Said out loud, rather than
          // letting the confirmation imply there are none.
          affected =
            `\n\nWhat this would affect could not be read (${String(problem)}), so this ` +
            'decision is being made without it.';
        }

        const ok = window.confirm(
          `${verb} ${view.definition.displayName}?` +
            affected +
            '\n\nArchiving keeps every graph attribution this agent already has - its past ' +
            'work stays readable and stays attributed to it. It is not a deletion: removing ' +
            'what an agent remembers is a separate, explicit operation.',
        );
        if (!ok) return;
      }
      setBusy(true);
      try {
        await agentRegistryService.setState(
          view.definition.agentId,
          view.definition.definitionVersion,
          state,
        );
        await load();
        setNotice(`${view.definition.displayName} is now ${state}.`);
        setError(null);
      } catch (problem) {
        setError(String(problem));
      } finally {
        setBusy(false);
      }
    },
    [load],
  );

  const clone = useCallback(
    async (view: AgentView) => {
      const name = window.prompt(
        `Name for the copy of ${view.definition.displayName}?`,
        `${view.definition.displayName} copy`,
      );
      if (!name) return;
      setBusy(true);
      try {
        const mutation = await agentRegistryService.clone(view.definition.agentId, name);
        await load();
        // A clone is a new identity, so it starts with nothing remembered. Said
        // out loud, because the opposite would be a reasonable guess.
        setNotice(
          `Created ${mutation.agentId}. It has a new identity and none of the original's memory.`,
        );
        setError(null);
      } catch (problem) {
        setError(String(problem));
      } finally {
        setBusy(false);
      }
    },
    [load],
  );

  const create = useCallback(async () => {
    const name = window.prompt('Name for the new agent?');
    if (!name) return;
    setBusy(true);
    try {
      // A whole definition, because that is what the command takes. The id and
      // the version are placeholders: the registry mints the id itself and
      // never takes one from a caller, which is what keeps identity out of a
      // form somebody can edit.
      const now = new Date().toISOString();
      const mutation = await agentRegistryService.create({
        agentId: '',
        definitionVersion: 0,
        displayName: name,
        description: '',
        instructions: '',
        role: 'reasoning',
        state: 'disabled',
        color: palette[agents.length % Math.max(1, palette.length)] ?? palette[0] ?? '#60A5FA',
        skills: [],
        allowedTools: [],
        deniedTools: [],
        memory: { scope: 'task', sharedWithTask: false },
        models: { defaultModelId: null, fallbackModelIds: [], eligibleModelIds: [] },
        outputSchema: 'retrieval',
        limits: {
          maxTurns: 8,
          maxOutputTokens: 2048,
          maxChildren: 0,
          maxDurationSeconds: 300,
        },
        maxConcurrent: 1,
        isolation: 'readOnly',
        writePolicy: 'none',
        classificationCeiling: 'internal',
        createdAt: now,
        updatedAt: now,
      });
      await load();
      await open(mutation.agentId);
      setNotice(`Created ${mutation.agentId}.`);
      setError(null);
    } catch (problem) {
      setError(String(problem));
    } finally {
      setBusy(false);
    }
  }, [agents.length, palette, load, open]);

  const selected = useMemo(
    () => agents.find((held) => held.definition.agentId === openId) ?? null,
    [agents, openId],
  );

  /** Edits one field of the draft without losing the rest. */
  const edit = <K extends keyof AgentDefinition>(key: K, value: AgentDefinition[K]) =>
    setDraft((held) => (held ? { ...held, [key]: value } : held));

  return (
    <div className={styles.page}>
      <header className={styles.header}>
        <h1 className={styles.title}>Agents</h1>
        <p className={styles.lede}>
          What each agent is for, what it may use, and what it runs on. A saved change is what
          the next run reads — there is nothing else to apply.
        </p>
      </header>

      {error && (
        <p className={styles.error} role="alert">
          <ShieldAlert size={15} aria-hidden />
          <span>{error}</span>
        </p>
      )}
      {notice && (
        <p className={styles.notice} role="status">
          <Check size={15} aria-hidden />
          <span>{notice}</span>
        </p>
      )}

      <div className={styles.toolbar}>
        <button
          type="button"
          className={styles.primary}
          onClick={() => void create()}
          disabled={busy}
        >
          <Plus size={14} aria-hidden /> New agent
        </button>
        <button type="button" className={styles.ghost} onClick={() => void load()} disabled={busy}>
          <RotateCw size={14} aria-hidden /> Refresh
        </button>
      </div>

      {loading ? (
        <p className={styles.quiet}>Reading the registry…</p>
      ) : agents.length === 0 ? (
        <p className={styles.quiet}>No agents yet.</p>
      ) : (
        <ul className={styles.list} aria-label="Agents">
          <li className={styles.headRow} aria-hidden>
            <span />
            <span>Name and identity</span>
            <span>Role</span>
            <span>Model</span>
            <span>Skills</span>
            <span>State</span>
            <span>Health</span>
          </li>
          {agents.map((view) => (
            <Row
              key={view.definition.agentId}
              view={view}
              selected={view.definition.agentId === openId}
              onOpen={() => void open(view.definition.agentId)}
            />
          ))}
        </ul>
      )}

      <OrchestratorJobs jobs={jobs} />

      {draft && selected && (
        <section className={styles.editor} aria-label={`Editing ${draft.displayName}`}>
          <div className={styles.editorHead}>
            <h2 className={styles.editorTitle} tabIndex={-1} ref={headingRef}>
              {draft.displayName}
            </h2>
            <button
              type="button"
              className={styles.ghost}
              onClick={() => {
                setDraft(null);
                setOpenId(null);
                setPreview(null);
              }}
              aria-label="Close the editor"
            >
              <X size={14} aria-hidden /> Close
            </button>
          </div>

          {/* The id, and the fact that renaming does not change it. */}
          <p className={styles.identity}>
            <code>{draft.agentId}</code> · version {draft.definitionVersion} · created{' '}
            {when(draft.createdAt)}. Renaming changes the label, never the identity: past work
            stays attributed to this id.
          </p>

          <div className={styles.grid}>
            <label className={styles.field}>
              <span className={styles.label}>Name</span>
              <input
                className={styles.input}
                value={draft.displayName}
                onChange={(event) => edit('displayName', event.target.value)}
              />
            </label>

            <label className={styles.field}>
              <span className={styles.label}>Role</span>
              <select
                className={styles.input}
                value={draft.role}
                onChange={(event) => edit('role', event.target.value)}
              >
                {ROLES.map((role) => (
                  <option key={role} value={role}>
                    {role}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.fieldWide}>
              <span className={styles.label}>Purpose</span>
              <input
                className={styles.input}
                value={draft.description}
                onChange={(event) => edit('description', event.target.value)}
                placeholder="What this agent is for, in one line."
              />
            </label>

            <label className={styles.fieldWide}>
              <span className={styles.label}>
                Instructions
                <span className={styles.hint}>The only field a model reads.</span>
              </span>
              <textarea
                className={styles.textarea}
                rows={5}
                value={draft.instructions}
                onChange={(event) => edit('instructions', event.target.value)}
              />
            </label>

            {/* Read-only, deliberately.
              *
              * `agent_registry_update` refuses a changed model binding outright:
              * moving an agent to a different model has to drain the current
              * round, freeze what the old model was working from, verify the
              * destination can hold it and stay resumable across a crash. None
              * of that can happen inside a form save.
              *
              * Leaving these editable would give an administrator a field that
              * accepts typing and a Save that always fails. The change lives on
              * its own button. */}
            <label className={styles.field}>
              <span className={styles.label}>
                Default model
                <span className={styles.hint}>Changed by "Change model".</span>
              </span>
              <input
                className={styles.input}
                readOnly
                value={draft.models.defaultModelId ?? 'chosen by routing for the role'}
              />
            </label>

            <label className={styles.field}>
              <span className={styles.label}>
                Fallback models
                <span className={styles.hint}>Tried in order when the default cannot serve.</span>
              </span>
              <input
                className={styles.input}
                readOnly
                value={draft.models.fallbackModelIds.join(', ') || 'none'}
              />
            </label>

            <label className={styles.field}>
              <span className={styles.label}>Output contract</span>
              <select
                className={styles.input}
                value={draft.outputSchema}
                onChange={(event) => edit('outputSchema', event.target.value as OutputSchema)}
              >
                {SCHEMAS.map((schema) => (
                  <option key={schema} value={schema}>
                    {/* Said in the option itself: an administrator choosing a
                        contract no worker produces should know before saving,
                        not when the first dispatch is refused. */}
                    {RUNNABLE_SCHEMAS.has(schema) ? schema : `${schema} (no worker yet)`}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span className={styles.label}>
                Memory access
                <span className={styles.hint}>What this agent may read and share.</span>
              </span>
              <select
                className={styles.input}
                value={draft.memory.scope}
                onChange={(event) =>
                  edit('memory', { ...draft.memory, scope: event.target.value as MemoryScope })
                }
              >
                {MEMORY_SCOPES.map((scope) => (
                  <option key={scope} value={scope}>
                    {scope}
                  </option>
                ))}
              </select>
            </label>

            <label className={styles.field}>
              <span className={styles.label}>
                Task sharing
                <span className={styles.hint}>
                  Whether other agents on the same task may read what this one records.
                </span>
              </span>
              <select
                className={styles.input}
                value={draft.memory.sharedWithTask ? 'shared' : 'private'}
                onChange={(event) =>
                  edit('memory', {
                    ...draft.memory,
                    sharedWithTask: event.target.value === 'shared',
                  })
                }
              >
                <option value="private">private to this agent</option>
                <option value="shared">shared with the task</option>
              </select>
            </label>

            <label className={styles.field}>
              <span className={styles.label}>Max turns</span>
              <input
                className={styles.input}
                type="number"
                min={1}
                value={draft.limits.maxTurns}
                onChange={(event) =>
                  edit('limits', { ...draft.limits, maxTurns: Number(event.target.value) })
                }
              />
            </label>

            <label className={styles.field}>
              <span className={styles.label}>Max output tokens</span>
              <input
                className={styles.input}
                type="number"
                min={1}
                value={draft.limits.maxOutputTokens}
                onChange={(event) =>
                  edit('limits', {
                    ...draft.limits,
                    maxOutputTokens: Number(event.target.value),
                  })
                }
              />
            </label>

            <label className={styles.fieldWide}>
              <span className={styles.label}>
                Allowed tools
                <span className={styles.hint}>
                  Comma separated. Narrowed against what you hold — never widened.
                </span>
              </span>
              <input
                className={styles.input}
                value={draft.allowedTools.join(', ')}
                onChange={(event) =>
                  edit(
                    'allowedTools',
                    event.target.value
                      .split(',')
                      .map((tool) => tool.trim())
                      .filter(Boolean),
                  )
                }
              />
            </label>

            <label className={styles.fieldWide}>
              <span className={styles.label}>
                Denied tools
                <span className={styles.hint}>Wins over everything above.</span>
              </span>
              <input
                className={styles.input}
                value={draft.deniedTools.join(', ')}
                onChange={(event) =>
                  edit(
                    'deniedTools',
                    event.target.value
                      .split(',')
                      .map((tool) => tool.trim())
                      .filter(Boolean),
                  )
                }
              />
            </label>
          </div>

          {/* Colour, previewed on the surface the graph actually draws on. A
            * swatch on a light panel says nothing about how it reads there. */}
          <fieldset className={styles.colors}>
            <legend className={styles.label}>Colour</legend>
            <div className={styles.swatches} role="radiogroup" aria-label="Agent colour">
              {palette.map((color) => (
                <button
                  key={color}
                  type="button"
                  role="radio"
                  aria-checked={draft.color === color}
                  aria-label={color}
                  className={draft.color === color ? styles.swatchOn : styles.swatchOff}
                  style={{ background: color }}
                  onClick={() => edit('color', color)}
                />
              ))}
              <ColorPreview color={draft.color} label={draft.displayName} />
            </div>
          </fieldset>

          {/* Skills, with the pinned bytes shown. A pin is the whole point: a
            * skill that changes under a working agent must be detectable. */}
          <fieldset className={styles.skills}>
            <legend className={styles.label}>Skills</legend>
            <ul className={styles.skillList}>
              {catalog.map((option) => {
                const bound = draft.skills.find((held) => held.name === option.name);
                const drifted = bound && bound.sha256 !== option.sha256;
                return (
                  <li key={option.name} className={styles.skillItem}>
                    <label className={styles.skillLabel}>
                      <input
                        type="checkbox"
                        checked={Boolean(bound)}
                        disabled={!option.available}
                        onChange={(event) =>
                          edit(
                            'skills',
                            event.target.checked
                              ? [
                                  ...draft.skills,
                                  {
                                    name: option.name,
                                    version: option.version,
                                    sha256: option.sha256,
                                  },
                                ]
                              : draft.skills.filter((held) => held.name !== option.name),
                          )
                        }
                      />
                      <span className={styles.skillName}>{option.name}</span>
                      <span className={styles.skillVersion}>{option.version}</span>
                    </label>
                    <span className={styles.skillMeta}>
                      <code title={bound?.sha256 ?? option.sha256}>
                        {(bound?.sha256 ?? option.sha256).slice(0, 12)}
                      </code>
                      {option.requiredTools.length > 0 && (
                        <span className={styles.hint}>
                          needs {option.requiredTools.join(', ')}
                        </span>
                      )}
                      {option.imported && <span className={styles.badge}>imported</span>}
                      {!option.available && (
                        <span className={styles.badgeWarn}>
                          {option.unavailableBecause ?? 'unavailable'}
                        </span>
                      )}
                      {drifted && (
                        <span className={styles.badgeWarn}>
                          pinned bytes differ from installed
                        </span>
                      )}
                    </span>
                  </li>
                );
              })}
              {catalog.length === 0 && <li className={styles.quiet}>No skills are installed.</li>}
            </ul>
          </fieldset>

          <div className={styles.actions}>
            <button
              type="button"
              className={styles.primary}
              onClick={() => void save()}
              disabled={busy}
            >
              <Save size={14} aria-hidden /> Save
            </button>
            <button
              type="button"
              className={styles.ghost}
              onClick={() => void runPreview(draft.agentId)}
              disabled={busy}
            >
              <FlaskConical size={14} aria-hidden /> Dry run
            </button>
            <button
              type="button"
              className={styles.ghost}
              onClick={() => void executeTestRun(draft.agentId)}
              disabled={busy || !selected.workerAvailable}
              title={
                selected.workerAvailable
                  ? 'Runs the agent once, for real, on a harmless objective. Writes a run record.'
                  : 'No worker is registered for this agent, so there is nothing to run.'
              }
            >
              <Play size={14} aria-hidden /> Test run
            </button>
            <button
              type="button"
              className={styles.ghost}
              onClick={() => void changeModel(draft.agentId, draft.definitionVersion)}
              disabled={busy || Boolean(transition?.open)}
              title={
                transition?.open
                  ? 'A handoff is already open for this agent.'
                  : 'Move this agent to a different model. Identity, memory and colour are kept.'
              }
            >
              <ArrowLeftRight size={14} aria-hidden /> Change model
            </button>
            <button
              type="button"
              className={styles.ghost}
              onClick={() => void clone(selected)}
              disabled={busy}
            >
              <Copy size={14} aria-hidden /> Clone
            </button>
            {selected.definition.state !== 'enabled' ? (
              <button
                type="button"
                className={styles.ghost}
                onClick={() => void setState(selected, 'enabled')}
                disabled={busy}
              >
                Enable
              </button>
            ) : (
              <button
                type="button"
                className={styles.ghost}
                onClick={() => void setState(selected, 'disabled')}
                disabled={busy}
              >
                Disable
              </button>
            )}
            {selected.definition.state !== 'archived' && (
              <button
                type="button"
                className={styles.danger}
                onClick={() => void setState(selected, 'archived')}
                disabled={busy}
              >
                Archive
              </button>
            )}
          </div>

          {testRun && (
            <section className={styles.dryRun} aria-label="Test run result">
              <h3 className={styles.previewTitle}>
                <Play size={14} aria-hidden />{' '}
                {testRun.ran ? 'Test run - the agent ran' : 'Test run - it did not run'}
              </h3>
              <dl className={styles.facts}>
                <dt>Concluded</dt>
                <dd>
                  {testRun.status}
                  <span className={styles.hint}>what it concluded, not whether it works</span>
                </dd>
                <dt>Run</dt>
                <dd>
                  <code>{testRun.runId}</code> - child <code>{testRun.childId}</code> -{' '}
                  {testRun.turnsUsed} turn(s)
                </dd>
                {testRun.findings.length > 0 && (
                  <>
                    <dt>Produced</dt>
                    <dd>{testRun.findings.join('; ')}</dd>
                  </>
                )}
                {testRun.uncertainty.length > 0 && (
                  <>
                    <dt>Could not establish</dt>
                    <dd>{testRun.uncertainty.join('; ')}</dd>
                  </>
                )}
                {testRun.detail && (
                  <>
                    <dt>Detail</dt>
                    <dd>{testRun.detail}</dd>
                  </>
                )}
              </dl>
            </section>
          )}
          {transition && <Transition status={transition} />}
          {preview && <Preview preview={preview} />}
        </section>
      )}
    </div>
  );
};

/**
 * Where the agent stands with respect to its model.
 *
 * Shows the phase as the sentence the back end already wrote, not as a raw
 * enum: "draining the current round" and "waiting for somebody to settle an
 * interrupted action" are different situations, and only one of them resolves
 * on its own. A spinner cannot tell them apart.
 */
const Transition: React.FC<{ status: TransitionStatus }> = ({ status }) => {
  const open = status.open;
  const recent = status.history.slice(0, 3);
  return (
    <section className={styles.dryRun} aria-label="Model transition">
      <h3 className={styles.previewTitle}>
        <ArrowLeftRight size={14} aria-hidden /> Model
      </h3>
      <dl className={styles.facts}>
        <dt>Running on</dt>
        <dd>{status.currentModelId ?? 'chosen by routing for the role'}</dd>
        {open && (
          <>
            <dt>In progress</dt>
            <dd>
              {open.summary}
              {isWaitingForSomebody(open) && (
                <span className={styles.refused}>
                  Waiting for a person: {awaiting(open).join('; ') || 'a settlement'}
                </span>
              )}
              {open.needsHuman && (
                <span className={styles.refused}>
                  This agent must not be given work until somebody looks.
                </span>
              )}
            </dd>
          </>
        )}
      </dl>
      {recent.length > 0 && (
        <ul className={styles.skillResolutions}>
          {recent.map((view) => (
            <li key={view.record.transitionId} data-status={view.outcome}>
              {view.summary}
              {view.notCarried && view.notCarried.length > 0 && (
                <> — {view.notCarried.length} thing(s) deliberately not carried across.</>
              )}
            </li>
          ))}
        </ul>
      )}
      <p className={styles.identity}>
        A model change keeps the agent's identity, everything it remembers and its colour. Only
        what it runs on changes.
      </p>
    </section>
  );
};

/**
 * What the main run's coordinator handed to these agents (P05).
 *
 * The coordinator is the model of the chat run itself — Spark X2.5 4B Q8 in
 * the target deployment — not an agent in the list above: it plans with
 * `task.plan_update` and hands plan steps to the agents here with
 * `agent.delegate`. Each row is the backend's record of one job: the definition
 * version it was pinned to when it was dispatched, the model it was routed to,
 * what happened to the card, and the status the manager settled. A step counts
 * as done only when its receipts say so; a job's own "completed" is shown next
 * to the step's status so the two can be told apart.
 */
const OrchestratorJobs: React.FC<{ jobs: OrchestratorJob[] | null }> = ({ jobs }) => (
  <section className={styles.dryRun} aria-label="Delegated jobs">
    <h3 className={styles.previewTitle}>
      <ArrowLeftRight size={14} aria-hidden /> Jobs the coordinator delegated
    </h3>
    <p className={styles.hint}>
      Plan steps the main run handed to these agents. A writer job waited for a person&apos;s
      approval; a step is complete only when its receipts, and any review it asked for, say so.
    </p>
    {jobs === null ? (
      <p className={styles.quiet}>The job record could not be read.</p>
    ) : jobs.length === 0 ? (
      <p className={styles.quiet}>No jobs have been delegated yet.</p>
    ) : (
      <ul className={styles.jobs} aria-label="Delegated jobs">
        <li className={styles.jobHead} aria-hidden>
          <span>Job</span>
          <span>Step</span>
          <span>Agent (pinned definition)</span>
          <span>Job status</span>
          <span>Model and card</span>
        </li>
        {jobs.map((job) => (
          <li key={job.jobId} className={styles.jobRow} data-status={job.status}>
            <span className={styles.rowMain}>
              <span className={styles.rowName}>{job.jobId}</span>
              <span className={styles.hint}>
                attempt {job.attempt} · {job.mode === 'writer' ? 'writer (approved)' : 'read-only'} ·{' '}
                {when(job.createdAt)}
              </span>
            </span>
            <span className={styles.rowMain}>
              <span>
                {job.stepId} {job.planVersion !== null && <span className={styles.hint}>plan v{job.planVersion}</span>}
              </span>
              <span className={styles.hint}>
                {job.stepStatus ?? 'no plan step'}
                {job.stepNote ? ` — ${job.stepNote}` : ''}
              </span>
            </span>
            <span className={styles.rowMain}>
              <span>{job.role}</span>
              <span className={styles.hint}>
                {job.definitionId} · {job.definitionVersion !== null ? `v${job.definitionVersion}` : 'no version'} ·{' '}
                {job.definitionOrigin}
              </span>
            </span>
            <span className={styles.rowMain}>
              <span>{job.status.replace('_', ' ')}</span>
              {job.result?.summary && <span className={styles.hint}>{job.result.summary}</span>}
              {job.result?.missing && job.result.missing.length > 0 && (
                <span className={styles.hint}>not done: {job.result.missing.join('; ')}</span>
              )}
            </span>
            <span className={styles.rowMain}>
              <span>{job.modelId ?? '—'}</span>
              <span className={styles.hint}>{job.lease}</span>
            </span>
          </li>
        ))}
      </ul>
    )}
  </section>
);

/** The dry run, shown as declared against effective. */
const Preview: React.FC<{ preview: AgentPreview }> = ({ preview }) => (
  <section className={styles.dryRun} aria-label="Dry run">
    <h3 className={styles.previewTitle}>
      <FlaskConical size={14} aria-hidden /> Dry run — nothing was started
    </h3>

    {preview.blocked.length > 0 ? (
      <ul className={styles.blocked}>
        {preview.blocked.map((why) => (
          <li key={why}>
            <AlertTriangle size={13} aria-hidden /> {why}
          </li>
        ))}
      </ul>
    ) : (
      <p className={styles.ok}>
        <Check size={13} aria-hidden /> Nothing is standing in the way of this agent running.
      </p>
    )}

    <dl className={styles.facts}>
      <dt>Model</dt>
      <dd>
        {preview.selectedModel ?? '—'}{' '}
        <span className={styles.hint}>
          {preview.modelChosenBy === 'routing'
            ? 'chosen by routing for the role'
            : `named by this agent${
                preview.modelPreference.length > 1
                  ? `, ahead of ${preview.modelPreference.length - 1} fallback(s)`
                  : ''
              }`}
        </span>
      </dd>

      <dt>Worker</dt>
      <dd>
        {preview.workerAvailable
          ? 'registered for this role'
          : 'none registered — this agent would accept a task and never perform it'}
      </dd>

      <dt>Tools declared</dt>
      <dd>{preview.policy.declaredTools.join(', ') || '—'}</dd>

      <dt>Tools effective</dt>
      <dd>
        {preview.policy.effectiveTools.join(', ') || '—'}
        {preview.policy.refusedTools.length > 0 && (
          <span className={styles.refused}>
            refused: {preview.policy.refusedTools.join(', ')}
          </span>
        )}
      </dd>

      {preview.policy.deniedTools.length > 0 && (
        <>
          <dt>Tools denied</dt>
          <dd>{preview.policy.deniedTools.join(', ')}</dd>
        </>
      )}

      <dt>Memory scope</dt>
      <dd>
        {preview.policy.memoryScope} · {preview.memoryItemsInScope} item(s) in scope
      </dd>

      <dt>Budget</dt>
      <dd>
        {preview.policy.maxTurns} turn(s), {preview.policy.maxOutputTokens} output token(s),{' '}
        {preview.policy.maxDurationSeconds}s
      </dd>
    </dl>

    {preview.skills.length > 0 && (
      <ul className={styles.skillResolutions}>
        {preview.skills.map((skill) => (
          <li key={skill.name} data-status={skill.status}>
            <strong>{skill.name}</strong> — {skill.because}
          </li>
        ))}
      </ul>
    )}
  </section>
);

export default Agents;
