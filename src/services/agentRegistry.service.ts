/**
 * The agent registry, as the frontend sees it.
 *
 * ## These types are one contract written twice
 *
 * Every shape here has a counterpart in `src-tauri/src/agents/mod.rs`, and the
 * field names are what `serde(rename_all = "camelCase")` produces from it. A
 * field added on one side and not the other is a runtime `undefined` that
 * typechecks, which is the failure mode this comment exists to make expensive
 * to cause.
 *
 * ## What the frontend is not allowed to decide
 *
 * Whether a mutation is permitted. Every one of these commands is refused in
 * Rust for anybody who is not an administrator, and the administration menu
 * being hidden is a courtesy rather than the control — the command stays
 * registered and reachable by anything that can make an IPC call. So a screen
 * built on this must render the refusal it gets back rather than assuming it
 * will not get one.
 */

import { getBackendService } from './api';

/** Where an agent stands with the deployment. */
export type AgentState = 'enabled' | 'disabled' | 'archived';

/** How long anything an agent records survives. */
export type MemoryScope = 'none' | 'task' | 'run';

/**
 * The shape an agent's output must take. Mirrors `subagents::profile::SchemaKind`.
 *
 * `document`, `deck` and `workbook` are registered contracts with no worker yet:
 * an agent may declare one, and dispatching it is refused as "registered and
 * cannot yet be run" until its writer lands. See `RUNNABLE_SCHEMAS`.
 */
export type OutputSchema =
  | 'extraction'
  | 'retrieval'
  | 'calculation'
  | 'review'
  | 'code'
  | 'document'
  | 'deck'
  | 'workbook';

/**
 * The schemas a worker in this build can actually produce. Mirrors
 * `subagents::definitions::capability_for`, and exists so the Agents screen can
 * say which contracts are runnable instead of offering all eight as equal.
 */
export const RUNNABLE_SCHEMAS: ReadonlySet<OutputSchema> = new Set<OutputSchema>([
  'extraction',
  'retrieval',
  'calculation',
  'review',
  'code',
]);

/** Whether an agent may run beside others. */
export type Isolation = 'readOnly' | 'writer' | 'approvalSensitive';

/** Where an agent may write, when it may write at all. */
export type WritePolicy = 'none' | 'ownDirectory';

/**
 * One skill an agent is bound to, pinned to the bytes it was bound to.
 *
 * The version and the hash both, because they answer different questions: the
 * version is what an author meant, and the hash is what is on disk. A skill
 * edited without its version being bumped is exactly the case that makes a
 * pinned version a fiction.
 */
export interface SkillBinding {
  name: string;
  version: string;
  sha256: string;
}

/**
 * Which models an agent may use.
 *
 * Mutable, and deliberately the only mutable identity-adjacent thing: changing
 * it changes what an agent runs on and changes nothing about who it is, what it
 * remembers, or what it has already done.
 *
 * **Not editable through `update`.** Rust refuses a submitted binding that
 * differs from the stored one, because changing the model an agent runs on has
 * to stop its work at a safe point, settle anything in flight, save its state,
 * prove the new model can hold the task, and be able to undo itself — none of
 * which a saved form can do. Send this one field through
 * `modelTransition.service.ts`'s `begin` and every other field through
 * `update`. A form that submits a changed binding gets a refusal naming that
 * path.
 */
export interface ModelBinding {
  /** `null` means "whatever routing picks for the role". */
  defaultModelId: string | null;
  fallbackModelIds: string[];
  /** Empty means any model registered for the agent's role. */
  eligibleModelIds: string[];
}

export interface MemoryPolicy {
  scope: MemoryScope;
  /** Whether other agents on the same task may read what this one records. */
  sharedWithTask: boolean;
}

/** Where a definition came from, when it was not written here. */
export interface ImportOrigin {
  source: string;
  profileName: string;
  profileSha256: string;
}

export interface Limits {
  maxTurns: number;
  maxOutputTokens: number;
  maxChildren: number;
  maxDurationSeconds: number;
}

/** An agent, as the deployment holds it. */
export interface AgentDefinition {
  /** Stable for the life of the agent. Never derived from anything editable. */
  agentId: string;
  /**
   * Which revision this is. Sent back with an edit so a concurrent change is
   * refused rather than silently overwritten.
   */
  definitionVersion: number;

  displayName: string;
  description: string;
  /** What the agent is told it is for. The only field that reaches a model. */
  instructions: string;
  role: string;
  state: AgentState;
  /** One of the palette. */
  color: string;

  skills: SkillBinding[];
  allowedTools: string[];
  /** Wins over `allowedTools`, over the parent's grant, over everything. */
  deniedTools: string[];

  memory: MemoryPolicy;
  models: ModelBinding;
  outputSchema: OutputSchema;

  limits: Limits;
  maxConcurrent: number;

  isolation: Isolation;
  writePolicy: WritePolicy;
  classificationCeiling: string;

  importedFrom?: ImportOrigin;
  /** RFC 3339, UTC. */
  createdAt: string;
  updatedAt: string;
}

/**
 * An agent plus what will not currently work.
 *
 * `unresolved` is kept apart from the definition on purpose. The definition is
 * what somebody configured; these are the models and skills this machine cannot
 * honour right now. Folding them together would let an agent be shown as ready
 * because its record parsed, which is the one thing the administration screen
 * must not do.
 */
export interface AgentView {
  definition: AgentDefinition;
  unresolved: string[];
  /** True when this agent could be given work this moment. */
  ready: boolean;
  /**
   * Whether a worker is registered for this agent's role.
   *
   * Separate from `unresolved`, which is about what the definition names. An
   * agent can name nothing missing and still have no worker — it would accept
   * a task and never perform it — so the list shows the two apart rather than
   * collapsing both into one "not ready".
   */
  workerAvailable: boolean;
}

/** One installed skill, as the editor's picker needs it. */
export interface SkillOption {
  name: string;
  description: string;
  version: string;
  /** SHA-256 of the whole SKILL.md. What a binding pins. */
  sha256: string;
  classification: string;
  /** Tools the skill's own manifest says it needs. */
  requiredTools: string[];
  approvalClass: string;
  network: string;
  available: boolean;
  unavailableBecause: string | null;
  imported: boolean;
}

/** What an agent declares, what it would actually get, and the difference. */
export interface PolicyPreview {
  declaredTools: string[];
  /** Never wider than `declaredTools`. */
  effectiveTools: string[];
  refusedTools: string[];
  deniedTools: string[];
  memoryScope: string;
  isolation: string;
  writePolicy: string;
  classificationCeiling: string;
  requiredSchema: string;
  maxTurns: number;
  maxOutputTokens: number;
  maxDurationSeconds: number;
}

/** How one bound skill stands against what is installed right now. */
export interface SkillResolution {
  name: string;
  pinnedSha256: string;
  installedSha256: string | null;
  /** `ok`, `missing`, `changed` or `quarantined`. */
  status: string;
  because: string;
}

/** A run with a live child of this agent. */
export interface DependentRun {
  runId: string;
  prompt: string;
  actor: string;
  state: string;
  children: string[];
}

/** Everything a destructive lifecycle change would affect. */
export interface AgentDependents {
  agentId: string;
  activeRuns: DependentRun[];
  /** Archiving keeps these and keeps them attributed. */
  memoryItems: number;
  /** Why archiving now would be unsafe. Empty means it is safe. */
  blocking: string[];
}

/** What a real test run did. */
export interface TestRunOutcome {
  agentId: string;
  runId: string;
  childId: string;
  /**
   * True when a child actually executed, whatever it concluded.
   *
   * The health signal. A mechanical worker given a harmless objective can
   * legitimately conclude `Failed` — "nothing was offered to check" — and that
   * is not a broken agent.
   */
  ran: boolean;
  status: string;
  findings: string[];
  uncertainty: string[];
  turnsUsed: number;
  detail: string | null;
  reused: boolean;
}

/** What a dry run found, without running anything. */
export interface AgentPreview {
  agentId: string;
  definitionVersion: number;
  selectedModel: string | null;
  modelPreference: string[];
  missingModels: string[];
  /** `binding` when the agent names its own models, `routing` otherwise. */
  modelChosenBy: string;
  policy: PolicyPreview;
  skills: SkillResolution[];
  workerAvailable: boolean;
  memoryItemsInScope: number;
  /** Everything standing between this agent and running. Empty means it runs. */
  blocked: string[];
}

/** What an accepted mutation did. */
export interface AgentMutation {
  agentId: string;
  definitionVersion: number;
  action:
    | 'created'
    | 'updated'
    | 'cloned'
    | 'enabled'
    | 'disabled'
    | 'archived'
    | 'imported';
  /**
   * True when nothing changed — a state set to the one it already held.
   * Reported rather than treated as a write, so a screen can say "no change"
   * instead of claiming a save.
   */
  unchanged: boolean;
}

/** How a delegated job stands, as Rust's `JobStatus` spells it. */
export type OrchestratorJobStatus =
  | 'queued'
  | 'running'
  | 'completed'
  | 'partial'
  | 'blocked'
  | 'failed'
  | 'timed_out'
  | 'cancelled'
  | 'refused'
  | 'interrupted';

/**
 * One job the main run's coordinator dispatched with `agent.delegate` (P05).
 *
 * The backend's record: the definition version the job was pinned to at
 * dispatch, the model it was routed to, the lease decision it actually got, and
 * the status the manager settled — never the coordinator's description of it.
 */
export interface OrchestratorJob {
  jobId: string;
  runId: string;
  stepId: string;
  role: string;
  mode: 'readOnly' | 'writer';
  attempt: number;
  childId: string;
  status: OrchestratorJobStatus;
  definitionId: string;
  definitionVersion: number | null;
  definitionOrigin: string;
  modelId: string | null;
  parentModelId: string | null;
  lease: string;
  deadlineAt: string;
  createdAt: string;
  settledAt: string | null;
  /** Counts, published ids, artifact versions and what was not done. */
  result: {
    summary?: string;
    findings?: number;
    evidenced?: number;
    published?: string[];
    artifacts?: string[];
    missing?: string[];
  } | null;
  /** The step's status in the run's newest plan version. */
  stepStatus: string | null;
  stepNote: string | null;
  planVersion: number | null;
}

export const agentRegistryService = {
  /** Every agent this session may see. */
  list(): Promise<AgentView[]> {
    return getBackendService().invoke<AgentView[]>('agent_registry_list');
  },

  get(agentId: string): Promise<AgentView> {
    return getBackendService().invoke<AgentView>('agent_registry_get', { agentId });
  },

  /**
   * Creates an agent.
   *
   * The `agentId`, `definitionVersion`, `createdAt` and `updatedAt` on the
   * definition sent are ignored — Rust mints them, so a form cannot choose an
   * id that collides with an existing agent's history.
   */
  create(definition: AgentDefinition): Promise<AgentMutation> {
    return getBackendService().invoke<AgentMutation>('agent_registry_create', {
      definition,
    });
  },

  /**
   * Saves an edit.
   *
   * `expectedVersion` is the version the form was opened at. Rust refuses the
   * save if the record has moved on since, naming both versions, so two
   * administrators editing the same agent do not silently undo each other.
   */
  update(
    agentId: string,
    expectedVersion: number,
    definition: AgentDefinition,
  ): Promise<AgentMutation> {
    return getBackendService().invoke<AgentMutation>('agent_registry_update', {
      agentId,
      expectedVersion,
      definition,
    });
  },

  /** Copies an agent's configuration under a new identity and empty memory. */
  clone(agentId: string, displayName: string): Promise<AgentMutation> {
    return getBackendService().invoke<AgentMutation>('agent_registry_clone', {
      agentId,
      displayName,
    });
  },

  /** Enables, disables or archives an agent. Version-checked like an edit. */
  setState(
    agentId: string,
    expectedVersion: number,
    state: AgentState,
  ): Promise<AgentMutation> {
    return getBackendService().invoke<AgentMutation>('agent_registry_set_state', {
      agentId,
      expectedVersion,
      state,
    });
  },

  /**
   * The colours an agent may be given.
   *
   * Fetched rather than restated here, because the same list decides what the
   * memory graph draws with and two copies of it would eventually disagree
   * about which colour an agent is.
   */
  palette(): Promise<string[]> {
    return getBackendService().invoke<string[]>('agent_registry_palette');
  },

  /**
   * Every installed skill, for the editor's picker.
   *
   * Reads the registry's cached snapshot. `refresh` re-walks the skills
   * directory, and is wired to a button rather than to a render: walking the
   * filesystem on every keystroke costs an operator nothing but latency, for
   * information that only changes when somebody installs something.
   */
  skillCatalog(refresh = false): Promise<SkillOption[]> {
    return getBackendService().invoke<SkillOption[]>('agent_skill_catalog', { refresh });
  },

  /**
   * Everything that would be affected by archiving or disabling this agent.
   *
   * Read before a destructive lifecycle change, because "are you sure" without
   * the consequences is a question nobody can answer.
   */
  dependents(agentId: string): Promise<AgentDependents> {
    return getBackendService().invoke<AgentDependents>('agent_dependents', { agentId });
  },

  /**
   * Actually runs the agent once, on a fixed harmless objective.
   *
   * A real run with a real id and a real event trail — not a simulation. Four
   * of the five shipped roles are mechanical and need no model loaded.
   */
  testRun(agentId: string): Promise<TestRunOutcome> {
    return getBackendService().invoke<TestRunOutcome>('agent_test_run', { agentId });
  },

  /**
   * A dry run: what would happen, with nothing happening.
   *
   * Starts no run, writes no memory and claims no worker. What comes back is
   * the model that would be chosen, the tools that would actually be held
   * after narrowing, how each bound skill resolves, and every reason the agent
   * would not run.
   */
  preview(agentId: string): Promise<AgentPreview> {
    return getBackendService().invoke<AgentPreview>('agent_preview', { agentId });
  },

  /**
   * The coordinator's recent delegated jobs, newest first. Scoped by Rust to
   * runs the signed-in person started; an administrator sees every run's.
   */
  orchestratorJobs(limit = 30): Promise<OrchestratorJob[]> {
    return getBackendService().invoke<OrchestratorJob[]>('agent_orchestrator_jobs', { limit });
  },
};
