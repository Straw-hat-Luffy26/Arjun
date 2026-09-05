import { listen, type UnlistenFn } from '@tauri-apps/api/event';
import { getBackendService } from './api';
import type { OcrDetent } from './ocr.service';

/**
 * Starting, watching and stopping an agent run.
 *
 * A run is not a request/response — it is minutes of work with tool calls in
 * the middle, and an operator who needs to see what is happening while it
 * happens. So the API is in two halves: `start` resolves once with the answer,
 * and `subscribe` streams the lifecycle in between.
 *
 * The loop itself runs in a Node child process built from OpenClaw's
 * `agent-core`; every tool call it wants to make is decided in Rust first. None
 * of that is visible here, and deliberately so: the UI's business is showing
 * work, not deciding what is permitted.
 */

/** Which material this is, so the router only considers models cleared for it. */
export type Classification =
  | 'internal'
  | 'processDiagram'
  | 'financial'
  | 'vendorNegotiation'
  | 'unreleasedDesign'
  | 'internalCorrespondence'
  | 'businessStrategy';

/**
 * What starts a run.
 *
 * Deliberately no model. Which model answers is the backend router's decision —
 * letting the UI name one would make automatic selection optional, which is the
 * opposite of what the product is demonstrating.
 */
/** A file the user attached in the composer, carried as bytes. */
export interface ComposerAttachment {
  name: string;
  mime: string;
  /** Base64 of the file itself, no data-URI prefix. */
  dataBase64: string;
}

/**
 * Reads a picked file into something that can cross the Tauri boundary.
 *
 * The backend cannot open a path the webview names — that would let the
 * frontend nominate any file on the machine — so the bytes travel instead.
 */
export async function toComposerAttachment(file: File): Promise<ComposerAttachment> {
  const dataUrl: string = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onerror = () => reject(new Error(`${file.name} could not be read.`));
    reader.onload = () => resolve(String(reader.result));
    reader.readAsDataURL(file);
  });
  const comma = dataUrl.indexOf(',');
  if (comma === -1) throw new Error(`${file.name} could not be encoded.`);
  return {
    name: file.name,
    // Browsers leave `type` empty for some picks; fall back to the extension
    // so a PNG chosen from an odd source is still recognised as one.
    mime: file.type || mimeFromName(file.name),
    dataBase64: dataUrl.slice(comma + 1),
  };
}

function mimeFromName(name: string): string {
  const ext = name.slice(name.lastIndexOf('.') + 1).toLowerCase();
  if (ext === 'png') return 'image/png';
  if (ext === 'jpg' || ext === 'jpeg') return 'image/jpeg';
  if (ext === 'webp') return 'image/webp';
  return 'application/octet-stream';
}

/**
 * What the backend is doing with an attachment right now.
 *
 * Only facts. `page`/`pages` arrive once the extractor has counted them, so
 * "Reading page 2 of 6" is never shown before those numbers are real.
 */
export interface AttachmentProgress {
  name: string;
  phase: 'reading' | 'preparing' | 'extracting' | 'understanding' | 'done';
  page: number | null;
  pages: number | null;
  /** `image` | `pdf-text` | `pdf-scan` | `text` | `docx` | `xlsx` */
  kind: string | null;
  /**
   * The turn this read belongs to.
   *
   * A filename cannot say which turn asked for a read, so a surface that
   * matched on the name alone would put one conversation's page counter under
   * another's question. Stamped by `agent_start_run` from the same tag the
   * `run_stage` events carry.
   */
  correlationId: string | null;
  messageId: string | null;
  conversationId: string | null;
}

export async function listenAttachmentProgress(
  callback: (payload: AttachmentProgress) => void
) {
  return listen<AttachmentProgress>('attachment:progress', (e) => callback(e.payload));
}

/** The line to show for a progress event. Real numbers only. */
export function describeAttachmentProgress(p: AttachmentProgress): string {
  if (p.phase === 'understanding' && p.page && p.pages && p.pages > 1) {
    return `Reading page ${p.page} of ${p.pages}…`;
  }
  switch (p.phase) {
    case 'reading':
      return 'Reading document…';
    case 'preparing':
      return 'Preparing pages…';
    case 'extracting':
      return 'Extracting text…';
    case 'understanding':
      return 'Understanding document…';
    default:
      return 'Generating answer…';
  }
}

/** The compact chip under the message: "6 pages · read locally". */
export function describeAttachmentKind(p: AttachmentProgress): string | null {
  if (!p.kind) return null;
  const how = p.kind === 'pdf-scan' || p.kind === 'image' ? 'read on device' : 'read locally';
  if (p.pages && p.pages > 1) return `${p.pages} pages · ${how}`;
  return how;
}

export interface StartRunRequest {
  prompt: string;
  classification?: Classification;
  /** Overrides the default instructions. Scripted demonstrations only. */
  /**
   * Extra context for a scripted scenario, appended *beneath* ARJUN's own
   * instructions.
   *
   * It used to be `systemPrompt` and it *replaced* them, so a demo scenario
   * could remove the retrieval rule, the citation rule and the instruction to
   * say plainly when a search found nothing — and the run would look normal
   * while answering from the model's weights. It is additive and bounded now:
   * see `compose_system_prompt` in `src-tauri/src/commands/agent.rs`.
   *
   * It grants nothing. Tools come from the plan and the gateway; the
   * classification comes from the request and the policy.
   */
  scenarioInstructions?: string;
  /**
   * Echoed back on the run's first event.
   *
   * `start` does not resolve until the run is over, so without this a caller
   * watching the event stream cannot tell its own run's events from another
   * window's until the very end. Naming the run is still the backend's job —
   * this only identifies the stream.
   */
  correlationId?: string;
  /**
   * The conversation this turn belongs to. When set, `start` will not create
   * a new conversation; the front-end already created one and reserved the
   * assistant cell via `agent_append_turn`. When absent, `start` creates a
   * fresh conversation for the first turn.
   */
  conversationId?: string;
  /**
   * Files attached to THIS turn. Nothing is remembered between runs, so one
   * turn's attachment cannot reappear in another's.
   */
  attachments?: ComposerAttachment[];
  /**
   * Where the accuracy-to-speed slider was left when this turn was sent.
   *
   * It governs only how attachments are read; which model *answers* is still
   * the backend router's decision and cannot be named from here. Omitted when
   * the caller shows no slider, and the backend then uses its default stop.
   */
  ocrDetent?: OcrDetent;
  /**
   * The id the front-end reserved for the assistant message via
   * `agent_append_turn`. Required when `conversationId` is set.
   */
  messageId?: string;
}

/** Why a model was chosen. Rendered verbatim in the task trace. */
export interface RoutingDecision {
  modelId: string;
  modelName: string;
  role: 'reasoning' | 'coding' | 'vision' | 'documentOcr' | 'embedding' | 'rerank';
  /** What the prompt was taken to be asking for. */
  intent: string;
  confidence: number;
  /** True when the first choice did not fit and something smaller was used. */
  usedFallback: boolean;
  /** Ordered and human-readable. Show these as given; do not summarise them. */
  reasons: string[];
  gpuPlanSummary: string;
  fullyOnGpu: boolean;
}

/** Where the model actually ran. */
export interface Endpoint {
  /** Always loopback. Both runtimes are reached the same way. */
  baseUrl: string;
  servedModelId: string;
  /** True when ARJUN started the server; false when an operator runs it. */
  managed: boolean;
  runtime: 'llamaCpp' | 'pythonSidecar';
}

/**
 * One planned step, and whether the run left behind the evidence for it.
 *
 * `done` is judged against what the run produced — a successful tool call, an
 * answer, a completed check — never against the model's account of itself. A
 * model that says it wrote the document and never called the tool leaves the
 * step unfinished, which is the whole point of the field.
 */
export interface PlanStep {
  ordinal: number;
  /** What the step is for, in the person's terms. */
  intent: string;
  done: boolean;
  /** What would settle this step. Shown on an unfinished one so the gap says
   *  what is missing rather than only that something is. */
  settledBy: string;
}

/** How one tool call ended. */
export type CallOutcome = 'succeeded' | 'failed' | 'refused';

/** One tool call the run made. Arguments are deliberately not carried. */
export interface ToolCallRecord {
  tool: string;
  outcome: CallOutcome;
  /** What the tool reported, trimmed — what the model saw, not a summary. */
  detail: string;
  at: string;
}

/** One thing a person was asked to allow during the run, and what they said. */
export interface ApprovalRecord {
  id: string;
  tool: string;
  target: string;
  arguments: string[];
  consequences: string;
  requestedAt: string;
  /** `approved`, `rejected`, or `pending` for one nobody answered. */
  state: string;
  decidedBy: string | null;
  decidedAt: string | null;
  because: string | null;
}

/**
 * The durable record of a milestone decision.
 *
 * Mirrors `MilestoneRecord` in `agent_runtime::memory`. A resumption
 * reads the last entry to know which gate the human approved last;
 * the UI reads the same list to render the decision history next
 * to the run.
 */
export interface MilestoneAcknowledgement {
  checkpointId: string;
  ordinal: number;
  decision: 'approved' | 'rejected';
  acknowledgedBy: string;
  /** RFC 3339, UTC. */
  at: string;
}

/**
 * Why a run ended.
 *
 * Mirrors `StopReason` in `orchestrator::plan`, tagged on `reason`. Read
 * `stoppedBecause` for the sentence to show; read this only where the shape
 * itself matters.
 */
export type StopReason =
  | { reason: 'completed' }
  | { reason: 'stepsExhausted'; taken: number; allowed: number }
  | { reason: 'timeExhausted'; allowedSeconds: number }
  | { reason: 'looping'; tool: string; repeats: number }
  | { reason: 'awaitingApproval'; tool: string }
  | { reason: 'failed'; detail: string };

/**
 * The plan a run is held to.
 *
 * Fixed before the model is told anything, and not extendable by it. Rendered
 * as given: the steps are what the run said it would do, and showing an
 * incomplete plan honestly is most of the point of having one.
 */
export interface PlanRecord {
  steps: PlanStep[];
  maxSteps: number;
  maxDurationSeconds: number;
  /** Tool names, exactly as the model would have had to write them. */
  permittedTools: string[];
  repeatLimit: number;
  stepsTaken: number;
  /** Absent while the run is still going. */
  stopReason: StopReason | null;
  /** The stop reason as a sentence, ready to show. */
  stoppedBecause: string;
}

/** How serious a verification finding is. */
export type Severity = 'blocking' | 'advisory';

export interface Finding {
  severity: Severity;
  /** What is wrong, in the words a reviewer would use. */
  detail: string;
  /** The text it is about, so a reviewer can find it. */
  excerpt: string | null;
}

/** Whether an answer may be presented as finished. */
export type Standing =
  | { standing: 'ready' }
  | { standing: 'needsReview'; blocking: number; advisory: number };

/**
 * What the verifier found in the final answer.
 *
 * It does not edit the answer and does not withhold it — it reports whether
 * every claim resolves to a passage the run actually retrieved, and whether
 * every figure matches a calculation the engine actually ran. An answer that
 * fails is shown with its findings attached, because a reviewer who cannot see
 * what the model said cannot judge it.
 */
export interface VerificationReport {
  standing: Standing;
  findings: Finding[];
  citationsResolved: number;
  figuresChecked: number;
}

export type ArtifactKind = 'document' | 'workbook' | 'text';

/** A file the run produced, re-opened and checked rather than taken on trust. */
export interface ArtifactReport {
  /** Relative to the run's working directory — the name the model wrote. */
  name: string;
  path: string;
  kind: ArtifactKind;
  /** The template a document was rendered from, so a re-check asks the same
   *  question. Null for files that have no template. */
  template: string | null;
  bytes: number;
  /** False when the file is missing, empty, will not open, or is incomplete. */
  sound: boolean;
  detail: string;
  problems: string[];
  producedAt: string;
}

/**
 * A safe, read-only excerpt of a produced file.
 *
 * The full file can be megabytes; rendering it in a React tree would freeze
 * the UI. Previews are capped server-side, and binary formats are converted
 * to a representation the browser can render directly.
 */
export type ArtifactPreview =
  | { kind: 'text'; mime: string; content: string; truncated: boolean }
  | { kind: 'markdown'; mime: string; content: string; truncated: boolean }
  | { kind: 'docxBody'; mime: string; content: string; truncated: boolean }
  | { kind: 'xlsxFirstSheet'; mime: string; content: string; truncated: boolean }
  | { kind: 'pptxSlideList'; mime: string; content: string; truncated: boolean }
  | { kind: 'image'; mime: string; dataUrl: string; truncated: boolean }
  | { kind: 'unsupported'; mime: string; reason: string };

/**
 * How a run ended.
 *
 * Mirrors `RunOutcome` in `src-tauri/src/agent_runtime/outcome.rs`, which is
 * the source of truth. Read this rather than inferring success from the fact
 * that the command returned: `text` is present for a run cut off at the output
 * cap too, and that fragment reads exactly like a short answer.
 *
 * - `completed` — the loop finished and the model stopped of its own accord.
 * - `failed` — the provider or the loop errored.
 * - `aborted` — a person, or the app shutting down, stopped it.
 * - `lengthLimited` — the model hit the output cap; the answer is a fragment.
 * - `budgetStopped` — it ran past the time or the steps its plan allowed.
 * - `policyStopped` — it needed to do something it is not permitted to do.
 */
export type RunOutcomeKind =
  | 'completed'
  | 'failed'
  | 'needsReview'
  | 'aborted'
  | 'lengthLimited'
  | 'budgetStopped'
  | 'policyStopped';

/** The typed ending of a run, with the one sentence a person is shown. */
export type RunOutcome =
  | { kind: 'completed' }
  | { kind: Exclude<RunOutcomeKind, 'completed'>; detail: string };

/** Whether an outcome means the run finished the work it set out to do. */
export function runSucceeded(outcome: RunOutcome | undefined | null): boolean {
  return outcome?.kind === 'completed';
}

/** The sentence to show for an outcome, or null when there is nothing to say. */
export function outcomeDetail(outcome: RunOutcome | undefined | null): string | null {
  if (!outcome || outcome.kind === 'completed') return null;
  return outcome.detail;
}

/**
 * How a finished or in-flight assistant turn should be described.
 *
 * ## The defect
 *
 * The chat cell derived its status as `isFailed ? 'failed' : isDone ?
 * 'verified' : ...`. "Not streaming and not failed" was rendered as
 * **Verified**, with a green tick — for every turn, including ones where the
 * verifier had never run, had found blocking problems, or where the run had
 * been stopped by a person part way through. The word on screen was the
 * strongest claim the product can make, and nothing had checked it.
 *
 * These are the states a person can act on, and each means one thing:
 *
 * - `thinking` / `usingTool` / `composing` — still going.
 * - `verified` — the verifier ran and every claim resolved.
 * - `needsReview` — the verifier ran and found something.
 * - `unverified` — there is an answer and nothing checked it.
 * - `completed` — the run finished and produced nothing to check.
 * - `stopped` — a person, a budget or a policy ended it early.
 * - `failed` — it did not finish.
 */
export type MessageStatusKind =
  | 'thinking'
  | 'usingTool'
  | 'composing'
  | 'verified'
  | 'needsReview'
  | 'unverified'
  | 'completed'
  | 'stopped'
  | 'failed';

/** What the surface knows about one assistant turn. */
export interface MessageStatusInput {
  /** Whether tokens are still arriving for this cell. */
  isStreaming: boolean;
  /** How much visible text has arrived. */
  contentLength: number;
  /** Tool calls currently running, for the live states. */
  runningTools: number;
  /**
   * How the run ended, when it is known.
   *
   * `undefined` for a turn from before the typed ending existed, or one whose
   * summary never arrived. Absence is not success — see below.
   */
  outcome?: RunOutcomeKind | null;
  /**
   * What the verifier concluded, when it ran.
   *
   * `null` means it did not run — there was nothing to check, or the run ended
   * before it got that far. That is *not* the same as passing, and this is the
   * distinction the old derivation could not make.
   */
  verification?: 'ready' | 'needsReview' | null;
}

/**
 * The status to show for one assistant turn.
 *
 * Nothing here infers a verdict from silence. A turn that stopped streaming has
 * stopped streaming; whether it is trustworthy is a separate question with a
 * separate answer, and if nobody answered it the status says so.
 */
export function messageStatus(input: MessageStatusInput): MessageStatusKind {
  if (input.isStreaming) {
    if (input.contentLength > 0) return 'composing';
    if (input.runningTools > 0) return 'usingTool';
    return 'thinking';
  }

  // How the run ended comes first: a stopped run's partial answer should not
  // be labelled by how well its fragment verifies.
  switch (input.outcome) {
    case 'needsReview':
      return 'needsReview';
    case 'failed':
      return 'failed';
    case 'aborted':
    case 'budgetStopped':
    case 'policyStopped':
    case 'lengthLimited':
      return 'stopped';
    default:
      break;
  }

  if (input.verification === 'ready') return 'verified';
  if (input.verification === 'needsReview') return 'needsReview';

  // Finished, and nothing checked it. An answer in this state is the one the
  // old derivation called "Verified".
  return input.contentLength > 0 ? 'unverified' : 'completed';
}

/** The words shown on the status pill. */
export const MESSAGE_STATUS_LABELS: Readonly<Record<MessageStatusKind, string>> = {
  thinking: 'Thinking',
  usingTool: 'Using a tool…',
  composing: 'Composing…',
  verified: 'Verified',
  needsReview: 'Needs review',
  unverified: 'Unverified',
  completed: 'Completed',
  stopped: 'Stopped',
  failed: 'Failed',
};

/**
 * What a `message_end` alone can honestly say about how the run ended.
 *
 * Narrower than the run's own ending, on purpose. This writer sees one thing:
 * the finish reason of the model's last turn. Two of those are conclusive on
 * their own — the output cap was hit, or the turn errored — and neither
 * depends on anything the backend decided afterwards. The rest are not: a turn
 * that stopped cleanly says nothing about whether the *run* was then stopped
 * by its budget or by policy, so no `outcome` is returned for them and the
 * field is left for the run's own write, which is the authority and lands
 * afterwards.
 *
 * Returning no `outcome` rather than `'completed'` is the whole point.
 * Claiming completion from a clean finish reason is the same class of mistake
 * as claiming it from a resolved request.
 */
export function endingFromFinishReason(
  finishReason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'error',
): { outcome?: RunOutcomeKind; failed: boolean } {
  if (finishReason === 'length') return { outcome: 'lengthLimited', failed: true };
  if (finishReason === 'error') return { outcome: 'failed', failed: true };
  return { failed: false };
}

/**
 * Whether this installation can still record what it does.
 *
 * Mirrors `AuditState` in `src-tauri/src/agent_runtime/audit_health.rs`.
 *
 * ARJUN's claim is that every run leaves a checkable record. When the store
 * that holds it cannot be written, the desktop stays usable read-only — past
 * runs open, settings open — and no run starts. A surface that showed nothing
 * would leave a person wondering why their prompt did nothing at all.
 */
export type AuditState =
  | { state: 'durable' }
  | {
      state: 'degraded';
      /** What went wrong, in the sentence to show. */
      because: string;
      /** True when the store never opened; false when a write failed later. */
      atStartup: boolean;
    };

/**
 * Whether this session's chats will still be here tomorrow.
 *
 * Mirrors `ConversationState` in
 * `src-tauri/src/agent_runtime/conversations.rs`.
 *
 * The store's failure path used to be a fixed temp directory opened silently,
 * so the chat behaved exactly as normal while the person's real conversations
 * were somewhere else — and the next session found the previous one's threads
 * sitting there looking like history. A session writing to scratch now says so.
 */
export type ConversationStorageState =
  | { state: 'durable' }
  | {
      state: 'ephemeral';
      /** What went wrong, in the sentence to show. */
      because: string;
      /** Where this session's chats are going, so an operator can find them. */
      directory: string;
    };

export interface RunSummary {
  runId: string;
  text: string;
  turns: number;
  /**
   * How this run ended. Never inferred from the command resolving — see
   * {@link RunOutcome}.
   */
  outcome: RunOutcome;
  routing: RoutingDecision;
  endpoint: Endpoint;
  plan: PlanRecord;
  /** Absent when the run produced no answer to check. */
  verification: VerificationReport | null;
  artifacts: ArtifactReport[];
  /**
   * The conversation this run was started in. Every run now lives in a
   * conversation; older callers that did not set `conversationId` will see
   * this set automatically to the newly-created one.
   */
  conversationId?: string;
  /**
   * The id of the assistant message this run produced. The chat surface
   * uses it to correlate `message_end` with the right `Message` cell.
   */
  messageId?: string;
  /**
   * Set when the run happened but could not be fully written down.
   *
   * The answer is real and the work was done; what is missing is the record of
   * it. Worth showing because it changes what the answer can be used for — an
   * approval note nobody can produce the provenance for is not one anybody
   * should sign.
   */
  recordFailure?: string | null;
  /** How the audit stores were doing when this run finished. */
  audit: AuditState;
}

/** One passage the run stood on, as its `[E3]` marker refers to it. */
export interface EvidenceRecord {
  marker: number;
  citation: string;
  documentName: string;
  page: number;
  excerpt: string;
}

/** One step of a calculation the engine performed. */
export interface CalculationStep {
  description: string;
  result: string;
}

export interface CalculationRecord {
  expression: string;
  inputs: string[];
  steps: CalculationStep[];
  value: number;
  unit: string;
  formatted: string;
  rounding: string;
  /** Always true — the engine computed this, not a model. */
  deterministic: boolean;
}

/** A side effect a run already performed. Read before a resumption acts. */
export interface CompletedEffect {
  tool: string;
  /** What it acted on — a file name, a path, an identifier. */
  target: string;
  at: string;
}

/**
 * A run's own bounded memory.
 *
 * Identifiers, not content: `evidenceIds` holds `E3`, never the passage. That
 * is what keeps this small enough to carry in context for a whole run and
 * cheap enough to persist with every task record.
 */
export interface RunMemory {
  goal: string;
  stage: { ordinal: number; intent: string };
  decisions: { what: string; because: string; at: string }[];
  evidenceIds: string[];
  calculationIds: string[];
  artifactIds: string[];
  openQuestions: string[];
  nextAction: string;
  completed: CompletedEffect[];
  /** How many entries the caps dropped, per list. Shown rather than hidden. */
  dropped: Record<string, number>;
}

/** Everything kept about one finished run. */
export interface TaskRecord {
  runId: string;
  prompt: string;
  startedAt: string;
  finishedAt: string;
  durationSeconds: number;
  userId: string;
  routing: RoutingDecision;
  endpoint: Endpoint;
  plan: PlanRecord;
  answer: string;
  turns: number;
  /**
   * Every time the run's history was replaced by a summary, in order.
   *
   * Optional so records written before this existed still parse. A run that
   * never compacted has an empty list — indistinguishable from an older record,
   * and for the reader's purposes the same thing.
   */
  compactions?: CompactionRecord[];
  /**
   * The run's bounded notes as they finished.
   *
   * What a resumption reads: the goal, the stage it reached, and — the part
   * that makes resuming safe rather than merely faster — the side effects that
   * already happened and must not happen again.
   */
  workingNotes?: RunMemory | null;
  /** Where the context window stood when the run ended. */
  contextLedger?: ContextLedgerRecord | null;
  verification: VerificationReport | null;
  artifacts: ArtifactReport[];
  evidence: EvidenceRecord[];
  calculations: CalculationRecord[];
  toolCalls: ToolCallRecord[];
  approvals: ApprovalRecord[];
  /** Set when the run ended badly, in the words shown to the person. */
  failure: string | null;
  /**
   * How the run ended, typed.
   *
   * Absent on records written before this existed, which reads as "not
   * recorded" rather than as a guess at which ending they had.
   */
  outcome?: RunOutcome | null;
}

/** A row on the Tasks screen. */
export interface TaskSummary {
  runId: string;
  /** Who ran it. The backend has already filtered the list to what the
   *  signed-in person may read. */
  userId: string;
  prompt: string;
  startedAt: string;
  finishedAt: string;
  durationSeconds: number;
  modelName: string;
  intent: string;
  turns: number;
  artifactCount: number;
  evidenceCount: number;
  toolCallCount: number;
  /** Steps planned but never reached. Non-zero is the signal to look. */
  unfinishedSteps: number;
  approvalsPending: number;
  /**
   * Times the run's older history was replaced by a summary so it could
   * continue.
   *
   * Non-zero on a short task is the signal that the routed model's window is
   * too small for the work it is being given — which is a routing decision to
   * revisit, not a fault in the run.
   */
  compactionCount: number;
  stoppedBecause: string;
  /** False when it failed, needs review, or produced an unsound file. */
  ready: boolean;
  failure: string | null;
  /** Where the run stands. The only value that can say `degraded_needs_human`. */
  state: RunState;
  /** True while it is still going. A live row has no finish time. */
  live: boolean;
}

/**
 * Where a run is.
 *
 * The nine live states name things a person might need to do something about —
 * "waiting for you to approve an action" is not the same as "the model is
 * thinking", and a single `running` cannot tell them apart.
 *
 * The six endings are deliberately distinct. `stoppedByBudget` is the budget
 * doing its job and `stoppedByPolicy` is the policy doing its job; neither is a
 * fault, and painting them the same colour as `failed` teaches people to skip
 * the row that actually broke. `degradedNeedsHuman` is not a verdict at all —
 * nothing decided it, the application closed on top of the run.
 */
export type RunState =
  | 'created'
  | 'classified'
  | 'routed'
  | 'planned'
  | 'running'
  | 'awaiting_approval'
  | 'executing_tool'
  | 'tool_result_recorded'
  | 'verifying'
  | 'completed'
  | 'cancelled'
  | 'failed'
  | 'stopped_by_budget'
  | 'stopped_by_policy'
  | 'degraded_needs_human';

/** The endings. Nothing follows one. */
export const TERMINAL_STATES: readonly RunState[] = [
  'completed',
  'cancelled',
  'failed',
  'stopped_by_budget',
  'stopped_by_policy',
  'degraded_needs_human',
];

export const isTerminal = (state: RunState) => TERMINAL_STATES.includes(state);

/**
 * A side effect nobody can account for.
 *
 * It was in flight when the process went away, so the file it names may or may
 * not have been written. Deliberately not retried: repeating it could do it
 * twice, and assuming it happened could mean it never does. A person has to go
 * and look.
 */
export interface UnknownEffect {
  idempotencyKey: string;
  tool: string;
  /** A file name — a reference to go and check, never contents. */
  target: string;
  at: string;
}

/** One of those, as the reconciliation queue lists it across every run. */
export interface RecordedEffect {
  idempotencyKey: string;
  runId: string;
  tool: string;
  argsFingerprint: string;
  status: 'pending' | 'succeeded' | 'failed' | 'unknown';
  result: string;
  target: string;
  at: string;
}

/** One thing a run did, as the recovered trace shows it. */
export interface ActivityRecord {
  toolCallId: string;
  tool: string;
  /** `running`, `done`, `failed`, `refused` or `replayed`. */
  status: string;
  at: string;
}

/**
 * What a run has done so far, without replaying its history.
 *
 * The thing a window reads when it mounts holding a run id — after a remount,
 * or after the whole application was restarted. Deliberately carries a
 * *reference* to the answer rather than the answer: a finished run's text is in
 * its task record, and one still going has no answer yet.
 */
/**
 * How the context window was divided at one moment.
 *
 * Counts only — how many tokens each section held, never what was in them. That
 * is what makes it safe to show on a screen read more widely than the
 * transcript it describes.
 */
export interface ContextLedgerRecord {
  system: number;
  skill: number;
  toolSchema: number;
  evidence: number;
  notes: number;
  transcript: number;
  compaction: number;
  /** Held back for the model's output. Committed rather than occupied. */
  reserve: number;
  /** Everything except `reserve`. */
  occupied: number;
  /** `occupied + reserve` — what the next turn has to fit inside. */
  committed: number;
  /** The model's window. Zero when the runtime was not told one. */
  window: number;
  /** `window - committed`. Negative means the next turn does not fit. */
  headroom: number;
  /**
   * The itemised rows under the sections.
   *
   * Optional so a record written before the itemisation existed still parses,
   * and absent is treated as "not itemised" rather than "nothing in it" — the
   * screen falls back to section rows instead of drawing an empty breakdown
   * under a full bar.
   */
  entities?: ContextEntity[];
  /** Every model call this run made, estimate against measurement. */
  reconciliations?: TurnReconciliation[];
  /** Sections whose rows do not add up to them. Empty in normal operation. */
  itemisationErrors?: { section: string; fromEntities: number; fromSection: number }[];
}

/** The eight sections, as the ledger names them. */
export type LedgerSectionName =
  | 'system'
  | 'skill'
  | 'toolSchema'
  | 'evidence'
  | 'notes'
  | 'transcript'
  | 'compaction'
  | 'reserve';

/** How an entity's token figure was arrived at. Never inferred. */
export type ContextMeasurement = 'exact' | 'provider' | 'estimated';

/**
 * Where one entity stands.
 *
 * `pending` is a document still being read. It reports no tokens, because
 * nothing has counted it yet — see the runtime's `context-entities.ts`.
 */
export type ContextEntityStatus = 'active' | 'pending' | 'summarised' | 'dropped';

/** One addressable thing occupying the context window. */
export interface ContextEntity {
  id: string;
  section: LedgerSectionName;
  label: string;
  tokens: number;
  measurement: ContextMeasurement;
  status: ContextEntityStatus;
  /** Protected from eviction by the person, and honoured by the compactor. */
  pinned: boolean;
  sequence: number;
  detail?: Record<string, string | number | boolean | null>;
}

/**
 * One model call's prediction measured against what it actually cost.
 *
 * `actualIn` is `null` when the server reported no usage. Left null rather than
 * back-filled from the estimate, so a screen can tell "checked, and it matched"
 * from "nobody checked" — the distinction the whole reconciliation exists for.
 */
export interface TurnReconciliation {
  turn: number;
  at: string;
  estimatedIn: number;
  actualIn: number | null;
  actualOut: number | null;
  /** `actualIn / estimatedIn`, or null when there is nothing to divide. */
  driftRatio: number | null;
}

/**
 * What one attachment cost and how much of it reached the model.
 *
 * Emitted on `attachment:context` by `agent_start_run` at the moment the
 * injection decision is taken. Mirrors `AttachmentContextEvent` in
 * `commands::agent`.
 */
export interface AttachmentContextEvent {
  name: string;
  /** Content address — the stable row id for this file in the meter. */
  sha256: string;
  pages: number;
  documentTokens: number;
  /** Equal to `documentTokens` only when the whole document went in. */
  injectedTokens: number;
  strategy: 'full' | 'chunked' | 'referenceOnly';
  /** Shown verbatim: how much of the document the answer rests on. */
  explanation: string;
}

/** Subscribes to per-attachment context costs. */
export async function listenAttachmentContext(
  callback: (payload: AttachmentContextEvent) => void,
) {
  return listen<AttachmentContextEvent>('attachment:context', (e) => callback(e.payload));
}

/** One time a run's older history was replaced by a summary. */
export interface CompactionRecord {
  /** Which compaction of this run, 1-based. */
  ordinal: number;
  at: string;
  tokensBefore: number;
  tokensAfter: number;
  messagesSummarised: number;
  /**
   * True when this pass extended the summary already held. A `false` on
   * anything but the first means the run started a second summary, and the
   * earlier half of its history is described twice or not at all.
   */
  refinedExistingSummary: boolean;
  /** Raw tool results replaced by an evidence reference, cumulatively. */
  toolResultsCleared: number;
  ledger: ContextLedgerRecord;
}

export interface TaskSnapshot {
  runId: string;
  /** The last event folded in. Ask for events after this to catch up. */
  seq: number;
  schemaVersion: number;
  state: RunState;
  startedAt: string;
  updatedAt: string;
  /** When the run must stop, if it has a deadline. */
  deadline: string | null;
  /** Who started it. */
  actor: string;
  prompt: string;
  modelName: string;
  classification: string | null;
  plan: PlanRecord | null;
  activity: ActivityRecord[];
  turns: number;
  compactions: number;
  /**
   * What each of those compactions actually did.
   *
   * The count says the window ran out; these say what filled it. A run that
   * compacted three times and cannot say which section grew is a run nobody can
   * diagnose afterwards — and the usual answer, "one enormous tool result", has
   * a remedy that costs the run nothing.
   *
   * Optional so a snapshot from an older backend still parses; absent and empty
   * mean the same thing to every reader here.
   */
  compactionEvents?: CompactionRecord[];
  /** Names of the files it produced. */
  artifacts: string[];
  approvalsPending: number;
  /** Side effects that were in flight when the process went away. Non-empty is
   *  why a run is `degraded_needs_human`. */
  unknownEffects: UnknownEffect[];
  stoppedBecause: string | null;
  failure: string | null;
  answerHash: string | null;
  answerChars: number;
  /** Events that were on disk and could not be read. Non-empty means the
   *  history has a hole in it, and the screen says so rather than pretending
   *  the trace is complete. */
  unreadableEvents: UnreadableEvent[];
  /** Events that could not legally follow the state they arrived in. Recorded,
   *  not applied — surfaced because two writers disagreeing about a run is
   *  worth somebody knowing. */
  anomalies: string[];
}

export interface UnreadableEvent {
  seq: number;
  eventId: string;
  /** What is wrong with it, in words. */
  problem: string;
}

/** What a durable event is called. */
export type TaskEventType =
  | 'runCreated'
  | 'runClassified'
  | 'runRouted'
  | 'planReady'
  | 'runStarted'
  | 'planStep'
  | 'planStopped'
  | 'turnEnded'
  | 'contextCompacted'
  | 'toolAuthorized'
  | 'toolRefused'
  | 'toolSucceeded'
  | 'toolFailed'
  | 'toolReplayed'
  | 'toolEffectPending'
  | 'toolEffectUnknown'
  | 'toolEffectReconciled'
  | 'artifactProduced'
  | 'approvalRequested'
  | 'approvalDecided'
  | 'milestoneReached'
  | 'milestoneAcknowledged'
  | 'verificationStarted'
  | 'runCompleted'
  | 'runFailed'
  | 'runCancelled'
  | 'runStoppedByBudget'
  | 'runStoppedByPolicy'
  | 'runDegraded'
  // Read back from a database written by an earlier build; never sent.
  | 'runTimedOut'
  | 'runInterrupted';

/**
 * One event from the durable history.
 *
 * Distinct from `AgentEvent`, which is the best-effort live stream. This one
 * was written down, is ordered by `seq`, and is what a window catching up
 * reads. Payloads are redacted at the source: fields that could carry document
 * text arrive as `{ sha256, chars }`, never as the text.
 */
export interface TaskEvent {
  runId: string;
  eventId: string;
  seq: number;
  eventType: TaskEventType;
  at: string;
  actor: string;
  schemaVersion: number;
  payload: Record<string, unknown>;
  payloadHash: string;
}

/**
 * One durable event as it arrives on the live channel.
 *
 * The same row [`TaskEvent`] describes, minus the payload hash — a client has
 * no way to check it and carrying it would only suggest otherwise.
 */
export interface DurableEvent {
  runId: string;
  seq: number;
  eventId: string;
  eventType: TaskEventType;
  at: string;
  actor: string;
  schemaVersion: number;
  payload: Record<string, unknown>;
}

export interface TaskEventPage {
  events: TaskEvent[];
  unreadable: UnreadableEvent[];
  /** The highest position accounted for, readable or not. */
  lastSeq: number;
}

/**
 * Lifecycle events from the agent loop.
 *
 * Mirrors `AgentEvent` in `@openclaw/agent-core`, narrowed to what the UI
 * renders. Tool *arguments* are stripped before they leave the backend — they
 * can carry document text and the audit record already holds them under access
 * control, so they do not travel a second path just to be displayed.
 */
/**
 * A stage of the work a run does before, during, and after generation.
 *
 * Mirrors `Stage` in `src-tauri/src/agent_runtime/stages.rs`. Adding a stage
 * on one side and not the other is a type error here rather than a silently
 * ignored event.
 */
export type RunStageName =
  | 'accepted'
  | 'readingAttachment'
  | 'attachmentsRead'
  | 'routing'
  | 'routed'
  | 'loadingModel'
  | 'modelReady'
  | 'planning'
  | 'generating'
  | 'thinking'
  | 'verifying'
  | 'complete';

export type AgentEvent =
  | { type: 'agent_start' }
  | { type: 'agent_end' }
  | { type: 'turn_start' }
  | { type: 'turn_end' }
  | { type: 'tool_execution_start'; toolCallId: string; toolName: string }
  | { type: 'tool_execution_update'; toolCallId: string; toolName: string }
  | {
      /**
       * Older history was replaced by a summary so the run could continue.
       *
       * Worth showing: the model's answers after this point are grounded in a
       * summary of the earlier turns rather than the turns themselves, and an
       * operator reading the trace should know that.
       */
      type: 'context_compacted';
      tokensBefore: number;
      tokensAfter: number;
      messagesSummarised: number;
    }
  | {
      /**
       * Where the context window stands, right now.
       *
       * Emitted every model turn and after every compaction, which is what
       * makes the meter live. Before this existed the surface could only learn
       * the ledger when the run finished, so the number a person watched while
       * deciding whether to attach one more file was always describing the
       * previous run.
       */
      type: 'context_ledger';
      /** `turn` or `compaction` — why this reading was taken. */
      reason: string;
      ledger: ContextLedgerRecord;
    }
  | {
      type: 'tool_execution_end';
      toolCallId: string;
      toolName: string;
      isError: boolean;
      /** False when the gateway refused before the tool ran. */
      executionStarted?: boolean;
    }
  | {
      /**
       * The plan this run is held to, published before the first turn.
       *
       * Emitted by the backend rather than the loop: the plan is fixed before
       * the model is told anything, so it is known before there is any loop
       * activity to report.
       */
      type: 'plan_ready';
      plan: PlanRecord;
      /** Whatever the caller sent on `StartRunRequest`, echoed once. */
      correlationId?: string | null;
    }
  | {
      /** A step spent. Sent after the tool ran, whatever it returned. */
      type: 'plan_step';
      tool: string;
      stepsTaken: number;
      maxSteps: number;
      stepsDone: number;
      stepsPlanned: number;
    }
  | {
      /**
       * The run hit its budget, or went in circles, and will do nothing more.
       *
       * Worth showing the moment it happens: the loop still has to wind down
       * and produce a final answer, and an operator watching a suddenly quiet
       * trace should know it is stopping rather than thinking.
       */
      type: 'plan_stopped';
      reason: string;
      tool: string;
    }
  | {
      /**
       * A milestone the model finished. The plan pauses here so a
       * person can confirm the model is on the right track before
       * the next leg of work starts. The UI shows a gate; once
       * the user approves, the phase returns to `running`.
       */
      type: 'milestone_reached';
      checkpointId: string;
      ordinal: number;
      summary: string;
    }
  | {
      /**
       * The user signed off on a milestone. The loop has been
       * resumed; subsequent steps are normal again.
       */
      type: 'milestone_acknowledged';
      checkpointId: string;
      acknowledgedBy: string;
    }
  // ─── Message streaming (relayed from OpenClaw via the Rust runtime) ───
  //
  // These three events are how the chat surface shows the model's answer as
  // it is being produced. They are best-effort, may be dropped on a slow
  // listener, and are NOT durable: on remount the chat surface reads the
  // final content from the conversation store, not from a replayed stream.
  //
  // The `messageId` is generated by the Rust runtime when an assistant
  // `Message` row is created and is stable for the lifetime of that
  // message. A UI that opens a conversation mid-stream uses the
  // `agent_run_conversation` Tauri command to look it up.
  | {
      type: 'message_start';
      messageId: string;
      role: 'assistant';
    }
  | {
      type: 'message_update';
      messageId: string;
      /** A token-or-chunk string. May be empty for a delta carrying only metadata. */
      delta: string;
    }
  | {
      type: 'message_end';
      messageId: string;
      /** Why the model stopped. Mirrors the runtime's stop reason. */
      finishReason: 'stop' | 'length' | 'tool_calls' | 'content_filter' | 'error';
      tokensIn?: number;
      tokensOut?: number;
    }
  // ─── Progress (emitted by the Rust command and by the runtime) ────────
  //
  // Neither of these is durable. They exist so the surface can say what is
  // happening while it happens; what actually happened is in the task record.
  | {
      /**
       * A stage of the run, emitted by `agent_start_run` as it reaches it.
       *
       * `runId` on the envelope is the caller's `correlationId` until the
       * server has issued its own id, which is what lets a stage emitted
       * before the run existed still reach the cell waiting for it.
       */
      type: 'run_stage';
      stage: RunStageName;
      /** Milliseconds since the command was entered. Measured, not modelled. */
      elapsedMs: number;
      correlationId?: string | null;
      messageId?: string | null;
      conversationId?: string | null;
      /** Stage-specific detail. Every field is something the backend measured. */
      [detail: string]: unknown;
    }
  | {
      /**
       * The model is reasoning, or has stopped.
       *
       * `characters` is the running size of the block, `elapsedMs` how long
       * it has been going, and `delta` the reasoning itself since the last
       * frame — absent on a frame carrying only the counter.
       *
       * **Live only.** The reasoning is shown while it happens and is not
       * kept: it is held in a buffer separate from the answer, is never
       * written to `Message.content`, never sent as `finalContent`, and never
       * reaches the verifier or the audit record. Reopening a conversation
       * shows the answer and no thought, which is correct — the thought was
       * never anywhere to be re-read from.
       *
       * See the translator in `agent-runtime/src/run.ts`, the only place the
       * reasoning stream is read and the only place this is produced.
       */
      type: 'model_thinking';
      messageId: string;
      state: 'start' | 'active' | 'end';
      characters: number;
      elapsedMs: number;
      delta?: string;
    };

/** One event, tagged with the run it belongs to. */
export interface AgentEventEnvelope {
  runId: string;
  event: AgentEvent;
}

/** Backend event channel. One stream for every run; filter on `runId`. */
const AGENT_EVENT = 'agent://event';

/**
 * The durable channel.
 *
 * Every message here names a row that is on disk and carries its sequence
 * number. That number is the whole difference between the two channels: a
 * client that receives seq 14 having applied seq 12 knows it missed one and can
 * go and fetch it. On `agent://event` a lost message and a quiet run look
 * identical.
 */
const AGENT_DURABLE_EVENT = 'agent://durable';

export const agentService = {
  /**
   * Runs one prompt to completion.
   *
   * Resolves with the final answer. Subscribe first if you want to show
   * anything before it settles — a run with tool calls takes a while, and an
   * interface that shows nothing until the end looks broken.
   */
  start(request: StartRunRequest): Promise<RunSummary> {
    return getBackendService().invoke<RunSummary>('agent_start_run', { request });
  },

  /**
   * Stops a run in flight.
   *
   * Resolves `false` when there was nothing to stop, which is an ordinary race
   * rather than a failure — do not surface it as an error.
   */
  abort(runId: string): Promise<boolean> {
    return getBackendService().invoke<boolean>('agent_abort_run', { runId });
  },

  /**
   * Corrects a run already in flight, without stopping it.
   *
   * Applied at the next point the loop is safe to interrupt — before an
   * unstarted tool call or the next model turn — never mid-tool. Resolves
   * `false` when the run had already finished, which is an ordinary race and
   * should not be surfaced as an error.
   */
  steer(runId: string, text: string): Promise<boolean> {
    return getBackendService().invoke<boolean>('agent_steer_run', { runId, text });
  },

  /**
   * Subscribes to run lifecycle events.
   *
   * Returns the unsubscribe function. Call it on unmount: the backend keeps
   * emitting for the life of the session, and a listener left behind will
   * update a component that is no longer mounted.
   *
   * Pass `runId` to receive only one run's events.
   */
  async subscribe(
    handler: (envelope: AgentEventEnvelope) => void,
    runId?: string,
  ): Promise<UnlistenFn> {
    return listen<AgentEventEnvelope>(AGENT_EVENT, ({ payload }) => {
      if (runId && payload.runId !== runId) return;
      handler(payload);
    });
  },

  /**
   * Whether the agent runtime can start on this machine.
   *
   * Starts it if it is not already running, so this doubles as the "can this
   * deployment run an agent at all" check for the health screen. Rejects with a
   * readable reason when the bundle is missing or Node is absent.
   */
  health(): Promise<{ ready: boolean; pid: number; node: string }> {
    return getBackendService().invoke('agent_runtime_health');
  },

  /**
   * Every task this machine has run, newest first.
   *
   * Read from disk on each call rather than cached: a record is written by the
   * run that produced it, and a list held in memory goes stale the moment a
   * second window runs something.
   */
  history(): Promise<TaskSummary[]> {
    return getBackendService().invoke<TaskSummary[]>('agent_task_history');
  },

  /** One task in full — its plan, routing, evidence, working and artifacts. */
  task(runId: string): Promise<TaskRecord> {
    return getBackendService().invoke<TaskRecord>('agent_task', { runId });
  },

  /**
   * What a run has done so far, without replaying its history.
   *
   * Call this when a component mounts holding a run id it did not start —
   * after a remount, or after the whole application was restarted. Resolves
   * `null` for a run id nothing is known about, which is an ordinary answer
   * rather than an error.
   */
  snapshot(runId: string): Promise<TaskSnapshot | null> {
    return getBackendService().invoke<TaskSnapshot | null>('agent_task_snapshot', { runId });
  },

  /**
   * A run's durable events after `afterSeq`, in order.
   *
   * The catch-up half of recovery: hold a snapshot at sequence 12, ask for
   * everything after 12, apply it. Not the same thing as `subscribe`, which is
   * the live best-effort stream and can drop a line.
   */
  events(runId: string, afterSeq = 0): Promise<TaskEventPage> {
    return getBackendService().invoke<TaskEventPage>('agent_task_events', { runId, afterSeq });
  },

  /**
   * Subscribes to the durable event stream.
   *
   * Prefer this for anything that has to be *correct*; `subscribe` is for
   * anything that has to be *immediate*. The two are not alternatives — a
   * window normally watches both, taking responsiveness from one and
   * reconciliation from the other.
   */
  async subscribeDurable(
    handler: (event: DurableEvent) => void,
    runId?: string,
  ): Promise<UnlistenFn> {
    return listen<DurableEvent>(AGENT_DURABLE_EVENT, ({ payload }) => {
      if (runId && payload.runId !== runId) return;
      handler(payload);
    });
  },

  /**
   * Side effects nobody can account for, across every run.
   *
   * Requires the permission to approve outputs: deciding whether work happened
   * is the same kind of judgement as signing off that it was done properly.
   */
  unknownEffects(): Promise<RecordedEffect[]> {
    return getBackendService().invoke<RecordedEffect[]>('agent_unknown_effects');
  },

  /**
   * Records what a person found out about an interrupted side effect.
   *
   * Resolves `false` when there was nothing left to reconcile — somebody else
   * got there first, which is an ordinary race and not an error.
   */
  reconcileEffect(runId: string, idempotencyKey: string, happened: boolean): Promise<boolean> {
    return getBackendService().invoke<boolean>('agent_reconcile_effect', {
      runId,
      idempotencyKey,
      happened,
    });
  },

  /**
   * Signs off on a milestone the model just reached.
   *
   * The run is paused at a checkpoint; this call records the
   * person's decision and resumes the loop. The decision is
   * durable: a later resume reads the same `MilestoneRecord` and
   * knows which gate was last acknowledged, so the audit log
   * shows the chain of decisions rather than the model's text.
   *
   * `decision` is `'approved'` to continue, `'rejected'` to stop
   * the run cleanly at the gate. A rejection is not a failure;
   * it is a deliberate end with the work that was done so far
   * preserved.
   */
  acknowledgeMilestone(
    runId: string,
    checkpointId: string,
    decision: 'approved' | 'rejected',
  ): Promise<MilestoneAcknowledgement> {
    return getBackendService().invoke<MilestoneAcknowledgement>('agent_acknowledge_milestone', {
      runId,
      checkpointId,
      decision,
    });
  },

  /**
   * The runs the record still considers live.
   *
   * How a window that has just opened finds a run to reattach to. Read from the
   * durable record rather than from anything in memory, because after a restart
   * there is nothing in memory and a run left mid-flight is exactly what
   * somebody needs to be told about.
   */
  activeTasks(): Promise<TaskSnapshot[]> {
    return getBackendService().invoke<TaskSnapshot[]>('agent_active_tasks');
  },

  /**
   * Re-opens a finished task's files and reports what is in them *now*.
   *
   * The saved record says what the check found when the run ended; this says
   * what it finds today. The two disagreeing is worth knowing — a deliverable
   * can be moved, replaced or truncated long after the run that made it.
   */
  taskArtifacts(runId: string): Promise<ArtifactReport[]> {
    return getBackendService().invoke<ArtifactReport[]>('agent_task_artifacts', { runId });
  },

  /**
   * Shows a produced file in the operating system's file manager.
   *
   * Reveals rather than opens: handing a path a model named to the shell to
   * *open* would let that file decide which application runs.
   */
  revealArtifact(runId: string, name: string): Promise<void> {
    return getBackendService().invoke<void>('agent_reveal_artifact', { runId, name });
  },

  /**
   * Fetches a safe preview of a produced file.
   *
   * Distinct from `revealArtifact`: reveal hands the file to the OS file
   * manager, preview returns content the UI can render inline. The Rust
   * side caps both bytes and image size so a runaway file cannot lock up
   * the renderer.
   */
  previewArtifact(runId: string, name: string): Promise<ArtifactPreview> {
    return getBackendService().invoke<ArtifactPreview>('artifact_preview', { runId, name });
  },

  // ─── Conversation methods (chat) ────────────────────────────────────
  //
  // These back the chat surface. The split between `start` and
  // `appendTurn` mirrors the back-end: a follow-up is a `start` with
  // `conversationId` already set, after `appendTurn` has reserved the
  // assistant cell.

  /**
   * Create a new conversation with one system welcome message.
   *
   * The chat surface calls this once on first open; later turns are added
   * via `appendTurn` and `start`.
   */
  createConversation(
    title: string,
    welcome?: string,
  ): Promise<Conversation> {
    return getBackendService().invoke<Conversation>('agent_create_conversation', {
      title,
      welcome: welcome ?? null,
    });
  },

  /** Read one conversation, including its `messages[]` and `runs[]`. */
  getConversation(id: string): Promise<Conversation | null> {
    return getBackendService().invoke<Conversation | null>('agent_get_conversation', { id });
  },

  /** All conversations, newest first by `lastActivityAt`. */
  listConversations(): Promise<Conversation[]> {
    return getBackendService().invoke<Conversation[]>('agent_list_conversations');
  },

  /**
   * Delete a conversation by id. Idempotent: a delete of a missing
   * id resolves to `false` rather than an error, so the UI can retry
   * without surfacing a misleading "not found" toast.
   *
   * The on-disk JSON file is removed. The in-memory run→conversation
   * index is rebuilt lazily on the next `appendTurn` for a run, so
   * a deleted conversation has no lingering references on the
   * back-end side.
   */
  deleteConversation(id: string): Promise<boolean> {
    return getBackendService().invoke<boolean>('agent_delete_conversation', { id });
  },

  /**
   * Reserve the user message and the streaming assistant cell for a new
   * turn. The assistant cell is empty and `Streaming`; the front-end
   * accumulates tokens into it as `message_update` events arrive.
   *
   * The actual run is started separately via `start({ conversationId, ... })`.
   */
  appendTurn(
    conversationId: string,
    runId: string,
    messageId: string,
    userPrompt: string,
  ): Promise<Conversation | null> {
    return getBackendService().invoke<Conversation | null>('agent_append_turn', {
      conversationId,
      runId,
      messageId,
      userPrompt,
    });
  },

  /**
   * Persist the current streaming content of an assistant message. Called
   * from the chat surface as tokens arrive, so a remount that lands mid-
   * stream reads the latest text from disk rather than from a (best-
   * effort) event channel.
   */
  updateStreamingContent(
    conversationId: string,
    messageId: string,
    content: string,
  ): Promise<Conversation | null> {
    return getBackendService().invoke<Conversation | null>(
      'agent_update_streaming_content',
      { conversationId, messageId, content },
    );
  },

  /**
   * Mark an assistant message as `done` (or `failed`) on the conversation.
   * Called by the front-end on `message_end` or on run completion; the
   * back-end may also write the final state itself when `start` resolves.
   */
  completeMessage(args: {
    conversationId: string;
    messageId: string;
    runId: string;
    finalContent?: string;
    elapsedMs?: number;
    modelName?: string;
    modelRole?: string;
    usedFallback?: boolean;
    error?: string;
    /**
     * How the run ended. Sent only by a caller that knows — the run itself.
     * A `message_end` writer omits it and leaves whatever the run recorded.
     */
    outcome?: RunOutcomeKind;
    /** What the verifier concluded. Sent only by the run. */
    verification?: 'ready' | 'needsReview';
    failed: boolean;
    tokensIn?: number;
    tokensOut?: number;
  }): Promise<Conversation | null> {
    return getBackendService().invoke<Conversation | null>('agent_complete_message', {
      conversationId: args.conversationId,
      messageId: args.messageId,
      runId: args.runId,
      finalContent: args.finalContent ?? null,
      elapsedMs: args.elapsedMs ?? null,
      modelName: args.modelName ?? null,
      modelRole: args.modelRole ?? null,
      usedFallback: args.usedFallback ?? null,
      error: args.error ?? null,
      outcome: args.outcome ?? null,
      verification: args.verification ?? null,
      failed: args.failed,
      tokensIn: args.tokensIn ?? null,
      tokensOut: args.tokensOut ?? null,
    });
  },

  /**
   * Reverse-lookup: which conversation does this run belong to?
   *
   * Used by the chat surface when a `message_*` event arrives on
   * `agent://event` to figure out which `Message` to update. The in-memory
   * index is set by `appendTurn` and cleared by `completeMessage`; on a
   * remount, the chat surface rebuilds the index by reading the
   * conversation's `runs[]` from disk.
   */
  /**
   * Whether this session's chats will persist.
   *
   * Asked once and shown as a banner. A person who has just been refused a new
   * conversation needs to know why, and one who has not yet tried needs to know
   * before they type.
   */
  conversationHealth(): Promise<ConversationStorageState> {
    return getBackendService().invoke<ConversationStorageState>('agent_conversation_health');
  },

  runConversation(runId: string): Promise<string | null> {
    return getBackendService().invoke<string | null>('agent_run_conversation', { runId });
  },

  /**
   * Read a single message by id. Used by the chat surface to recover
   * mid-stream state for an in-flight run after a remount.
   */
  getMessage(conversationId: string, messageId: string): Promise<ChatMessage | null> {
    return getBackendService().invoke<ChatMessage | null>('agent_get_message', {
      conversationId,
      messageId,
    });
  },
};

// ─── Conversation types (chat) ────────────────────────────────────────

/** The role a participant plays in a conversation. */
export type ChatRole = 'user' | 'assistant' | 'system';

/** The status of a single `ChatMessage` in a conversation. */
export type ChatMessageStatus = 'streaming' | 'done' | 'failed';

/**
 * One turn in a conversation. The assistant `content` starts empty and is
 * filled token-by-token as `message_update` events arrive. The user
 * `content` is set in full when the user submits.
 */
export interface ChatMessage {
  id: string;
  conversationId: string;
  role: ChatRole;
  content: string;
  status: ChatMessageStatus;
  /** Present on assistant messages; absent on user and system. */
  runId?: string | null;
  createdAt: string;
  completedAt?: string | null;
  elapsedMs?: number | null;
  error?: string | null;
  modelName?: string | null;
  modelRole?: string | null;
  usedFallback?: boolean | null;
  /** Token counts from the model (assistant messages only). */
  tokensIn?: number | null;
  tokensOut?: number | null;
  /**
   * How the run that produced this message ended.
   *
   * Persisted per message so the surface can say what happened to a turn it is
   * rendering from disk, long after the run's events have gone. `null` on
   * messages written before this field existed, and on user and system
   * messages, which no run produced.
   */
  outcome?: RunOutcomeKind | null;
  /**
   * What the verifier concluded about this answer, when it ran.
   *
   * Persisted alongside the outcome for the same reason: a cell rendered from
   * disk has no run events to consult, and "not streaming" is not a verdict.
   * `null` means the verifier did not run — there was nothing to check, or the
   * run ended before it got that far.
   */
  verification?: 'ready' | 'needsReview' | null;
}

/** A run that produced an assistant message in a conversation. */
export interface ChatRunMeta {
  runId: string;
  messageId: string;
  startedAt: string;
  finishedAt?: string | null;
  modelName?: string | null;
  live: boolean;
}

/**
 * One chat thread. The shape mirrors the Rust `Conversation` and the
 * camelCase fields the back-end emits.
 */
export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  lastActivityAt: string;
  messages: ChatMessage[];
  runs: ChatRunMeta[];
  compactions: number;
}
