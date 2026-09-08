/**
 * Keeping a long run inside the model's context window.
 *
 * ## The failure this prevents
 *
 * A refinery task — read a scanned report, search the manuals, compute
 * something, draft a note — is twenty or more turns, and every tool result adds
 * to the transcript. A local model's window is 8k to 32k tokens, not 200k. Left
 * alone the run does not slow down or degrade; it **stops**, because the
 * inference server refuses a prompt at or over its window. That is a demo
 * failing in front of an audience with an error about token counts.
 *
 * ARJUN's Rust engine has that refusal at `ai_engine/runtime.rs`, and
 * `ContextManager::trim_context` next to it was never implemented. This is what
 * fills that hole, using OpenClaw's compaction rather than a fresh attempt —
 * the parts that are hard to get right are exactly the parts already solved
 * there.
 *
 * ## What is reused, and why each part matters
 *
 * - `estimateTokens` / `estimateContextTokens` — counts an image block as a flat
 *   {@link IMAGE_BLOCK_TOKENS}. ARJUN feeds rendered PDF pages to vision models,
 *   and a character-count heuristic would read a page image as ~0 tokens and
 *   compact far too late.
 * - `findCutPoint` — chooses where to cut so an assistant tool call is never
 *   separated from its tool result. Cutting between them produces a transcript
 *   the provider rejects as malformed, which looks like a bug in the loop.
 * - `generateSummary` — carries the previous summary forward, so compacting
 *   twice refines one summary instead of summarising a summary.
 * - `capCompactionSummary` — bounds the summary, so it cannot itself grow into
 *   the thing that overflows the window.
 *
 * ## Where it runs
 *
 * `Agent.transformContext`, which rewrites the context before each provider
 * request and leaves the stored transcript intact. That split is what we want:
 * the model sees a summary, while the audit record keeps every message.
 */

import {
  createCompactionSummaryMessage,
  DEFAULT_COMPACTION_SETTINGS,
  estimateContextTokens,
  estimateTokens,
  findCutPoint,
  generateSummary,
  shouldCompact,
  type AgentMessage,
  type CompactionSettings,
} from "@openclaw/agent-core";
// Not on the package barrel upstream. Imported through the subpath the vendored
// tsconfig already maps rather than by patching the barrel, which would add a
// conflict to every future re-sync for one symbol.
import { capCompactionSummary } from "@openclaw/agent-core/harness/compaction";
import type { Model } from "@openclaw/ai";
import type { AgentCoreCompletionRuntimeDeps } from "@openclaw/agent-core";
import { ContextLedger, type ContextLedgerSnapshot } from "./context-ledger.js";
import { WorkingNotes } from "./working-notes.js";

/**
 * State that must survive compaction verbatim rather than through a summary.
 *
 * A summariser is a model, and a model asked to compress twenty turns will
 * sometimes drop the sentence that said an approval was granted. That is not a
 * quality problem to be tuned away — it is the difference between a run that
 * stops for a person and a run that proceeds believing it already did. So the
 * things whose loss changes what the run is *allowed* to do are carried across
 * as text this code writes, not as text a model writes.
 *
 * Supplied per compaction rather than held, because all of it lives on the Rust
 * side and can change between turns.
 */
export interface PreservedState {
  /** The plan the run is being held to, and where it has got to. */
  activePlan?: string;
  /**
   * Approvals granted or refused, and any policy refusal already issued.
   *
   * The load-bearing one. A granted approval that is summarised away is
   * re-requested, which is merely annoying; a *refusal* summarised away is
   * retried, which is the failure that matters.
   */
  policyDecisions?: string[];
  /** Evidence markers the run holds, as references — never the passages. */
  evidenceRefs?: string[];
  /** Questions the run has not settled. */
  unresolvedIssues?: string[];
  /** Files the run has recently read or produced, by name. */
  recentFiles?: string[];
  /**
   * Context entries a person has asked to keep, by ledger entity id.
   *
   * ## Why this is here rather than only in the surface
   *
   * The context meter draws a pin beside every evictable row, labelled "Keep
   * this when the window fills". Pressing it set a boolean in the component's
   * own React state and stopped there — nothing sent it anywhere, and this side
   * had never heard of it. So the row went dark, the "what goes first" line
   * moved on to the next row, and the compactor cleared the pinned document on
   * its next pass exactly as if nothing had been pressed.
   *
   * A control that appears to protect something and does not is worse than no
   * control at all: it converts a decision the person could still have made —
   * attach less, ask differently, start a new thread — into one they believe
   * they already made.
   *
   * Carried in {@link PreservedState} rather than on a channel of its own
   * because it is exactly what this structure is for: state the Rust side owns,
   * that changes between turns, and whose loss changes what the run may do. It
   * is pushed by `run.note`, from `agent_pin_context`.
   *
   * The ids are the ledger's own entity ids, which are what the meter shows: a
   * document's content hash, an evidence marker. So what a person pinned and
   * what this protects are the same names, not two lists somebody has to
   * reconcile.
   */
  pinned?: string[];
}

/** What compaction did, for the event stream and the run record. */
export interface CompactionEvent {
  tokensBefore: number;
  tokensAfter: number;
  /** Transcript messages now represented by the summary rather than sent whole. */
  messagesSummarised: number;
  /** Which compaction of this run this was, 1-based. */
  ordinal: number;
  /**
   * True when this compaction extended the summary already held rather than
   * writing a new one. Recorded because a second compaction that *replaced* the
   * summary would silently lose the first half of the run, and a counter that
   * cannot tell the two apart cannot show that it did not happen.
   */
  refinedExistingSummary: boolean;
  /** Raw tool results replaced by an evidence reference on this pass. */
  toolResultsCleared: number;
  /** The ledger as it stood after the compaction. */
  ledger: ContextLedgerSnapshot;
  /**
   * When this happened, ISO 8601.
   *
   * Stamped here rather than by whoever receives the event, because this is the
   * side that knows. A surface folding compactions in live had no timestamp to
   * put on the row and would have had to invent one from its own clock — which
   * is a different clock, read at a later moment, presented as the moment the
   * history was lost.
   */
  at: string;
}

export interface CompactorOptions {
  model: Model;
  runtime: AgentCoreCompletionRuntimeDeps;
  /** Placeholder credential; a loopback server wants none but the client demands one. */
  apiKey: string;
  /** Called when a compaction happens, so an operator can be told. */
  onCompacted?: (event: CompactionEvent) => void;
  settings?: Partial<CompactionSettings>;
  /**
   * The run's bounded notes.
   *
   * Rendered into the context ahead of the transcript on every turn, not only
   * after a compaction: notes that appear only once the window is full are
   * notes the model was never shown while it was deciding what to record.
   */
  notes?: WorkingNotes;
  /** Where the section counts are accumulated. One per run. */
  ledger?: ContextLedger;
  /** Read at each compaction. See {@link PreservedState}. */
  preserved?: () => PreservedState;
}

/**
 * Settings for a local model, which has far less room than a cloud one.
 *
 * Upstream reserves 16k tokens and keeps 20k of recent context — sensible
 * against a 200k window, and larger than the entire window of a model ARJUN
 * routinely runs. Both are therefore derived from the window rather than fixed:
 * a fifth reserved for the summary request and its output, two fifths kept as
 * recent context. On a 200k window this lands near the upstream numbers; on an
 * 8k one it stays proportionate instead of demanding more than exists.
 */
export function settingsForWindow(contextWindow: number): CompactionSettings {
  if (!Number.isFinite(contextWindow) || contextWindow <= 0) {
    return { ...DEFAULT_COMPACTION_SETTINGS, enabled: false };
  }
  return {
    enabled: true,
    reserveTokens: Math.max(512, Math.floor(contextWindow * 0.2)),
    keepRecentTokens: Math.max(512, Math.floor(contextWindow * 0.4)),
  };
}

/**
 * Wraps messages as the session entries `findCutPoint` expects.
 *
 * The ids are positional and exist only for the length of one call. ARJUN does
 * not adopt OpenClaw's session tree as a persistence format — the audit ledger
 * is the record — but the cut-point selection is written against that shape and
 * is the part worth reusing, so the shape is supplied.
 */
function asEntries(messages: AgentMessage[]) {
  return messages.map((message, index) => ({
    type: "message" as const,
    id: `m${index}`,
    parentId: index === 0 ? null : `m${index - 1}`,
    timestamp: new Date(message.timestamp ?? 0).toISOString(),
    message,
  }));
}

/**
 * A message timestamp as epoch milliseconds.
 *
 * The harness types allow either a number or an RFC 3339 string depending on
 * where a message came from. Normalised here rather than at each use, so a
 * string timestamp produces a correct instant instead of `NaN`.
 */
function asEpoch(timestamp: string | number | undefined): number | undefined {
  if (typeof timestamp === "number") return timestamp;
  if (typeof timestamp !== "string") return undefined;
  const parsed = Date.parse(timestamp);
  return Number.isNaN(parsed) ? undefined : parsed;
}

/** A message that carries assistant tool calls, viewed structurally. */
interface ToolCallish {
  role?: string;
  toolCallId?: string;
  content?: unknown;
}

/** The ids of the tool calls an assistant message issued. */
function toolCallIdsIn(message: AgentMessage): string[] {
  const shape = message as ToolCallish;
  if (shape.role !== "assistant" || !Array.isArray(shape.content)) return [];
  return shape.content
    .filter(
      (block): block is { type: string; id?: string; toolCallId?: string } =>
        typeof block === "object" && block !== null && (block as { type?: string }).type === "toolCall",
    )
    .map((block) => block.id ?? block.toolCallId)
    .filter((id): id is string => typeof id === "string");
}

/** The call id a tool-result message answers, if it is one. */
function toolResultIdOf(message: AgentMessage): string | undefined {
  const shape = message as ToolCallish;
  return shape.role === "toolResult" ? shape.toolCallId : undefined;
}

/** A message's text blocks, concatenated. Empty for a message with none. */
function textOf(message: AgentMessage): string {
  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) return "";
  return content
    .map((block) =>
      typeof block === "object" && block !== null && typeof (block as { text?: unknown }).text === "string"
        ? (block as { text: string }).text
        : "",
    )
    .join("");
}

/**
 * Whether this message is retrieved evidence rather than conversation.
 *
 * A tool result carrying at least one `[E<n>]` marker: that is what the
 * retrieval side stamps on every passage it returns, and on the reference stub
 * left behind once the passage text is cleared. The stub form is why the
 * pattern allows a list — a result that carried two passages is cleared to
 * `[E1, E2]`, and insisting on a bracket straight after the digits would stop
 * recognising exactly the results that retrieved the most.
 *
 * The distinction is the whole point of booking evidence separately. An
 * operator told "transcript: 6,000 tokens" reaches for the only lever that
 * phrasing offers — compact sooner — which shortens the conversation and
 * degrades the run. Told "evidence: 5,200 tokens" they reach for the lever that
 * costs nothing: ask for three pages instead of thirty, because the rest is
 * still retrievable by marker. Same window, same overflow, opposite remedy.
 */
export function isEvidenceMessage(message: AgentMessage): boolean {
  if ((message as ToolCallish).role !== "toolResult") return false;
  return /\[E\d+(?:,\s*E\d+)*\]/.test(textOf(message));
}

/** The sentence a truncated block ends on, so the model knows it was cut. */
const TRUNCATION_NOTE =
  "\n\n[… cut here to fit this model's context window. Ask for this material again in " +
  "smaller pieces — by page, by section, or by search — rather than treating what is above " +
  "as the whole of it.]";

/**
 * Shortens a message's text blocks until it costs about `roomTokens`.
 *
 * Iterative rather than arithmetic: the character-to-token ratio is an average
 * and this is the one place where being wrong about it means the request is
 * still refused, so the cut is measured with the same estimator the ceiling is
 * compared against and halved again if it did not land. Six passes take any
 * message down by a factor of sixty, which is more than the gap has ever been.
 *
 * Non-text blocks are left alone. An image block is a flat cost that cannot be
 * made smaller, and dropping it would change what the message *says* rather
 * than how much of it is carried.
 */
function truncateMessageText(message: AgentMessage, roomTokens: number): AgentMessage {
  const content = (message as { content?: unknown }).content;
  if (!Array.isArray(content)) return message;

  const textAt = (blocks: unknown[]) =>
    blocks.filter(
      (block): block is { type: "text"; text: string } =>
        typeof block === "object" &&
        block !== null &&
        (block as { type?: string }).type === "text" &&
        typeof (block as { text?: unknown }).text === "string",
    );
  if (textAt(content).length === 0) return message;

  let share = 1;
  for (let pass = 0; pass < 6; pass += 1) {
    const total = textAt(content).reduce((sum, block) => sum + block.text.length, 0);
    const keep = Math.max(120, Math.floor((total * share) / 2));
    const cut = content.map((block) => {
      const [text] = textAt([block]);
      if (!text || text.text.length <= keep) return block;
      return { ...text, text: text.text.slice(0, keep) + TRUNCATION_NOTE };
    });
    const candidate = { ...(message as object), content: cut } as AgentMessage;
    if (estimateTokens(candidate) <= roomTokens) return candidate;
    share /= 2;
    if (pass === 5) return candidate;
  }
  return message;
}

/** One line the model can read, saying what the window could not hold. */
function ceilingMarker(dropped: number, truncated: number): string {
  const parts: string[] = [];
  if (dropped > 0) {
    parts.push(
      `${dropped} earlier message${dropped === 1 ? "" : "s"} in this task ${
        dropped === 1 ? "was" : "were"
      } removed`,
    );
  }
  if (truncated > 0) {
    parts.push(`${truncated} message${truncated === 1 ? " was" : "s were"} shortened`);
  }
  return (
    `[Context notice: ${parts.join(" and ")} because this model's context window could not ` +
    "hold them. Do not assume what is missing agreed with you, and do not state anything you " +
    "can no longer see. If the answer depends on it, search or read it again — or say which " +
    "part you can no longer account for.]"
  );
}

/**
 * Whether every tool result in this window has the call that produced it.
 *
 * The property a provider enforces and rejects the whole request over. Exposed
 * rather than kept private because it is the thing worth asserting in a test:
 * a cut that orphans a tool result does not degrade the run, it ends it with a
 * malformed-request error that reads like a bug in the agent loop.
 */
export function pairingIsIntact(messages: AgentMessage[]): boolean {
  const issued = new Set<string>();
  for (const message of messages) {
    for (const id of toolCallIdsIn(message)) issued.add(id);
    const answered = toolResultIdOf(message);
    if (answered !== undefined && !issued.has(answered)) return false;
  }
  return true;
}

/**
 * Moves a cut earlier until it no longer orphans a tool result.
 *
 * `findCutPoint` already chooses turn boundaries and is the primary defence.
 * This is the second one, and it exists because the two disagree in exactly one
 * case: the cut is computed over *session entries*, and ARJUN synthesises those
 * entries positionally from a message list that agent-core may have rewritten —
 * an interrupt message, a repaired tool call. Re-deriving the property directly
 * from the messages costs one pass and removes the need to reason about whether
 * those two representations can drift.
 *
 * Returns an index at or before `cut`, never after: this may keep more history
 * than asked, and must never keep less.
 */
export function alignCutToPairs(messages: AgentMessage[], cut: number): number {
  let aligned = Math.max(0, Math.min(cut, messages.length));
  // Walk back while the first kept message is a tool result whose call is not
  // also kept. Each step swallows one more message, so this terminates at 0.
  for (;;) {
    const kept = messages.slice(aligned);
    if (pairingIsIntact(kept) || aligned === 0) return aligned;
    aligned -= 1;
  }
}

/** How many trailing messages are never pruned, however stale they look. */
const PRUNE_KEEPS_RECENT = 6;

/**
 * Replaces raw tool-result bodies with a reference once the evidence is durable.
 *
 * ## Why this is safe, and only here
 *
 * A search result is the largest thing in a document run's context and the most
 * redundant: the passage text is already in the Rust evidence table under the
 * marker the model was told to cite, and it stays there for the life of the
 * run. So once `[E3]` is recorded in the notes, the *text* of the result that
 * produced it is a second copy of something retrievable, and dropping it costs
 * the model nothing it cannot ask for again.
 *
 * Two conditions, both required, because getting either wrong loses real work:
 *
 * - **The marker must already be in the notes.** The notes are what is
 *   persisted, so a marker present there is one a recovered run can still
 *   resolve. Pruning against markers seen only in the live transcript would
 *   discard text whose reference dies with the process.
 * - **The most recent {@link PRUNE_KEEPS_RECENT} messages are untouched.** The
 *   model is usually still working with what it just read, and a result pruned
 *   in the same breath it was returned reads to the model as a tool that
 *   silently failed.
 *
 * The message is rewritten, never removed: removing it would orphan the tool
 * call that produced it, which is the failure the rest of this file exists to
 * prevent.
 */
export function pruneStaleToolResults(
  messages: AgentMessage[],
  durableMarkers: readonly string[],
  /**
   * Entries a person asked to keep. Never cleared, however durable they are.
   *
   * A third condition on top of the two above, and the only one that comes from
   * outside this process. The other two are about whether clearing is *safe*;
   * this is about whether it is *wanted*. Somebody watching the meter fill and
   * pinning the drawing they are working from has said which of these results
   * they still need, and that answer outranks a rule about what is retrievable
   * — retrievable costs a tool call and a turn, and the person pinned it
   * because they did not want to spend those.
   *
   * Matched against both the evidence markers a result carries (`E3`) and the
   * document ids in its text, because the meter shows rows of both kinds and a
   * pin has to mean the same thing whichever row it was pressed on.
   */
  pinned: readonly string[] = [],
): { messages: AgentMessage[]; cleared: number } {
  if (durableMarkers.length === 0) return { messages, cleared: 0 };
  const markers = new Set(durableMarkers.map((marker) => marker.toUpperCase()));
  const kept = new Set(pinned.map((id) => id.toUpperCase()).filter((id) => id.length > 0));
  const cutoff = messages.length - PRUNE_KEEPS_RECENT;
  let cleared = 0;

  const rewritten = messages.map((message, index) => {
    if (index >= cutoff) return message;
    const shape = message as ToolCallish & { content?: unknown };
    if (shape.role !== "toolResult" || !Array.isArray(shape.content)) return message;

    const text = textOf(message);
    if (!text) return message;

    // Pinned, and therefore not this pass's to reclaim. Checked before the
    // durability rules below, because a person's answer to "do you still need
    // this?" is not improved by this code's opinion about whether it could be
    // fetched again.
    if (kept.size > 0) {
      const upper = text.toUpperCase();
      if ([...kept].some((id) => upper.includes(id))) return message;
    }

    // Every marker this result carried, and only markers that are durable.
    const found = [...text.matchAll(/\[E(\d+)\]/g)].map((match) => `E${match[1]}`);
    const durable = [...new Set(found)].filter((marker) => markers.has(marker));
    if (durable.length === 0 || durable.length !== new Set(found).size) {
      // Nothing durable here, or the result carried a marker that is not yet
      // recorded. Pruning a partially-durable result would drop the half that
      // cannot be looked up again, so it is left whole.
      return message;
    }

    cleared += 1;
    return {
      ...(message as object),
      content: [
        {
          type: "text",
          text: `[${durable.join(", ")}] Passage text cleared from context. These passages are held as this run's evidence and can be cited by marker; use load_more_evidence to read a specific page again.`,
        },
      ],
    } as AgentMessage;
  });

  return { messages: rewritten, cleared };
}

/**
 * The state carried across a compaction as text, not as a summary.
 *
 * Written as a user message rather than a system one so it cannot be reordered
 * away from the summary it belongs beside, and so a model that follows the last
 * instruction it saw sees this after the summary rather than before it.
 */
function preservedMessage(state: PreservedState, notes: WorkingNotes, timestamp: number): AgentMessage | undefined {
  const lines: string[] = [];
  if (state.activePlan) lines.push(`Active plan: ${state.activePlan}`);
  if (state.policyDecisions?.length) {
    lines.push("Policy and approval decisions still in force:");
    for (const decision of state.policyDecisions) lines.push(`  - ${decision}`);
  }
  if (state.evidenceRefs?.length) {
    lines.push(`Evidence available by marker: ${state.evidenceRefs.join(", ")}`);
  }
  if (state.unresolvedIssues?.length) {
    lines.push("Still unresolved:");
    for (const issue of state.unresolvedIssues) lines.push(`  - ${issue}`);
  }
  if (state.recentFiles?.length) {
    lines.push(`Files in play: ${state.recentFiles.join(", ")}`);
  }
  if (state.pinned?.length) {
    // Named across the compaction as well as protected by it. The person who
    // pinned these said they are what the rest of the task depends on, and a
    // model that keeps the text without being told why has lost the reason it
    // was kept.
    lines.push(
      `Kept at the operator's request, and still current: ${state.pinned.join(", ")}`,
    );
  }

  const rendered = notes.render();
  if (rendered) lines.push(rendered);
  if (lines.length === 0) return undefined;

  return {
    role: "user",
    content: [
      {
        type: "text",
        text: `The earlier history was replaced by the summary above. These facts were carried across unchanged and are current:\n${lines.join("\n")}`,
      },
    ],
    timestamp,
  } as AgentMessage;
}

/**
 * Compacts one run's context as it grows.
 *
 * Stateful across turns: it remembers the summary produced so far and how much
 * of the transcript that summary already covers, so each compaction extends the
 * previous one rather than starting again.
 */
export class RunCompactor {
  readonly #options: CompactorOptions;
  readonly #settings: CompactionSettings;
  readonly #notes: WorkingNotes;
  readonly #ledger: ContextLedger;
  #summary?: string;
  /** Messages the summary stands in for: `messages[0..covered)`. */
  #covered = 0;
  #compactions = 0;
  /**
   * Raw tool results replaced by a reference in the current projection.
   *
   * Assigned, not accumulated. Pruning recomputes over the whole transcript
   * every turn, so adding each turn's count to the last would report a run that
   * cleared three results as having cleared thirty by turn ten — a number that
   * grows with turns rather than with anything that happened.
   */
  #cleared = 0;

  constructor(options: CompactorOptions) {
    this.#options = options;
    this.#settings = {
      ...settingsForWindow(options.model.contextTokens ?? options.model.contextWindow ?? 0),
      ...options.settings,
    };
    this.#notes = options.notes ?? new WorkingNotes();
    this.#ledger =
      options.ledger ??
      new ContextLedger(options.model.contextTokens ?? options.model.contextWindow ?? 0);
    this.#ledger.set("reserve", this.#settings.reserveTokens);
  }

  get compactions(): number {
    return this.#compactions;
  }

  /** The run's notes, so a caller can record into the same instance. */
  get notes(): WorkingNotes {
    return this.#notes;
  }

  /** The ledger, for a caller that wants to show or persist it. */
  get ledger(): ContextLedger {
    return this.#ledger;
  }

  /**
   * What the model is shown, given the transcript and any summary so far.
   *
   * `preamble` counts the leading messages this class wrote rather than the
   * conversation contributing: the summary standing in for older history, the
   * carried state, the working notes. {@link RunCompactor} needs to know where
   * they stop, because they are the messages the ceiling pass must not drop —
   * the summary *is* the earlier history, and evicting it to make room would
   * throw away the very thing compaction produced to save space.
   */
  #project(messages: AgentMessage[]): { messages: AgentMessage[]; preamble: number } {
    const notes = this.#notes.render();

    if (!this.#summary || this.#covered === 0) {
      // Before any compaction the notes still go in, ahead of the transcript.
      // A model asked to maintain notes it has never been shown maintains
      // nothing, and the first thing it would have recorded is the goal — which
      // is exactly what the first compaction is most likely to lose.
      if (!notes) return { messages, preamble: 0 };
      return {
        messages: [this.#notesMessage(notes, asEpoch(messages[0]?.timestamp)), ...messages],
        preamble: 1,
      };
    }

    const summary = createCompactionSummaryMessage(
      this.#summary,
      this.#tokensAt(messages.slice(0, this.#covered)),
      new Date(messages[0]?.timestamp ?? Date.now()).toISOString(),
    ) as unknown as AgentMessage;

    const timestamp = asEpoch(messages[this.#covered]?.timestamp) ?? Date.now();
    const carried = preservedMessage(
      this.#options.preserved?.() ?? {},
      this.#notes,
      timestamp,
    );

    // The cut is re-aligned here and not only where it was chosen, because the
    // kept tail is what is actually sent. See `alignCutToPairs`.
    const tail = messages.slice(alignCutToPairs(messages, this.#covered));
    return carried
      ? { messages: [summary, carried, ...tail], preamble: 2 }
      : { messages: [summary, ...tail], preamble: 1 };
  }

  #notesMessage(rendered: string, timestamp?: number): AgentMessage {
    return {
      role: "user",
      content: [{ type: "text", text: rendered }],
      timestamp: timestamp ?? Date.now(),
    } as AgentMessage;
  }

  #tokensAt(messages: AgentMessage[]): number {
    return estimateContextTokens(messages).tokens;
  }

  /**
   * The `transformContext` hook.
   *
   * Measures the *projected* context, not the raw transcript: once a summary
   * exists, the raw transcript stays over the limit forever and measuring it
   * would compact on every single turn.
   */
  async transform(messages: AgentMessage[], signal?: AbortSignal): Promise<AgentMessage[]> {
    const window = this.#options.model.contextTokens ?? this.#options.model.contextWindow ?? 0;

    // Cheapest saving first, and it happens whether or not this turn compacts:
    // a passage whose marker is already durable is a second copy of something
    // retrievable, and clearing it may be enough that no summary is needed at
    // all. Doing it only at compaction time would mean the run summarises
    // history it did not have to lose.
    // Read at the moment of the turn rather than captured, so a pin pressed
    // thirty seconds ago is honoured on the very next projection instead of
    // whenever the compactor happens to be rebuilt.
    const preserved = this.#options.preserved?.() ?? {};
    const pinned = preserved.pinned ?? [];
    // The meter is redrawn from this snapshot, so a pin that landed here shows
    // as held in the panel the person pressed it in. Before this the ledger had
    // a `setPinned` that nothing ever called, and the panel's pin state lived
    // only in the component that drew it.
    //
    // Released as well as set, and that direction is not optional. `pinned` is
    // the whole set every time, so an id that has left it has been *unpinned* —
    // and a loop that only ever set `true` would leave the row drawn as
    // protected for the rest of the run while `pruneStaleToolResults`, reading
    // the same list, correctly stopped protecting it. That is the same lie this
    // work exists to remove, pointing the other way.
    this.#ledger.applyPins(pinned);

    const pruned = pruneStaleToolResults(messages, this.#notes.state.evidenceIds, pinned);
    const working = pruned.messages;
    this.#cleared = pruned.cleared;

    let { messages: projected, preamble } = this.#project(working);
    // What the *request* costs, not what the messages cost.
    //
    // The system prompt and the tool schemas go out on every call and are
    // measured once into the ledger; adding them here is what makes this
    // decision about the thing the server actually refuses. Asked without
    // them, `shouldCompact` reported that 2,700 tokens of conversation fitted
    // an 8,192-token window while the request around it came to 9,238 — so
    // nothing was compacted, and the turn died at the provider with the one
    // error the compactor exists to prevent.
    const fixed = this.#ledger.fixed();
    const tokensBefore = this.#tokensAt(projected) + fixed;
    this.#measure(projected);

    if (!shouldCompact(tokensBefore, window, this.#settings)) {
      return this.#enforceCeiling(projected, preamble, window);
    }

    const entries = asEntries(working);
    const { firstKeptEntryIndex } = findCutPoint(
      entries,
      this.#covered,
      entries.length,
      this.#settings.keepRecentTokens,
    );

    // Nothing new to fold in — everything older is already summarised. There is
    // no second summary to write, so the ceiling pass takes it from here: it
    // drops whole messages rather than summarising them, which is worse for the
    // model and still incomparably better than a refused request.
    if (firstKeptEntryIndex <= this.#covered) {
      return this.#enforceCeiling(projected, preamble, window);
    }

    const toSummarise = working.slice(this.#covered, firstKeptEntryIndex);
    // Recorded before the summariser is asked, because the answer to "did this
    // extend the existing summary or replace it?" is decided by whether one was
    // held going in, and `#summary` is overwritten below.
    const refinedExistingSummary = this.#summary !== undefined;

    // Two failure shapes, both of which must leave the run alive: a returned
    // error result, and a throw. `generateSummary` propagates whatever the
    // completion function raises, so the transport being down surfaces here as
    // an exception rather than an `err`. Catching only one of the two would
    // mean a model server that dies mid-run takes the task with it — a failure
    // an operator experiences as ARJUN crashing, not as summarisation failing.
    let summary: string | undefined;
    try {
      const result = await generateSummary(
        toSummarise,
        this.#options.model,
        this.#settings.reserveTokens,
        this.#options.apiKey,
        undefined,
        signal,
        undefined,
        this.#summary,
        undefined,
        undefined,
        this.#options.runtime,
      );
      summary = result.ok ? result.value : undefined;
    } catch {
      summary = undefined;
    }

    if (summary === undefined) {
      // Summarisation is what failed, not the run. The context still has to fit
      // — a model server that could not write a summary is in no better
      // position to accept an over-long request — so the ceiling pass drops
      // messages instead. Mechanical rather than clever, and it needs nothing
      // from the model that just declined to answer.
      return this.#enforceCeiling(projected, preamble, window);
    }

    this.#summary = capCompactionSummary(summary);
    // Aligned before it is stored, so the covered boundary and the boundary the
    // projection actually cuts at can never be two different numbers.
    this.#covered = alignCutToPairs(working, firstKeptEntryIndex);
    this.#compactions += 1;
    this.#ledger.countCompaction();

    ({ messages: projected, preamble } = this.#project(working));
    // The ceiling pass runs after a successful compaction too. A summary is a
    // large saving and not an unbounded one: a single tool result carrying
    // forty pages can still be in the kept tail, and the request either fits or
    // it does not — a compaction that halved it is not an answer to that
    // question, only a better starting point for it.
    projected = this.#enforceCeiling(projected, preamble, window);
    this.#measure(projected);

    this.#options.onCompacted?.({
      tokensBefore,
      tokensAfter: this.#tokensAt(projected) + fixed,
      messagesSummarised: this.#covered,
      ordinal: this.#compactions,
      refinedExistingSummary,
      toolResultsCleared: this.#cleared,
      ledger: this.#ledger.snapshot(),
      at: new Date().toISOString(),
    });
    return projected;
  }

  /**
   * The last thing between a projection and the wire: makes it fit, whatever
   * it takes.
   *
   * ## Why a mechanical pass exists at all
   *
   * Everything above it is a *good* way to shrink a context — clear a tool
   * result that is retrievable by marker, summarise the oldest half, carry the
   * decisions across verbatim. Every one of them can decline. Summarisation
   * needs the model server, which may be the thing that is failing. Cut-point
   * selection returns nothing when the whole transcript is already summarised.
   * Pruning finds nothing when there is nothing stale. On each of those paths
   * the projection went to the provider exactly as it was, and the run ended
   * with `400 request (…) exceeds the available context size`.
   *
   * A person watching that cannot tell it from the model failing, and there is
   * nothing they can do about it. So this pass is deliberately stupid and
   * cannot decline: it drops whole messages, oldest first, until the projection
   * fits, and if one message is on its own larger than the window it truncates
   * that message's text and says so in the text.
   *
   * ## What it will not do
   *
   * - **The preamble stays.** The summary is the earlier history; dropping it
   *   to make room would discard what compaction was run to produce.
   * - **The last user message stays.** That is the question. A request that
   *   fits because the question was removed is a request that will be answered
   *   confidently and about nothing.
   * - **Pairing stays intact.** A tool result without the call that produced it
   *   is a malformed request — a *different* provider refusal, reached by
   *   trying to avoid this one.
   * - **Nothing is silent.** Every drop and every truncation leaves a marker in
   *   the context the model can read, and a line on stderr the operator can.
   *
   * ## Why the estimate is inflated before it is compared
   *
   * See {@link ContextLedger.driftFactor}. The estimator counts characters ÷ 4,
   * and the material this product carries tokenises denser than that. Fitting
   * to the optimistic count is how a request calculated to fit exactly is
   * refused for being 6% over.
   */
  #enforceCeiling(
    projected: AgentMessage[],
    preamble: number,
    window: number,
  ): AgentMessage[] {
    if (!Number.isFinite(window) || window <= 0) return projected;

    const fixed = this.#ledger.fixed();
    // What the messages may occupy: the window, less the reply the model has to
    // have room to write, less everything that is not a message. Never below a
    // token, so the arithmetic below always has somewhere to aim.
    const ceiling = Math.max(1, window - this.#settings.reserveTokens - fixed);
    const drift = this.#ledger.driftFactor();
    const cost = (messages: AgentMessage[]) => Math.ceil(this.#tokensAt(messages) * drift);

    if (cost(projected) <= ceiling) return projected;

    // The notice this pass adds is itself part of the request, so it is charged
    // before anything is dropped rather than spliced in afterwards. Fitting to
    // the ceiling and *then* adding a hundred tokens of explanation is how a
    // pass whose entire job is to make the request fit sends one that does not.
    const noticeCost = Math.ceil(
      estimateTokens({
        role: "user",
        content: [{ type: "text", text: ceilingMarker(1, 1) }],
        timestamp: 0,
      } as AgentMessage) * drift,
    );
    const target = Math.max(1, ceiling - noticeCost);

    const kept = [...projected];
    // The question, or the newest user turn standing in for it. Found by index
    // so the identity survives the splicing below.
    let protectedFrom = kept.length;
    for (let index = kept.length - 1; index >= preamble; index -= 1) {
      if ((kept[index] as { role?: string }).role === "user") {
        protectedFrom = index;
        break;
      }
    }

    let dropped = 0;
    // Oldest first, from just after the preamble, and never into the protected
    // tail. `alignCutToPairs` moves the cut forward off a tool result whose
    // call would be left behind.
    while (cost(kept) > target && preamble < protectedFrom) {
      const cut = alignCutToPairs(kept, preamble + 1);
      if (cut <= preamble || cut > protectedFrom) break;
      const removed = cut - preamble;
      kept.splice(preamble, removed);
      protectedFrom -= removed;
      dropped += removed;
    }

    let truncated = 0;
    if (cost(kept) > target) {
      // What is left is the preamble and the question, and it still does not
      // fit — a single message larger than the window. Almost always a document
      // pasted into the prompt, occasionally a tool result that came back
      // enormous. Truncating it keeps the shape of the request correct and
      // tells the model what happened, which is a thing it can work with;
      // sending it unchanged is a thing nobody can work with.
      const room = Math.max(1, Math.floor(target / Math.max(1, kept.length)));
      for (let index = 0; index < kept.length; index += 1) {
        const before = Math.ceil(estimateTokens(kept[index]!) * drift);
        if (before <= room) continue;
        kept[index] = truncateMessageText(kept[index]!, room);
        truncated += 1;
      }
    }

    if (dropped > 0 || truncated > 0) {
      // A drop nobody can see is indistinguishable from a model that forgot.
      const marker = ceilingMarker(dropped, truncated);
      kept.splice(preamble, 0, {
        role: "user",
        content: [{ type: "text", text: marker }],
        timestamp: asEpoch(kept[preamble]?.timestamp) ?? Date.now(),
      } as AgentMessage);
      process.stderr.write(
        `[agent-runtime:log] [context] ceiling enforced: ${dropped} message(s) dropped, ` +
          `${truncated} truncated, to fit ${ceiling} token(s) of a ${window}-token window ` +
          `(fixed cost ${fixed}, drift x${drift.toFixed(2)})\n`,
      );
    }
    return kept;
  }

  /**
   * Books the projected context into the ledger.
   *
   * Only the sections this side can see: the summary, the notes, the retrieved
   * evidence and the conversation around it. `system`, `skill` and
   * `toolSchema` are set once by the caller that owns them, and are deliberately
   * not recomputed here — this must not silently zero a section it has no view
   * of.
   *
   * Recomputed from scratch on every call rather than adjusted, so a section
   * cannot drift away from the projection it claims to describe over a long run.
   */
  #measure(projected: AgentMessage[]): void {
    const summaryTokens = this.#summary ? estimateContextTokens([projected[0]!]).tokens : 0;
    const notesText = this.#notes.render();
    this.#ledger.set("compaction", summaryTokens);
    this.#ledger.setText("notes", notesText);

    const transcript = this.#summary ? projected.slice(1) : projected;
    // The notes are already booked under `notes`; counting the message that
    // carries them again here would report them twice and overstate the total
    // the next turn has to fit inside.
    const withoutNotes = transcript.filter(
      (message) => !notesText || !textOf(message).includes("## Working notes"),
    );

    // Split rather than summed into one line. See `isEvidenceMessage`: the two
    // halves have different remedies, and a single number names neither.
    this.#ledger.setMessages("evidence", withoutNotes.filter(isEvidenceMessage));
    this.#ledger.setMessages(
      "transcript",
      withoutNotes.filter((message) => !isEvidenceMessage(message)),
    );
  }
}
