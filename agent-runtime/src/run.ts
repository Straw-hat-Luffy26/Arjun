/**
 * One agent run: model turns, tool calls, and the events a person watches.
 *
 * This is the thin layer between OpenClaw's `Agent` and ARJUN's Rust core. It
 * owns three things and deliberately nothing else:
 *
 * - building the model record from what Rust chose (the *routing* decision is
 *   Rust's, in `registry::router` -- this side never picks a model);
 * - installing the authorisation hook so every tool call is put to the gateway;
 * - forwarding lifecycle events so the UI can show work as it happens.
 *
 * Everything that decides anything -- which model, whether a tool may run, what
 * a tool does -- lives on the other side of the wire.
 */

import { Agent, convertToLlm, type AgentEvent, type AgentMessage } from "@openclaw/agent-core";
import { createLlmRuntime, type Model } from "@openclaw/ai";
import { registerBuiltInApiProviders } from "@openclaw/ai/providers";
import type { RpcPeer } from "./peer.js";
import { RunCompactor, settingsForWindow, type PreservedState } from "./compaction.js";
import { estimateTextTokens, fitToolsToBudget } from "./tool-budget.js";
import { ContextLedger } from "./context-ledger.js";
import { WorkingNotes, type WorkingNotesState } from "./working-notes.js";
import { payloadPolicy } from "./providers.js";
import { withToolCallRepair } from "./repair.js";
import { withCallTiming } from "./timing.js";
import { GrantLedger, authorizeToolCall, buildTools, fetchCatalogue } from "./tools.js";
import { observeToolResult } from "./note-taking.js";

/** What Rust sends with `run.start`. */
export interface RunRequest {
  runId: string;
  /**
   * The id of the assistant `Message` row the chat surface reserved for this
   * turn via `agent_append_turn`. Attached to every `message_start`,
   * `message_update`, and `message_end` event so the consumer can route each
   * token to the right cell without filtering by `runId`.
   */
  messageId: string;
  prompt: string;
  systemPrompt: string;
  /**
   * The conversation so far, oldest first. Never this turn's question.
   *
   * ## Why a run is given history at all
   *
   * Every chat message starts a new run, and a run used to be handed exactly
   * one thing: the prompt. So the second turn of a conversation began with a
   * model that had never seen the first, and "what was the rating you just
   * quoted?" was answered by a model with no quote. The history was on screen
   * and in a file on disk; it simply never entered a model request.
   *
   * ## Why it is seeded and not prompted
   *
   * These become `initialState.messages` — the transcript the loop *starts*
   * from — and `request.prompt` is then submitted through `agent.prompt`
   * exactly as it always was. Two things follow, and both are the point:
   *
   * - The new question is appended by exactly one line of code, so it cannot be
   *   asked twice however this is called.
   * - The seeded transcript never ends on the live question, so no path exists
   *   on which the loop reads it as already-asked and returns without
   *   generating. A run that seeded the question *into* the history would have
   *   both failure modes available to it depending on where the seam fell.
   *
   * ## Why this is not `notes`
   *
   * {@link RunRequest.notes} is same-run recovery: what an earlier attempt at
   * *this run* already did, including the side effects that must not happen
   * twice. This is fresh-turn initialisation: what was said in earlier turns of
   * this conversation. They arrive together and are kept apart, because a first
   * turn should not inherit a resumption's machinery and a resumption should not
   * re-read a conversation it never lost.
   *
   * Fitted to this model's window on the Rust side, which is where the model was
   * chosen and where the signed-in owner is known. Absent or empty for the first
   * turn of a conversation.
   */
  history?: { role: "user" | "assistant"; content: string }[];
  /**
   * Earlier messages the window could not hold, dropped oldest-first.
   *
   * Forwarded to the surface so a person can see that a conversation is being
   * shortened. A run that quietly forgets its own history looks, from the
   * outside, exactly like the defect this replaced.
   */
  historyDropped?: number;
  /** The routed model. Chosen by `registry::router` on the Rust side. */
  model: {
    id: string;
    name?: string;
    provider: string;
    /** Endpoint of the local inference server. Must be loopback. */
    baseUrl: string;
    contextWindow?: number;
    maxTokens?: number;
    input?: ("text" | "image")[];
    /** Whether reasoning is wanted for this run. */
    reasoning?: boolean;
    /**
     * Whether this model can be asked for reasoning at all.
     *
     * Read on the Rust side from the model's own chat template — the presence
     * of the `enable_thinking` variable it branches on — rather than matched
     * against a list of model families. A model with no switch must not be
     * sent the kwarg in either direction: telling a model that never reasons
     * not to reason is noise, and telling one that always reasons to stop is a
     * template variable it will ignore while the operator waits for a panel
     * that never fills.
     *
     * Absent means unknown, and the runtime falls back to recognising the two
     * families it knows by name — the behaviour that shipped before this
     * existed.
     */
    supportsReasoning?: boolean;
  };
  /**
   * When this run must stop, as epoch milliseconds.
   *
   * The same instant the Rust side is holding. Sent so the loop can stop
   * *itself* at a point it knows is safe, rather than being killed from outside
   * in the middle of a turn — this side knows where its own safe points are and
   * the other side does not.
   *
   * It is not a second authority. The only thing it can do is end the run
   * earlier than Rust would; every tool call still goes through the gateway,
   * and nothing here decides whether an action is permitted.
   */
  deadlineMs?: number;
  /**
   * Notes carried over from an earlier attempt at this run.
   *
   * Sent when a run is resumed after the process went away. What makes the
   * resumption safe rather than merely faster is `completed`: it names the side
   * effects that already happened, so the model is told not to repeat them
   * instead of rediscovering by doing them twice.
   */
  notes?: Partial<WorkingNotesState>;
  /**
   * State the Rust side owns and this side must carry across compaction
   * unchanged. Refreshed by `run.note`; see {@link PreservedState}.
   */
  preserved?: PreservedState;
}

/**
 * How a run ended.
 *
 * A run has exactly one of these, and it is derived from what the loop actually
 * did rather than from whether this side managed to reply. The distinction is
 * the whole point: a JSON-RPC request that resolves says the *transport* worked,
 * and before this existed the core read that as "the task succeeded" -- so a run
 * an operator stopped, a run that hit the model's output cap mid-sentence, and a
 * run that answered were recorded, listed and shown as the same thing.
 *
 * - `completed` -- the loop finished and the model stopped of its own accord.
 * - `failed` -- the provider or the loop errored. `detail` is the sentence.
 * - `aborted` -- a person, or the core, stopped it.
 * - `lengthLimited` -- the model hit the output cap. The answer is a fragment,
 *   and calling that complete is the failure mode this product cannot have.
 * - `budgetStopped` -- it ran past the time or the steps its plan allowed.
 * - `policyStopped` -- it needed to do something it is not permitted to do.
 */
export type RunOutcomeKind =
  | "completed"
  | "failed"
  | "aborted"
  | "lengthLimited"
  | "budgetStopped"
  | "policyStopped";

/** The typed ending of a run, with the one sentence a person is shown. */
export interface RunTermination {
  kind: RunOutcomeKind;
  /**
   * Why, in a sentence. Absent only for `completed`, which needs no excuse.
   *
   * Bounded and safe to display: it is the loop's own wording for the ending,
   * never a tool result and never model output.
   */
  detail?: string;
}

export interface RunOutcome {
  runId: string;
  /** Assistant text of the final turn, for callers that want just the answer. */
  text: string;
  turns: number;
  /**
   * How this run ended, typed.
   *
   * The core maps this onto the run's terminal event. It does **not** infer the
   * ending from the fact that this request resolved.
   */
  outcome: RunTermination;
  stopReason?: string;
  /**
   * The run's notes as they finished.
   *
   * Returned so Rust can persist them with the task record. A run that ends
   * without handing these back is a run whose next attempt starts from nothing,
   * which is the case this whole mechanism exists to remove.
   */
  notes: WorkingNotesState;
  /** Where the context stood at the end. Shown on the task trace. */
  ledger: ReturnType<ContextLedger["snapshot"]>;
}

/**
 * Turns the conversation Rust sent into the loop's own message shape.
 *
 * ## Why this validates rather than casts
 *
 * The history crosses a JSON-RPC channel, and a message the loop does not
 * understand does not fail here — it fails at the provider, as a
 * malformed-request error partway through a turn, which reads to everyone
 * involved like a bug in the agent loop. So an entry that is not a `user` or
 * `assistant` turn carrying non-empty text is dropped, and nothing is guessed
 * at: a role this does not recognise is not silently rewritten to `user`,
 * because putting the model's own words in the person's mouth is a worse
 * outcome than a shorter history.
 *
 * Timestamps are synthesised in order rather than carried. The loop uses them
 * only to order messages for compaction's cut-point search, and the order is
 * exactly what the array already encodes — whereas a real `createdAt` from the
 * conversation file would be days old and would sort this turn's seeded history
 * against a live tool result by wall clock, which is not the ordering
 * compaction wants.
 */
export function seedMessages(
  history: RunRequest["history"],
  model: Pick<Model, "id" | "api" | "provider">,
  dropped = 0,
): AgentMessage[] {
  if (!Array.isArray(history) || history.length === 0) return [];
  const base = Date.now() - history.length - 1;
  const seeded: AgentMessage[] = [];

  // A conversation that was shortened says so, before the part that survived.
  //
  // The same rule the attachment budget follows on the Rust side, for the same
  // reason: a model handed a truncated document with nothing marking the
  // truncation answers as though it read the whole thing, and a model handed a
  // truncated *conversation* will answer as though it remembers the whole
  // thing. "I do not have that earlier message" is a true answer; behaving as
  // if the conversation began where the trimming did is not.
  //
  // Written as a `user` message for the same reason `preservedMessage` in
  // `compaction.ts` is: a system-role message can be reordered away from what
  // it describes, and this has to stay immediately before the history it is
  // about.
  if (dropped > 0) {
    seeded.push({
      role: "user",
      content: [
        {
          type: "text",
          text:
            `[${dropped} earlier message(s) in this conversation did not fit this model's ` +
            `context window and are not shown below. If the answer depends on something said ` +
            `before what follows, say that you do not have it rather than reconstructing it.]`,
        },
      ],
      timestamp: base,
    } as AgentMessage);
  }
  for (const turn of history) {
    if (turn?.role !== "user" && turn?.role !== "assistant") continue;
    const text = typeof turn.content === "string" ? turn.content.trim() : "";
    if (text.length === 0) continue;
    const timestamp = base + seeded.length;
    if (turn.role === "user") {
      seeded.push({
        role: "user",
        content: [{ type: "text", text }],
        timestamp,
      } as AgentMessage);
      continue;
    }
    seeded.push({
      role: "assistant",
      content: [{ type: "text", text }],
      // The model answering *now*, not the one that answered then.
      //
      // These three fields are the loop's record of which provider produced a
      // message, and the honest answer for a replayed turn is "we are not
      // carrying that" — the conversation store keeps the model *name* for the
      // person to read, not the api/provider triple this wants. Naming the
      // current model is what the transports expect to find on a message they
      // are about to send back, and it is also the truthful description of the
      // request being made: this text is going to *this* model, whoever wrote
      // it. Which model originally answered stays visible where it has always
      // been, on the message in the chat surface.
      api: model.api,
      provider: model.provider,
      model: model.id,
      // Zeroes, because nothing here was measured on this turn. The ledger
      // reconciles against `message_end` events, which only the loop's own
      // turns emit, so these never reach a total — and a fabricated count here
      // would show up in the context meter as tokens somebody was charged.
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
      // Every message that reaches here passed the Rust side's eligibility
      // check, which admits only turns that finished. A seeded message claiming
      // `length` or `error` would tell the model the conversation is full of
      // failures it is being asked to continue.
      stopReason: "stop",
      timestamp,
    } as AgentMessage);
  }
  return seeded;
}

/**
 * A model served locally has no price and no vendor.
 *
 * agent-core requires the cost table, and zeros are the truthful entry: the
 * marginal cost of a token on a machine the organisation already owns is not a
 * number this product should invent. Anything non-zero here would show up in
 * run manifests as a fabricated figure.
 */
const LOCAL_COST = { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } as const;

/** Stands in for the credential a loopback inference server does not want. */
const LOCAL_PLACEHOLDER_KEY = "sovereign-local";

/** Loopback-only. Anything else is a routing bug and is refused before a socket opens. */
function assertLoopback(baseUrl: string): void {
  let url: URL;
  try {
    url = new URL(baseUrl);
  } catch (cause) {
    throw new Error(`Model baseUrl is not a URL: ${baseUrl}`, { cause });
  }
  const host = url.hostname.replace(/^\[|\]$/g, "");
  const loopback = host === "localhost" || host === "::1" || /^127\./.test(host);
  if (!loopback) {
    throw new Error(
      `Model endpoint ${url.origin} is not loopback. This runtime only reaches inference servers on this machine.`,
    );
  }
}

function toModel(spec: RunRequest["model"]): Model {
  assertLoopback(spec.baseUrl);
  return {
    id: spec.id,
    name: spec.name ?? spec.id,
    api: "openai-completions",
    provider: spec.provider,
    baseUrl: spec.baseUrl,
    reasoning: spec.reasoning ?? false,
    input: spec.input ?? ["text"],
    cost: LOCAL_COST,
    contextWindow: spec.contextWindow,
    maxTokens: spec.maxTokens ?? 4096,
  } as Model;
}

/**
 * The chat template's own scaffolding, which is charged to every request and
 * appears in none of the text this process holds.
 *
 * Role markers, the begin/end-of-turn tokens, the tool-call preamble a
 * template emits when tools are present. Measured against llama-server's
 * `/tokenize` on this product's own prompts, it lands between 150 and 300
 * tokens for a request with a tool catalogue attached. Budgeting as though it
 * were zero is how a request that was calculated to fit exactly does not.
 *
 * Matches `TEMPLATE_OVERHEAD_TOKENS` in `commands/agent.rs`, which charges the
 * same scaffolding to the document budget.
 */
const TEMPLATE_OVERHEAD_TOKENS = 256;

/**
 * The least of the window that is kept for the conversation, whatever the
 * tools cost.
 *
 * Tool *results* land in the transcript, and a run whose window is all schema
 * has nowhere to put the passage it just retrieved. A fifth is the smallest
 * share that leaves room for a search result and the turn that reads it.
 */
const CONVERSATION_FLOOR_SHARE = 0.2;

/**
 * The most of the window the tool catalogue may occupy, however much is free.
 *
 * A large window is not a reason to spend half of it on prose the model reads
 * once. This ceiling is what stops a 32k model carrying the full catalogue
 * *and* being unable to hold the document it was asked about.
 */
const TOOL_CEILING_SHARE = 0.45;

/**
 * How many tokens the tool catalogue may occupy on this model.
 *
 * The arithmetic is the whole point, so it is written out rather than tuned:
 * the window, less the reply the model must have room to write, less the
 * system prompt and the question that are going out whatever happens, less the
 * template's scaffolding, less a floor kept for the conversation and the tool
 * results that land in it.
 *
 * Returns `0` — meaning "no budget could be worked out" — only when the window
 * itself is unknown. When the window is known but tiny, the answer is `1`
 * rather than `0`: the catalogue is then compressed as far as it goes and
 * trimmed of tools, which is a reported degradation. Returning `0` there would
 * be read as "leave the catalogue alone", which is the behaviour that sent a
 * 9,238-token request at an 8,192-token server.
 */
export function toolBudgetFor(window: number, systemPrompt: string, prompt: string): number {
  if (!Number.isFinite(window) || window <= 0) return 0;
  const reserve = settingsForWindow(window).reserveTokens;
  const committed =
    reserve +
    estimateTextTokens(systemPrompt) +
    estimateTextTokens(prompt) +
    TEMPLATE_OVERHEAD_TOKENS;
  const free = window - committed;
  const conversationFloor = Math.max(512, Math.floor(window * CONVERSATION_FLOOR_SHARE));
  const affordable = Math.min(free - conversationFloor, Math.floor(window * TOOL_CEILING_SHARE));
  return Math.max(1, affordable);
}

/** A run in flight, so `run.abort` can reach it. */
export interface ActiveRun {
  abort(reason?: unknown): void;
  /**
   * Injects a correction into a run already in flight.
   *
   * The alternative an operator has otherwise is to stop the run and start
   * again, losing every tool result gathered so far. On a task that has already
   * read a 200-page drawing set that is an expensive way to say "use the 2019
   * revision".
   *
   * Applied at the next point the loop is safe to interrupt — before an
   * unstarted tool call or the next model turn — never in the middle of one.
   */
  steer(text: string): void;
  /**
   * Updates the state this side must preserve, and the run's notes.
   *
   * Pushed from Rust rather than pulled, because everything in it — the plan,
   * the approvals, the evidence markers — is decided there, and a pull would
   * mean this side asking mid-compaction over a channel that is also carrying
   * the tool call it is compacting around.
   */
  note(update: { preserved?: PreservedState; notes?: Partial<WorkingNotesState> }): void;
  /** The notes as they stand. Read when the run ends. */
  readonly notes: WorkingNotes;
}

/**
 * Runs one prompt to completion.
 *
 * Resolves when the agent goes idle. Rejects only for failures that are not the
 * model's to recover from -- a refused tool call is a tool result, not an
 * exception, because the model can read it and try something else.
 */
/**
 * A timer that only fires when nothing has happened.
 *
 * Separated from the run so the policy can be tested without a model, a loop or
 * a child process: the question "does a run that keeps working stay alive, and
 * does one that goes quiet get stopped?" needs neither.
 */
export function createStallGuard(windowMs: number, onStall: () => void) {
  let timer: ReturnType<typeof setTimeout> | undefined;
  const stop = () => {
    if (timer) clearTimeout(timer);
    timer = undefined;
  };
  return {
    /** Something happened. Start the window again. */
    progress() {
      stop();
      timer = setTimeout(onStall, windowMs);
      // Never a reason on its own to hold the process open.
      timer.unref?.();
    },
    stop,
  };
}

export async function startRun(
  peer: RpcPeer,
  request: RunRequest,
  register: (run: ActiveRun) => void,
): Promise<RunOutcome> {
  const { runId } = request;
  const ledger = new GrantLedger();
  const runtime = createLlmRuntime();
  registerBuiltInApiProviders(runtime.registry);

  const model = toModel(request.model);

  // Seeded from what Rust sent. On a first attempt that is nothing; on a
  // resumption it is the record of what already happened, including the side
  // effects that must not happen twice.
  const notes = WorkingNotes.from(request.notes);
  let preserved: PreservedState = { ...(request.preserved ?? {}) };

  // The notes are kept from what the tools returned rather than from what the
  // model chose to write down. See `note-taking.ts` — the entries that make a
  // resumption safe are exactly the ones a model does not think to record.
  // Deferred discovery: Rust says which tools this run is eligible for, and
  // only those get their schema loaded. The plan that decides it was fixed
  // before the model was told anything, so nothing the model does afterwards
  // can widen this — and a tool it is never shown is one it cannot spend a turn
  // being refused for asking about.
  //
  // A catalogue that could not be fetched comes back empty, which is the
  // failing-closed reading: silence from the gateway is not a list of tools.
  const catalogue = await fetchCatalogue(peer, runId);
  const offered = buildTools(
    peer,
    ledger,
    runId,
    request.model.id,
    (observation) => observeToolResult(notes, observation),
    catalogue.tools,
  );

  // The conversation this turn continues, as loop messages.
  //
  // Validated rather than trusted: this arrives over a JSON-RPC channel, and a
  // role the loop does not understand becomes a message the provider rejects —
  // which surfaces as the whole turn failing with a malformed-request error
  // that reads like a bug in the agent loop. An entry that is not a `user` or
  // `assistant` turn with text is dropped, not guessed at.
  const seeded = seedMessages(request.history, model, request.historyDropped ?? 0);

  const contextLedger = new ContextLedger(request.model.contextWindow ?? 0);
  // Measured once. Neither the system prompt nor the tool catalogue changes
  // during a run, and re-counting them every turn would spend real time
  // counting characters that are identical to last turn's.
  contextLedger.setText("system", request.systemPrompt);

  // ───────────────────────────────────────────────────────────────────────
  // The tool catalogue, cut to what this window can actually carry.
  //
  // This is the fix for the failure that produced
  // `400 request (9238 tokens) exceeds the available context size (8192
  // tokens)` on a five-word question. The catalogue serialises to about 9,200
  // tokens; the server was started with 8,192. No amount of history trimming
  // on the Rust side or compaction on this one could recover that, because
  // none of it is history and none of it is a message — it is the fixed cost
  // of the request, and until now nothing measured it against the window
  // before sending it.
  //
  // What is left for tools is the window, less everything else the request
  // must carry: the system prompt, the question, the conversation seeded from
  // earlier turns, the reply the model has to have room to write, and the
  // chat template's own scaffolding. See `tool-budget.ts` for why the
  // catalogue is compressed rather than truncated and why tools are dropped
  // only as a last resort.
  // ───────────────────────────────────────────────────────────────────────
  const fitted = fitToolsToBudget(
    offered,
    toolBudgetFor(request.model.contextWindow ?? 0, request.systemPrompt, request.prompt),
  );
  const tools = fitted.tools;
  contextLedger.setText(
    "toolSchema",
    tools.map((tool) => `${tool.name}${tool.description ?? ""}${JSON.stringify(tool.parameters ?? {})}`).join(""),
  );
  if (fitted.report.stage !== "full" || fitted.report.dropped.length > 0) {
    // Said out loud rather than done quietly. A run whose tool guidance was
    // shortened may pick a worse tool, and a run whose tools were dropped
    // cannot do part of its job — neither is a thing to discover from an
    // answer that is subtly wrong.
    process.stderr.write(
      `[agent-runtime:log] [context] run=${runId} tools fitted stage=${fitted.report.stage} ` +
        `tokens=${fitted.report.tokens} was=${fitted.report.tokensBefore} ` +
        `budget=${fitted.report.budget} kept=${tools.length} ` +
        `dropped=${fitted.report.dropped.length}${
          fitted.report.dropped.length > 0 ? ` (${fitted.report.dropped.join(", ")})` : ""
        }\n`,
    );
    peer.notify("run.event", {
      runId,
      event: { type: "tools_fitted", ...fitted.report, kept: tools.length },
    });
  }

  const compactor = new RunCompactor({
    model,
    runtime,
    apiKey: LOCAL_PLACEHOLDER_KEY,
    notes,
    ledger: contextLedger,
    // Read at the moment of compaction rather than captured, so a decision
    // taken two turns ago is carried and one taken since the run started is
    // not silently the stale copy.
    preserved: () => ({
      ...preserved,
      evidenceRefs: preserved.evidenceRefs ?? notes.state.evidenceIds,
      unresolvedIssues: preserved.unresolvedIssues ?? notes.state.openQuestions,
      recentFiles: preserved.recentFiles ?? notes.state.artifactIds,
    }),
    onCompacted: (event) => {
      peer.notify("run.event", {
        runId,
        event: { type: "context_compacted", ...event },
      });
      // The ledger moved, and the meter should show it moving. Emitted after
      // the compaction frame so a consumer folding both in order ends on the
      // post-compaction reading rather than the one that triggered it.
      publishLedger("compaction");
    },
  });

  const agent = new Agent({
    // Timed on the outside of the repair wrapper, so a call the repair layer
    // re-issues is counted as the second call it is. Counting them together
    // would report one very slow model instead of two ordinary ones, which is
    // the distinction the measurement exists to make.
    streamFn: withCallTiming(
      withToolCallRepair(
        runtime.streamSimple,
        tools.map((tool) => tool.name),
      ),
      runId,
    ),
    /**
     * The harness converter, not the default.
     *
     * agent-core's default keeps only user, assistant and tool-result messages
     * and silently drops everything else. Two things depend on that not
     * happening, and both fail quietly rather than loudly:
     *
     * - **Compaction.** Its output is a `compactionSummary` message. Dropped,
     *   compaction appears to work while discarding the very thing it produced,
     *   and the model simply loses the earlier history.
     * - **Interrupt recovery.** When an operator stops a run mid-tool, `Agent`
     *   appends a `custom` message saying the previous turn was interrupted and
     *   tools may have partially executed. Dropped, a continuation is never told,
     *   and may repeat a write that already happened.
     */
    convertToLlm,
    transformContext: (messages, signal) => compactor.transform(messages, signal),
    /**
     * Read-only tools run together.
     *
     * A document task typically wants several searches at once. Executing them
     * one at a time makes the operator wait for the sum rather than the slowest,
     * for no safety gain — each call is still authorised individually, and a
     * search cannot affect what another search returns. Anything that writes
     * declares `executionMode: "sequential"` on itself.
     */
    toolExecution: "parallel",
    /**
     * A correction is applied at the next safe point, not queued behind the
     * whole run. `one-at-a-time` so two rapid corrections do not both land
     * before the model has responded to either.
     */
    steeringMode: "one-at-a-time",
    initialState: {
      systemPrompt: request.systemPrompt,
      model,
      tools,
      /**
       * The conversation this turn continues.
       *
       * Seeded as the loop's starting transcript, and deliberately *not*
       * merged into the prompt. `agent.prompt(request.prompt)` below appends
       * the new question to whatever this holds — one append, one question,
       * always — and because nothing here is the live question, there is no
       * arrangement of these messages under which the loop can decide the work
       * is already done.
       *
       * Empty on a first turn, which is `Agent`'s own default and therefore
       * the behaviour that shipped before this existed.
       */
      messages: seeded,
      /**
       * Why this is set at all, and why it decides whether anything streams.
       *
       * `Agent` defaults `thinkingLevel` to `"off"`. That value is not just a
       * display preference — it flows into the transport and decides whether
       * the model's reasoning is *forwarded at all*:
       *
       *   thinkingLevel "off"
       *     -> resolveAgentReasoningOption returns the off fallback
       *     -> streamSimpleOpenAICompletions computes reasoningEffort = undefined
       *     -> shouldEmitReasoning = false
       *     -> every reasoning delta is dropped, and the partitioner that
       *        watches for reasoning tags holds the visible text back with it.
       *
       * The consequence, measured on this machine: a Qwen3.5-9B turn that
       * reasoned for fifty-seven seconds delivered its whole 342-character
       * answer as a *single* `text_delta`, with no reasoning events at all.
       * The event census the translator writes read `text_delta=1 text_end=1
       * text_start=1`. Nothing streamed because nothing was being sent.
       *
       * The model reasons either way — that was confirmed by reading the raw
       * SSE off llama-server, which emitted sixty-two `reasoning_content`
       * frames with `content` null throughout. So this does not make the model
       * do more work. It stops ARJUN throwing away work already done.
       *
       * `"off"` for a model whose chat template has no reasoning switch, which
       * is the honest answer for it and keeps the request unchanged.
       */
      thinkingLevel: request.model.supportsReasoning ? "medium" : "off",
    },
    beforeToolCall: (context) => authorizeToolCall(peer, ledger, runId, context),
    /**
     * A local inference server needs no credential, but the OpenAI client
     * refuses to construct without one. So a placeholder is supplied rather
     * than the transport being special-cased for local providers.
     *
     * It is a constant on purpose: there is no secret here to leak, and reading
     * a real key from the environment would create a path by which one could
     * reach a local endpoint and, from there, a log.
     */
    getApiKey: () => LOCAL_PLACEHOLDER_KEY,
    /**
     * Local-model quirks, applied to every request.
     *
     * Installed unconditionally because the policy is a no-op for models that
     * need nothing — a per-model switch is one an operator would have to
     * remember to set, and forgetting produces an approval note that opens with
     * the model thinking out loud.
     */
    onPayload: payloadPolicy(request.model.reasoning ?? false, request.model.supportsReasoning),
  });

  /**
   * Stops the run when its deadline passes.
   *
   * `agent.abort` is the same path an operator's stop button takes, so the
   * wind-down is the one already tested: the loop finishes what it is doing,
   * appends the message saying the turn was interrupted, and returns whatever
   * it had. A deadline that killed the process instead would leave a tool call
   * in flight and nobody able to say whether it took effect.
   */
  /**
   * Why this run was stopped, when something stopped it.
   *
   * The loop reports an abort as `stopReason: "aborted"` and says nothing about
   * who asked. A deadline and an operator's stop button are the same event to
   * agent-core and different endings to a person reading the run afterwards, so
   * the cause is recorded at the point it is known -- here -- rather than
   * guessed from the wording of a message later.
   *
   * First writer wins: the run stops once, and the first thing to ask for it is
   * the reason it stopped.
   */
  let abortCause: RunTermination | null = null;
  const causedBy = (cause: RunTermination) => {
    if (abortCause === null) abortCause = cause;
  };

  /**
   * How long the run may produce *nothing at all* before it is stopped.
   *
   * The deadline below is a ceiling on total time, and a ceiling on total time
   * cannot tell a slow model from a stuck one. On this machine Nemotron3-Nano
   * decodes at about five tokens a second, so a 2.3k-token answer takes over
   * eight minutes of continuous, healthy work - and a ten-minute ceiling
   * stopped it partway with nothing to show, for a question it had already
   * answered correctly once.
   *
   * Stalling is the thing actually worth catching, and it has a signal: a run
   * that is working emits reasoning deltas, text deltas or tool activity. This
   * timer is rearmed by any of them, so a healthy slow run resets it every few
   * hundred milliseconds and a wedged one never does.
   *
   * Four minutes rather than one, because the quiet period before the first
   * token is real work too: a long prompt has to be prefilled before anything
   * can be emitted, and on a small GPU that is not instant.
   */
  const STALL_MS = 4 * 60 * 1000;

  const guard = createStallGuard(STALL_MS, () => {
    causedBy({
      kind: "budgetStopped",
      detail:
        "Stopped: it produced nothing for four minutes, so it was treated as stuck rather " +
        "than slow.",
    });
    agent.abort("the task stopped producing output");
  });
  const noteProgress = () => guard.progress();
  const clearStall = () => guard.stop();
  noteProgress();

  let deadlineTimer: ReturnType<typeof setTimeout> | undefined;
  if (typeof request.deadlineMs === "number") {
    const remaining = request.deadlineMs - Date.now();
    if (remaining <= 0) {
      // Already past it before the first turn. Refused rather than started, so
      // a run that waited too long in a queue does not spend a model call to
      // discover it has no time left.
      throw new Error(
        "This task's time budget had already expired before the loop started, so nothing was run.",
      );
    }
    deadlineTimer = setTimeout(() => {
      causedBy({
        kind: "budgetStopped",
        detail: "Stopped: it ran past the time its plan allowed.",
      });
      agent.abort("the task reached the time limit its plan allowed");
    }, remaining);
    // Never hold the process open on its own account: if everything else has
    // finished, a pending deadline is not a reason to stay alive.
    deadlineTimer.unref?.();
  }

  let turns = 0;
  // Stateful translator. Without state, llama-server's `text_start` +
  // `text_delta*` + `text_end` triple would each carry the full accumulated
  // text, producing the same answer 2-3 times in a row. The translator
  // tracks which (run, content block) has been sent and only emits a
  // `message_update` for genuinely new text.
  const translator = new MessageTranslator(request.messageId);

  /**
   * Publishes the ledger as it stands.
   *
   * Sent on its own frame rather than folded into `context_compacted`, because
   * the two answer different questions. A compaction event says history was
   * lost; this says where the window stands *right now*, including on the great
   * majority of turns where nothing was lost at all. Before this existed the
   * surface could only learn the ledger when the run finished, so the meter a
   * person watches while deciding whether to attach one more file was always
   * describing the previous run.
   */
  const publishLedger = (reason: string) => {
    peer.notify("run.event", {
      runId,
      event: { type: "context_ledger", reason, ledger: contextLedger.snapshot() },
    });
  };

  agent.subscribe((event: AgentEvent) => {
    // Any event at all is a sign of life: a reasoning delta, a token of the
    // answer, a tool starting or finishing. What matters is that something
    // happened, not what it was.
    noteProgress();
    if (event.type === "turn_end") turns += 1;

    // Reconciliation, on every model call rather than periodically.
    //
    // `message_end` is the moment the provider's usage is in hand, and it is
    // the only moment it is: the numbers are on the finished assistant message
    // and nowhere else. Doing this per turn is what keeps the running total a
    // measurement for the whole life of the run instead of only at the end.
    if (event.type === "message_end") {
      const message = event.message;
      const usage = message.role === "assistant" ? message.usage : undefined;
      // `estimatedIn` is read before the correction is applied, or the drift
      // would be computed against a figure that had already been corrected and
      // would read as zero on every turn.
      const estimatedIn = contextLedger.snapshot().occupied;
      const actualIn = usage?.input ?? null;
      contextLedger.reconcile({
        estimatedIn,
        actualIn,
        actualOut: usage?.output ?? null,
      });
      contextLedger.applyMeasuredInput(actualIn);
      publishLedger("turn");
    }
    // Best-effort by design: a dropped event costs the operator a progress line,
    // whereas awaiting delivery would let a slow UI stall the run.
    //
    // Two-pass forwarding. The OpenClaw `message_*` events carry a different
    // shape than the Arjun chat surface expects, so `Translator` maps them
    // to the wire contract (with the front-end's `messageId` attached).
    // Everything else is forwarded as-is after `redactEvent` strips tool
    // arguments. Both lists are merged so the chat sees a single ordered
    // stream of `run.event` frames.
    const translated = translator.translate(event);
    for (const wire of translated) {
      peer.notify("run.event", { runId, event: wire });
    }
    // The message stream belongs to the translator, exclusively.
    //
    // Forwarding the raw event when the translator produced nothing is what
    // put the loop's own `message_start` / `message_end` frames -- which carry
    // the *whole* `AgentMessage`, prompt text and tool output included -- onto
    // the chat channel for every user and tool-result message. The consumer
    // ignored them because they carry no `messageId`, so the only thing they
    // ever did was send document text down a second path. A translator that
    // declines to translate a message event has decided it is not part of the
    // chat cell; that decision is the answer, not a reason to fall back.
    if (!isMessageStreamEvent(event) && translated.length === 0) {
      peer.notify("run.event", { runId, event: redactEvent(event) });
    }
  });

  register({
    abort: (reason) => {
      causedBy({
        kind: "aborted",
        detail: typeof reason === "string" && reason.trim().length > 0
          ? `Stopped: ${reason}`
          : "Stopped by request.",
      });
      agent.abort(reason);
    },
    steer: (text) =>
      agent.steer({
        role: "user",
        content: [{ type: "text", text }],
        timestamp: Date.now(),
      }),
    note: (update) => {
      if (update.preserved) preserved = { ...preserved, ...update.preserved };
      if (update.notes) applyNotes(notes, update.notes);
    },
    notes,
  });

  /** Whether the salvage turn below has already been spent. */
  let salvaged = false;

  try {
    await agent.prompt(request.prompt);

    // ─────────────────────────────────────────────────────────────────────
    // The turn that ended having said nothing.
    //
    // Reported as: "sometimes it generates no output and stops answering
    // automatically without showing any answer or error." That is not a
    // metaphor for a crash — it is precisely what happened. The loop returned
    // normally with `stopReason: "stop"`, the ending classified as
    // `completed`, the answer was the empty string, and the chat cell closed
    // with nothing in it. `completed` carries no detail to display, so the
    // surface had an empty message and no error to show beside it.
    //
    // The cause, measured on this machine: a reasoning model can spend a whole
    // turn in `reasoning_content` and emit no `content` at all — the second
    // screenshot of the bug report shows Qwen3.5-9B composing an entire
    // document inside its thinking block — and a loop that stops on a tool
    // call it never followed up leaves the same shape.
    //
    // Both are recoverable, and the remedy is one more turn: the work is
    // already done and in the transcript, and what is missing is the sentence
    // that says so. Asked for it, the model writes it. Asking costs one call
    // on a path that currently produces nothing at all.
    //
    // Deliberately inside the `try`. The `finally` below closes the chat cell,
    // and text streamed after that reaches a cell the surface has already
    // finished — so the recovery has to happen while the cell is still open.
    // ─────────────────────────────────────────────────────────────────────
    if (
      needsSalvage(agent.state, abortCause) &&
      // An operator who pressed stop is not asking for one more model call, and
      // a deadline that has passed has no time to spend on one.
      (typeof request.deadlineMs !== "number" || Date.now() < request.deadlineMs)
    ) {
      salvaged = true;
      process.stderr.write(
        `[agent-runtime:log] [answer] run=${runId} the turn ended with no visible text; ` +
          "asking once for the answer itself\n",
      );
      peer.notify("run.event", {
        runId,
        event: { type: "answer_salvage_started", runId },
      });
      noteProgress();
      await agent.prompt(SALVAGE_PROMPT);
    }
  } finally {
    if (deadlineTimer) clearTimeout(deadlineTimer);
    clearStall();
    // Exactly one terminal event per run, on every path out.
    //
    // `agent_end` covers the ordinary exits and has already closed the cell by
    // the time control reaches here; this covers the ones where the loop threw
    // before it could emit anything. `finalize` is idempotent, so the common
    // case costs nothing and the uncommon one no longer leaves a chat cell
    // streaming forever.
    for (const wire of translator.finalize(agent.state.messages)) {
      peer.notify("run.event", { runId, event: wire });
    }
    // A run that ends holding grants means authorisation outlived its call. That
    // is a defect, but clearing is the safe half of handling it either way.
    ledger.clear();
  }

  const messages = agent.state.messages;
  // The last *assistant* message, not the last message.
  //
  // These are not the same thing on exactly the path where it matters most. A
  // run that is stopped has an interrupt message appended after the assistant
  // turn — `Agent` adds a `custom` message saying the previous turn was
  // interrupted and tools may have partially executed — so reading
  // `messages[messages.length - 1]` found *that*, not the answer, and the run
  // reported no text at all.
  //
  // The consequence was quiet and only visible afterwards: the chat cell kept
  // what had streamed into it, because the reducer persists its own buffer, but
  // the *task record* recorded an empty answer for every stopped run. So a turn
  // somebody stopped halfway through a useful answer was, in the durable
  // record, a turn that produced nothing.
  //
  // Found the same way `terminationOf` below already does it, which is what
  // makes the inconsistency obvious in hindsight: the ending was read off the
  // last assistant message while the text was read off whatever happened to be
  // last.
  const { text, finalAssistant } = answerOf(messages);

  let outcome = terminationOf({
    finalAssistant,
    errorMessage: agent.state.errorMessage,
    abortCause,
  });

  // A run that finished cleanly and said nothing is not a run that finished.
  //
  // The salvage turn above is the recovery; this is what happens when it did
  // not work, or when the ending ruled it out. `completed` is the one ending
  // that carries no sentence to display, so classifying this as `completed`
  // is what produced a blank chat cell with no answer and no error beside it.
  // `failed` is the honest reading — the turn was asked a question and did not
  // answer it — and it is the one that puts something on the screen.
  //
  // Only `completed` is reclassified. Every other ending already says what
  // happened, and a stopped or length-limited run that produced no text has
  // been explained by the thing that stopped it.
  if (outcome.kind === "completed" && text.trim().length === 0) {
    outcome = {
      kind: "failed",
      detail: salvaged
        ? "The model finished without writing an answer, and did not write one when asked " +
          "again. Its reasoning may have run to the end of the turn without producing any " +
          "visible text. Try the question again, or a model with a larger output limit."
        : "The model finished without writing an answer. Try the question again.",
    };
    process.stderr.write(
      `[agent-runtime:log] [answer] run=${runId} ended with no text after ${turns} turn(s)` +
        `${salvaged ? " and one salvage attempt" : ""}\n`,
    );
  }

  return {
    runId,
    text,
    turns,
    outcome,
    stopReason: agent.state.errorMessage ? "error" : undefined,
    notes: notes.state,
    ledger: contextLedger.snapshot(),
  };
}

/**
 * What to say to a model that finished a turn without saying anything.
 *
 * Three clauses, and each one closes a way the second attempt could go wrong:
 *
 * - **Do not call any tool.** The work is already done and its results are in
 *   the transcript. A salvage turn that re-ran a write would produce the
 *   document twice, which is a far worse failure than the silence it is
 *   recovering from.
 * - **Write it as the answer, not as reasoning.** This turn exists precisely
 *   because a model spent the last one thinking; saying so is the whole
 *   instruction.
 * - **Say what you could not do.** A model with nothing to report must be able
 *   to report that, or it is being pushed towards inventing an answer — which
 *   would turn a visible failure into an invisible one.
 */
const SALVAGE_PROMPT =
  "You ended that turn without writing anything the person can read. Write the answer now, " +
  "as your visible reply rather than as reasoning. Do not call any tool: use only what is " +
  "already in this conversation, including any tool results above. If a file or document was " +
  "produced, say what it is and where it is. If you could not complete the task, say plainly " +
  "what you could not do and why — do not invent an answer to fill the gap.";

/**
 * The answer a run produced, and the message it came from.
 *
 * The last *assistant* message, not the last message.
 *
 * These are not the same thing on exactly the path where it matters most. A
 * run that is stopped has an interrupt message appended after the assistant
 * turn — `Agent` adds a `custom` message saying the previous turn was
 * interrupted and tools may have partially executed — so reading
 * `messages[messages.length - 1]` found *that*, not the answer, and the run
 * reported no text at all.
 *
 * The consequence was quiet and only visible afterwards: the chat cell kept
 * what had streamed into it, because the reducer persists its own buffer, but
 * the *task record* recorded an empty answer for every stopped run. So a turn
 * somebody stopped halfway through a useful answer was, in the durable record,
 * a turn that produced nothing.
 *
 * Found the same way `terminationOf` does it, which is what makes the old
 * inconsistency obvious in hindsight: the ending was read off the last
 * assistant message while the text was read off whatever happened to be last.
 *
 * Lifted out of `startRun` so the run can ask the same question twice — once
 * to decide whether the turn needs rescuing, and once at the end to report
 * what it produced — without two readings of "the answer" that could disagree.
 */
export function answerOf(messages: readonly unknown[]): {
  text: string;
  finalAssistant: { stopReason?: unknown; errorMessage?: unknown } | undefined;
} {
  const finalAssistant = [...messages].reverse().find((message) => isAssistantMessage(message));
  const assistantContent = (finalAssistant as { content?: unknown } | undefined)?.content;
  const text = Array.isArray(assistantContent)
    ? assistantContent
        .filter((block): block is { type: "text"; text: string } => block?.type === "text")
        .map((block) => block.text)
        .join("\n")
    : "";
  return { text, finalAssistant };
}

/**
 * Whether this turn ended having said nothing, and could still be rescued.
 *
 * Four conditions, and each one excludes a case where asking again would be
 * wrong rather than merely wasteful:
 *
 * - **Nothing visible was produced.** The whole premise.
 * - **Nothing stopped it.** A run an operator halted is not asking for one more
 *   model call, and a run its budget stopped has none to spend.
 * - **Nothing errored.** A failed provider call is already explained, and
 *   re-issuing it against a server that just refused is not a recovery.
 * - **The loop thinks it finished.** `completed` is the one ending that carries
 *   no sentence to display, which is exactly why an empty one is invisible;
 *   every other ending already says what happened.
 *
 * Separated from the run so the policy is readable on its own, which matters
 * because the cost of getting it wrong is a duplicate model call on a path that
 * was working.
 */
export function needsSalvage(
  state: { messages: readonly unknown[]; errorMessage?: string },
  abortCause: RunTermination | null,
): boolean {
  if (abortCause !== null) return false;
  if (state.errorMessage) return false;
  const { text, finalAssistant } = answerOf(state.messages);
  if (text.trim().length > 0) return false;
  return (
    terminationOf({ finalAssistant, errorMessage: state.errorMessage, abortCause }).kind ===
    "completed"
  );
}

/**
 * Reads the run's ending off what the loop actually did.
 *
 * Deliberately not "the request resolved, so it worked". A loop that was
 * stopped, that errored, or that ran into the model's output cap all return
 * normally from `agent.prompt` -- they are ordinary endings of an agent loop,
 * not transport faults -- and treating a resolved promise as success is what
 * made every one of them indistinguishable from an answer.
 */
export function terminationOf(state: {
  finalAssistant?: { stopReason?: unknown; errorMessage?: unknown };
  errorMessage?: string;
  abortCause?: RunTermination | null;
}): RunTermination {
  const stopReason = state.finalAssistant?.stopReason;

  if (stopReason === "aborted") {
    // Who asked is knowable only where the abort was requested, so it is
    // recorded there. Without a recorded cause the honest answer is the
    // general one: something stopped it.
    return state.abortCause ?? { kind: "aborted", detail: "Stopped before it finished." };
  }

  if (stopReason === "error") {
    const detail =
      firstString(state.finalAssistant?.errorMessage, state.errorMessage) ??
      "The model call failed.";
    return { kind: "failed", detail };
  }

  if (stopReason === "length") {
    return {
      kind: "lengthLimited",
      detail:
        "Stopped: the answer reached the output limit for one turn, so it is cut off mid-way.",
    };
  }

  // An error the loop recorded without it reaching the final message -- a
  // provider that failed after the last assistant turn, say. Still a failure.
  if (typeof state.errorMessage === "string" && state.errorMessage.trim().length > 0) {
    return { kind: "failed", detail: state.errorMessage };
  }

  // `stop` and `toolUse` both land here. A loop that stopped on `toolUse`
  // exhausted its turns rather than its model, and the core -- which owns the
  // budget -- is the side that can say so.
  return { kind: "completed" };
}

/** The first of these that is a non-empty string. */
function firstString(...candidates: unknown[]): string | undefined {
  for (const candidate of candidates) {
    if (typeof candidate === "string" && candidate.trim().length > 0) return candidate;
  }
  return undefined;
}

/**
 * Folds an update into the notes, through the setters rather than by assignment.
 *
 * The caps and the de-duplication live in the setters. Assigning the fields
 * directly would let one `run.note` carrying a hundred evidence markers put a
 * hundred markers into a list whose ceiling is sixty-four — which is how a
 * bounded structure quietly stops being bounded.
 */
function applyNotes(notes: WorkingNotes, update: Partial<WorkingNotesState>): void {
  if (typeof update.goal === "string") notes.setGoal(update.goal);
  if (update.stage) notes.atStage(update.stage.ordinal, update.stage.intent);
  if (typeof update.nextAction === "string") notes.setNextAction(update.nextAction);
  for (const decision of update.decisions ?? []) {
    notes.decided(decision.what, decision.because, decision.at);
  }
  for (const id of update.evidenceIds ?? []) notes.sawEvidence(id);
  for (const id of update.calculationIds ?? []) notes.calculated(id);
  for (const id of update.artifactIds ?? []) notes.produced(id);
  for (const question of update.openQuestions ?? []) notes.asked(question);
  for (const effect of update.completed ?? []) {
    notes.didEffect(effect.tool, effect.target, effect.at);
  }
}

/**
 * Whether an event belongs to the chat message stream.
 *
 * These three are the translator's alone. Everything else is forwarded to the
 * surface as-is, redacted.
 */
function isMessageStreamEvent(event: AgentEvent): boolean {
  return (
    event.type === "message_start" ||
    event.type === "message_update" ||
    event.type === "message_end"
  );
}

/**
 * Strips tool *arguments* from the event stream.
 *
 * The UI needs to know a tool ran; it does not need the arguments echoed back
 * over a second channel, and those can carry document text or a file path that
 * the audit record already holds under access control. Sending less is cheaper
 * to defend than sending everything and redacting at the display.
 */
function redactEvent(event: AgentEvent): AgentEvent {
  if (
    event.type === "tool_execution_start" ||
    event.type === "tool_execution_update" ||
    event.type === "tool_execution_end"
  ) {
    return { ...event, args: undefined } as AgentEvent;
  }
  return event;
}

/**
 * Wire shape the Arjun chat surface consumes.
 *
 * The TypeScript `AgentEvent` union in `src/services/agent.service.ts` is the
 * source of truth for the contract. We mirror it here as a structural type so
 * a drift on either side fails to type-check rather than failing at runtime.
 *
 * Only the three message-stream events are part of the streaming contract;
 * every other event is forwarded as-is after `redactEvent`.
 */
type WireEvent =
  | { type: "message_start"; messageId: string; role: "assistant" }
  | { type: "message_update"; messageId: string; delta: string }
  | {
      /**
       * The model is reasoning, or has stopped.
       *
       * `characters` is the running size of the block and `elapsedMs` is how
       * long it has been going. `delta` is the reasoning itself, since the
       * last frame.
       *
       * ## Why `delta` exists, having deliberately not existed
       *
       * This event was built to carry a size and never the text: a byte
       * counter, so a reasoning pass was visibly happening without any of it
       * leaving the runtime. That held the wrong line. A model that reasons
       * for two and a half minutes before its first visible word — measured,
       * on Qwen3.5-9B at 3 tok/s — left the surface with a static label for
       * the whole run, and the person watching could not tell a long thought
       * from a hang.
       *
       * The text is **live only**, and that part has not changed. It is
       * forwarded on the streaming channel, held in a buffer separate from
       * the answer, and never written to `Message.content`, never sent as
       * `finalContent`, never resolved against by the verifier, and never
       * recorded. What survives the run is the answer and the counts; the
       * thought is shown while it happens and then it is gone.
       *
       * Absent on a frame that carries only the counter — a provider that
       * signals reasoning without sending any, or a tick between flushes.
       */
      type: "model_thinking";
      messageId: string;
      state: "start" | "active" | "end";
      characters: number;
      elapsedMs: number;
      delta?: string;
    }
  | {
      type: "message_end";
      messageId: string;
      finishReason: "stop" | "length" | "tool_calls" | "content_filter" | "error";
      tokensIn?: number;
      tokensOut?: number;
    };

/**
 * Translates OpenClaw agent events into the wire shape the Arjun chat expects.
 *
 * The chat subscribes to `agent://event` for token-level updates and filters
 * each event on `event.messageId`. OpenClaw's `message_*` events do not carry
 * a `messageId` — they carry the whole `AgentMessage` (or its
 * `assistantMessageEvent` slice) — so without this translation every
 * streaming event is dropped at the consumer and the cell stays on
 * "thinking…". The translation is the single source of contract between the
 * two event worlds and is the only place that needs to change if the wire
 * shape ever evolves.
 *
 * Safety rules:
 *  - Only `text_delta` contributes to the visible answer. `thinking_delta`
 *    is the model's chain-of-thought and is intentionally *not* exposed on
 *    the live channel; the audit record holds it under access control.
 *  - `toolcall_delta` is a wire-format repair artefact, not visible prose,
 *    so it is not forwarded as a `message_update` either.
 *  - The `messageId` is the one the front-end reserved on
 *    `agent_append_turn`; the same id appears on every event in the stream.
 *  - One OpenClaw `message_update` may carry several `assistantMessageEvent`
 *    sub-events, so the translator yields zero or more wire events per input.
 */
export class MessageTranslator {
  /**
   * Characters of each block already forwarded as `message_update`.
   *
   * Replaces three boolean sets that tried to answer "has this block been
   * sent?" and could not, because the question has a length rather than a
   * yes or no. See the streaming contract below.
   */
  private forwarded = new Map<number, number>();
  /** Blocks that have produced at least one `text_delta`. */
  private sawTextDelta = new Set<number>();
  /**
   * Whether the assistant chat cell has been opened.
   *
   * One cell per run, opened by the first *assistant* `message_start` and by
   * nothing else. The loop emits `message_start` and `message_end` for user and
   * tool-result messages too, and before this was role-aware the very first one
   * -- the user's own prompt -- opened the cell and the very first `message_end`
   * -- the same user message -- closed it again, terminating the stream before
   * the model had produced a token.
   */
  private cellOpen = false;
  /** Whether the single terminal `message_end` has gone out. */
  private cellClosed = false;
  /** When the current run of private reasoning began, or null if none is open. */
  private thinkingSince: number | null = null;
  /** How many characters of reasoning this block has produced. */
  private thinkingChars = 0;
  /** When the last `active` tick went out, so the channel is not flooded. */
  private thinkingTickedAt = 0;
  /**
   * Reasoning produced since the last frame went out.
   *
   * Buffered rather than forwarded per delta: a reasoning model emits one
   * delta per token, and one stdio frame per token is thousands a second to
   * move text a person reads at reading speed. The buffer is emptied onto
   * every frame that leaves, the closing one included, so no reasoning is
   * dropped at the end of a block.
   */
  private thinkingBuffer = "";
  /**
   * How many of each inner event type the loop delivered.
   *
   * The chat surface can only stream as finely as the events it is given, so
   * when an answer arrives in one lump the question is always the same: did
   * the model not stream, or did something between the model and here glue
   * the pieces back together? A shape count answers it in one log line, and
   * counts are all it holds — never a fragment of what the events carried.
   */
  private readonly shape = new Map<string, number>();

  constructor(
    private readonly messageId: string,
    private readonly now: () => number = Date.now,
  ) {}

  /**
   * Opens or advances the reasoning signal.
   *
   * Ticked on two clocks, because the frame carries two things now. Reasoning
   * text goes out at [`THINKING_TEXT_TICK_MS`], fast enough to read as typing
   * and slow enough that a decode at hundreds of tokens a second still costs
   * a dozen frames. The counter goes out on its own at [`THINKING_TICK_MS`]
   * regardless, so a provider that signals reasoning without sending any text
   * still moves the elapsed figure rather than looking stalled.
   */
  private thinking(delta: string): WireEvent[] {
    const at = this.now();
    this.thinkingChars += delta.length;
    this.thinkingBuffer += delta;

    if (this.thinkingSince === null) {
      this.thinkingSince = at;
      this.thinkingTickedAt = at;
      return [this.thinkingFrame("start", at)];
    }

    const sinceLast = at - this.thinkingTickedAt;
    const dueForText = this.thinkingBuffer.length > 0 && sinceLast >= THINKING_TEXT_TICK_MS;
    if (!dueForText && sinceLast < THINKING_TICK_MS) return [];
    this.thinkingTickedAt = at;
    return [this.thinkingFrame("active", at)];
  }

  /**
   * One reasoning frame, emptying the text buffer onto it.
   *
   * `delta` is omitted rather than sent empty, so a consumer can tell a frame
   * carrying reasoning from one carrying only the counter without inspecting
   * a string's length.
   */
  private thinkingFrame(state: "start" | "active", at: number): WireEvent {
    const delta = this.thinkingBuffer;
    this.thinkingBuffer = "";
    const frame = {
      type: "model_thinking" as const,
      messageId: this.messageId,
      state,
      characters: this.thinkingChars,
      elapsedMs: this.thinkingSince === null ? 0 : at - this.thinkingSince,
    };
    return delta.length > 0 ? { ...frame, delta } : frame;
  }

  /**
   * Closes the private-reasoning signal, if one is open.
   *
   * Called on `thinking_end`, but also on the first visible text and on
   * `message_end`: a model that stops reasoning by simply starting to answer
   * never sends `thinking_end`, and without this the surface would show
   * "Thinking" underneath an answer that was already being written.
   */
  private endThinking(): WireEvent[] {
    if (this.thinkingSince === null) return [];
    const elapsed = this.now() - this.thinkingSince;
    const characters = this.thinkingChars;
    // Whatever had not reached a tick yet. Without this the last fraction of
    // a second of reasoning — the sentence the model was in the middle of
    // when it started answering — is thrown away, and the panel stops
    // mid-word on every single run.
    const delta = this.thinkingBuffer;
    this.thinkingSince = null;
    this.thinkingChars = 0;
    this.thinkingBuffer = "";
    const frame = {
      type: "model_thinking" as const,
      messageId: this.messageId,
      state: "end" as const,
      characters,
      elapsedMs: elapsed,
    };
    return [delta.length > 0 ? { ...frame, delta } : frame];
  }

  /**
   * Resets the per-turn block bookkeeping.
   *
   * The dedupe sets are keyed by `contentIndex`, which restarts at zero on
   * every assistant message. Carrying turn one's set into turn two would make
   * the loop's second answer look like an echo of the first and drop it.
   */
  private beginAssistantTurn(): void {
    this.forwarded.clear();
    this.sawTextDelta.clear();
    this.thinkingSince = null;
    this.thinkingChars = 0;
    this.thinkingBuffer = "";
  }

  /**
   * Emits the run's one terminal event, if it has not been emitted already.
   *
   * Every path out of a run funnels through here, which is what makes "exactly
   * one `message_end` per run" a property of the translator rather than a thing
   * each call site has to remember.
   */
  private closeCell(
    stopReason: unknown,
    usage: { input?: number; output?: number } | undefined,
  ): WireEvent[] {
    if (this.cellClosed) return [];
    this.cellClosed = true;
    const shape = [...this.shape.entries()]
      .map(([type, count]) => `${type}=${count}`)
      .sort()
      .join(" ");
    process.stderr.write(`[agent-runtime:log] [stream] messageId=${this.messageId} ${shape}
`);
    return [
      {
        type: "message_end",
        messageId: this.messageId,
        finishReason: mapStopReason(stopReason),
        tokensIn: usage?.input,
        tokensOut: usage?.output,
      },
    ];
  }

  /**
   * Closes the cell from the run's final state, whatever path the loop took out.
   *
   * Called on `agent_end` and again by `startRun` in its `finally`, because a
   * loop that throws never reaches `agent_end` and a chat cell whose stream
   * simply stops arriving spins forever. Idempotent: the second call returns
   * nothing.
   *
   * A run that never opened a cell -- no assistant turn ever started -- closes
   * nothing. There is no cell on the surface to terminate, and inventing a
   * terminal event for one would tell the chat an answer finished that was
   * never begun.
   */
  finalize(messages: readonly unknown[] = []): WireEvent[] {
    if (!this.cellOpen || this.cellClosed) return [];
    const last = [...messages]
      .reverse()
      .find((message) => isAssistantMessage(message)) as
      | { stopReason?: unknown; usage?: { input?: number; output?: number } }
      | undefined;
    const closed = this.endThinking();
    // No final assistant message behind an open cell means the loop stopped
    // before it produced one. `error` is the honest reading; `stop` would tell
    // the surface an answer completed.
    return [...closed, ...this.closeCell(last ? last.stopReason : "error", last?.usage)];
  }

  translate(event: AgentEvent): WireEvent[] {
    if (event.type === "message_start") {
      // Role-aware. The loop emits `message_start` for user and tool-result
      // messages as well, and neither is the assistant's answer: a user
      // message opening the cell is what previously made the chat show an
      // empty assistant bubble the moment the prompt was submitted.
      if (!isAssistantMessage(event.message)) return [];
      this.beginAssistantTurn();
      // Turn two and later continue the cell turn one opened. Re-emitting
      // `message_start` would tell the consumer to clear the buffer, which is
      // how everything the model said before its first tool call disappeared.
      if (this.cellOpen) return [];
      this.cellOpen = true;
      return [{ type: "message_start", messageId: this.messageId, role: "assistant" }];
    }

    if (event.type === "message_update") {
      if (!isAssistantMessage(event.message)) return [];
      const inner = event.assistantMessageEvent;
      this.shape.set(inner.type, (this.shape.get(inner.type) ?? 0) + 1);

      if (inner.type === "text_delta") {
        const delta = (inner as { delta?: unknown }).delta;
        if (typeof delta !== "string" || delta.length === 0) return [];
        const contentIndex = (inner as { contentIndex?: number }).contentIndex ?? 0;
        this.sawTextDelta.add(contentIndex);
        // Visible text means the reasoning pass is over, whether or not the
        // model bothered to say so.
        const closed = this.endThinking();
        // Always forwarded. A delta is new text by definition, and the
        // previous code suppressed it whenever a `text_start` payload had
        // been sent for the block — which, under the race described in the
        // streaming contract above, was every run whose first network chunk
        // carried more than one frame. Everything the model said after that
        // chunk was dropped on the floor.
        this.forwarded.set(contentIndex, (this.forwarded.get(contentIndex) ?? 0) + delta.length);
        return [...closed, { type: "message_update", messageId: this.messageId, delta }];
      }

      if (inner.type === "text_start" || inner.type === "text_end") {
        const contentIndex = (inner as { contentIndex?: number }).contentIndex ?? 0;

        // A block opening carries no text worth forwarding. It used to, and
        // that is precisely what broke: its `partial` is a live reference to
        // the message the producer is still writing into, so by the time this
        // ran it held everything that had arrived in the chunk — which was
        // then emitted as one lump and used to suppress every real delta.
        // The block is opened by the deltas that follow it.
        if (inner.type === "text_start") return [];

        const partial = (inner as { partial?: { content?: Array<{ type: string; text?: string }> } })
          .partial;
        // Indexed by the block this event names, not by zero. A reasoning
        // block ahead of the text one put the thinking at index 0, and
        // reading it here compared the wrong block against the wrong length.
        const block = partial?.content?.[contentIndex] ?? partial?.content?.[0];
        if (!block || block.type !== "text" || typeof block.text !== "string") {
          return [];
        }

        // The close is a reconciliation, not a repeat: forward only the part
        // of the finished block that the deltas did not already carry.
        //
        // For a server that streams, that suffix is empty and nothing is
        // sent. For one that returns the whole message at once — no
        // `text_delta` at all — it is the entire answer, which is the case
        // this branch exists for. For one that streams and then pads, it is
        // the padding. None of the three can duplicate or drop text, which is
        // what the three booleans this replaced could not guarantee.
        const already = this.forwarded.get(contentIndex) ?? 0;
        if (block.text.length <= already) return [];
        const delta = block.text.slice(already);
        this.forwarded.set(contentIndex, block.text.length);
        return [{ type: "message_update", messageId: this.messageId, delta }];
      }

      // Reasoning. Forwarded on `model_thinking`, buffered and ticked, and
      // never on `message_update` — which is the line that matters, because
      // `message_update` is what becomes the answer, what is persisted, and
      // what the verifier resolves citations against. A thought that reached
      // that buffer would be signed as part of the deliverable.
      if (inner.type === "thinking_start") {
        return this.thinking("");
      }
      if (inner.type === "thinking_delta") {
        const delta = (inner as { delta?: unknown }).delta;
        return this.thinking(typeof delta === "string" ? delta : "");
      }
      if (inner.type === "thinking_end") {
        return this.endThinking();
      }

      // toolcall_* is a wire-format repair artefact, not visible prose, so it
      // is not forwarded either. The audit record holds the model-side view
      // under access control; the chat surface only ever sees the text.
      return [];
    }

    if (event.type === "message_end") {
      // A user message ending is not the answer ending, and a tool result
      // arriving is not the answer ending either. Only an assistant turn can
      // close the cell -- and only one that is not handing off to a tool.
      if (!isAssistantMessage(event.message)) return [];
      const message = event.message;
      // A model that reasoned and then stopped without answering still has an
      // open thinking block. Closing it here is what stops the surface
      // spinning on a run that is already over.
      const closed = this.endThinking();
      if (message.stopReason === "toolUse") {
        // A tool-use transition, not an outcome. The loop will run the tools
        // and come back with another assistant turn into the same cell;
        // terminating here truncated every tool-using run at its first call.
        return closed;
      }
      return [...closed, ...this.closeCell(message.stopReason, message.usage)];
    }

    // The loop is done. This is the backstop that guarantees a terminal event
    // for a run whose last assistant turn ended on `toolUse` -- a run stopped
    // by its step budget, its deadline, or an operator -- where no assistant
    // `message_end` will ever carry a final outcome.
    if (event.type === "agent_end") {
      return this.finalize(event.messages);
    }

    return [];
  }
}

/**
 * Whether a loop message is the assistant's own.
 *
 * `AgentMessage` spans user, tool-result and several harness-internal roles
 * (`custom`, `compactionSummary`, `branchSummary`, `bashExecution`). Only the
 * assistant's turns belong in the chat cell; the rest are the machinery around
 * them and are not the model speaking.
 */
function isAssistantMessage(
  message: unknown,
): message is { role: "assistant"; stopReason?: unknown; usage?: { input?: number; output?: number } } {
  return (
    typeof message === "object" &&
    message !== null &&
    (message as { role?: unknown }).role === "assistant"
  );
}

/**
 * How often a reasoning pass reports its size when it has no text to send.
 *
 * A second is slower than the model produces and faster than a person reads,
 * which is the whole requirement for a counter. Lower would spend frames on a
 * number nobody looked at; higher would make a long pause look like a stall
 * again.
 */
const THINKING_TICK_MS = 1000;

/**
 * How often buffered reasoning text is flushed.
 *
 * Twelve frames a second: fast enough that the panel reads as typing rather
 * than as paragraphs appearing, slow enough that a model decoding at three
 * hundred tokens a second costs twelve stdio frames rather than three
 * hundred. A reading-speed choice, not a throughput one — the text arrives
 * whole either way.
 */
const THINKING_TEXT_TICK_MS = 80;

/**
 * Backwards-compatible stateless wrapper. Used by the unit tests; the
 * production path uses {@link MessageTranslator} so a run cannot double-emit.
 */
export function translateForWire(
  event: AgentEvent,
  runId: string,
  messageId: string,
): WireEvent[] {
  void runId;
  const translator = new MessageTranslator(messageId);
  return translator.translate(event);
}

/**
 * Maps OpenClaw's `StopReason` to the chat surface's `finishReason` union.
 *
 * OpenClaw uses `"stop" | "length" | "toolUse" | "error" | "aborted"`. The
 * chat surface accepts `"stop" | "length" | "tool_calls" | "content_filter" |
 * "error"`. We collapse the variants an operator never needs to distinguish
 * into the closest equivalent; a future revision that needs the distinction
 * can extend the chat union.
 */
function mapStopReason(
  reason: unknown,
): "stop" | "length" | "tool_calls" | "content_filter" | "error" {
  if (reason === "length") return "length";
  if (reason === "toolUse") return "tool_calls";
  if (reason === "aborted" || reason === "error") return "error";
  // Default: any unknown future value lands in the safe bucket.
  return "stop";
}
