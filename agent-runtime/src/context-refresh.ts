/**
 * The boundary every model round passes through, on this side of the wire.
 *
 * ## Why every round and not once at the start
 *
 * Run-start injection was what this product did, and the way it failed was
 * quiet. A turn that makes twelve tool calls used the context compiled before
 * the first one: an operator correction recorded at call three reached the
 * model at call four only if the model happened to re-read it, and a fact
 * another agent committed to the same task never arrived at all.
 *
 * ## What a round asks of Rust, and in what order
 *
 * 1. **`context.refresh`, before compaction.** Rust takes the one GPU lease for
 *    this run, re-binds the model endpoint — the server may have been stopped
 *    to make room for a child and started again on another port, with another
 *    window — and compiles the four-scope context against the window the
 *    server actually has. It runs before compaction because compaction itself
 *    may call the model to summarise, and that call must hold the card too.
 * 2. **The re-bound endpoint.** The round is sent to the base URL Rust just
 *    bound, never the one the run was started with; the compactor is told too.
 * 3. **`context.count`, with the exact outgoing request.** Hooked into the
 *    provider's `onPayload`, so what is counted is the body that is about to
 *    be sent — rendered by the served model's own template and counted by its
 *    own tokenizer, images accounted separately. A request that does not fit is
 *    refused before it is sent.
 * 4. **`context.settle`, when the stream ends.** The card goes back, and the
 *    count meets the server's own usage report.
 *
 * ## What fails the run, and what only degrades it
 *
 * Rust answers a round that must not go ahead with `round_refused`: no GPU
 * lease, a model that would not load, mandatory state or a rendered request
 * that does not fit. That is thrown, the run fails with Rust's own sentence,
 * and the model is not called — a round that went ahead anyway would be the
 * over-budget request, or the second heavy call on an 8 GB card, that the
 * refusal exists to prevent.
 *
 * Anything else — a Rust side too old to know the method, a transport hiccup —
 * degrades: the round runs on what it already had, and says so.
 */

import { RpcError, type RpcPeer } from "./peer.js";

/** One block of context Rust authorised for this round. */
export interface ContextBlock {
  kind:
    | "objective"
    | "constraint"
    | "plan"
    | "receipt"
    | "pendingApproval"
    | "neighbour"
    | "evidence"
    | "artifact"
    | "procedure"
    | "preference"
    | "digest";
  itemId?: string;
  revision?: number;
  content: string;
  tokens: number;
  contentHash: string;
}

/** The endpoint Rust bound this round to. */
export interface RoundEndpoint {
  baseUrl: string;
  /** The id the server answers to. */
  modelId: string;
  /** The window the server was actually started with. */
  servedWindow: number;
  /** `server` when the server said so, `registryDeclared` when it did not. */
  windowSource: string;
  warm: boolean;
  /** A different server process from the last round: its prompt cache is gone. */
  restarted: boolean;
  /** `sameServer` or `cold`. Nothing is ever carried across a new process. */
  cache: string;
}

/** What Rust answered. */
export interface ContextRefresh {
  /** The changefeed cursor the rows were actually read at. */
  graphRevision: number | null;
  manifestHash: string;
  /**
   * Always false on an answer: a round whose mandatory state does not fit is
   * refused with `round_refused` instead. Kept for readers of older records.
   */
  mandatoryOverflowed: boolean;
  blocks: ContextBlock[];
  contentHashes: string[];
  omissions: Array<{ what: string; reason: string; detail: string }>;
  graph?: { available: boolean; because?: string };
  /** Absent for a run nobody bound to a model; see `lease`. */
  endpoint?: RoundEndpoint | null;
  lease?: { status: string; because?: string; waitedMs?: number };
}

export interface RefreshRequest {
  runId: string;
  taskId: string;
  agentId: string;
  definitionVersion: number;
  modelId: string;
  servedWindow: number;
  question: string;
  /** Content hashes this round already carries, so nothing is injected twice. */
  alreadyCarried: string[];
  projectId?: string;
  templateId?: string;
  /** What this agent does, so an activated procedure reaches only its role. */
  capability?: string;
  reservedToolSchemas: number;
  reservedOutput: number;
  reservedFraming: number;
  /** Held back for what the per-block estimates may miss. */
  reservedSafety: number;
}

/** The wire code Rust uses for a round that must not go ahead. */
export const ROUND_REFUSED = "round_refused";

/** A round Rust refused. Thrown, so the run fails with Rust's sentence. */
export class RoundRefused extends Error {
  constructor(message: string) {
    super(message);
    this.name = "RoundRefused";
  }
}

function describe(cause: unknown): string {
  if (cause instanceof RpcError) return `${cause.code}: ${cause.message}`;
  if (cause instanceof Error) return cause.message;
  return String(cause);
}

const stderrLog = (line: string) => {
  process.stderr.write(`[agent-runtime:log] ${line}\n`);
};

/**
 * Asks for the context this round may use.
 *
 * Returns `undefined` when the refresh could not be made for a reason that
 * only degrades the round; throws {@link RoundRefused} when Rust refused it.
 */
export async function refreshContext(
  peer: RpcPeer,
  request: RefreshRequest,
  log: (line: string) => void = stderrLog,
): Promise<ContextRefresh | undefined> {
  try {
    const refreshed = (await peer.request("context.refresh", request)) as ContextRefresh;
    for (const omission of refreshed?.omissions ?? []) {
      // Only the reasons a person can act on. A deduplication is the compiler
      // working and would be noise on every round.
      if (omission.reason === "budget" || omission.reason === "revoked") {
        log(`[context] run=${request.runId} ${omission.detail}`);
      }
    }
    if (refreshed?.graph && !refreshed.graph.available) {
      log(`[context] run=${request.runId} ${refreshed.graph.because ?? "no task memory was compiled"}`);
    }
    if (refreshed?.lease?.status === "unbound") {
      log(`[context] run=${request.runId} ${refreshed.lease.because ?? "no GPU lease was taken"}`);
    }
    return refreshed;
  } catch (cause) {
    if (cause instanceof RpcError && cause.code === ROUND_REFUSED) {
      log(`[context] run=${request.runId} round refused: ${cause.message}`);
      throw new RoundRefused(cause.message);
    }
    log(
      `[context] run=${request.runId} this round could not refresh its context (${describe(cause)}), ` +
        `so it runs on what it already had — a correction or a fact recorded since the last ` +
        `refresh will not have reached the model`,
    );
    return undefined;
  }
}

/**
 * Renders authorised blocks into the one message the round prepends.
 *
 * One message rather than several, because the transcript's shape is what the
 * compactor and the translator reason about. Every block carries its own
 * status label — and a block from outside the task says what it yields to — so
 * a stored sentence that tries to give instructions arrives as something with a
 * provenance rather than as a voice.
 */
export function renderContextBlocks(blocks: readonly ContextBlock[]): string | undefined {
  if (blocks.length === 0) return undefined;
  const scoped = blocks.some(
    (block) => block.kind === "procedure" || block.kind === "preference",
  );
  const header = scoped
    ? "--- TASK MEMORY (authorised for this turn; each line states its own status. Where two " +
      "lines disagree, this task's constraints and corrections win, then project rules, then " +
      "activated procedures, then preferences) ---\n"
    : "--- TASK MEMORY (authorised for this turn; each line states its own status) ---\n";
  return header + blocks.map((block) => block.content).join("\n");
}

type StreamArgs = [model: unknown, context: unknown, options?: unknown];

/** The subset of a provider payload hook this side calls. */
type PayloadHook = (payload: unknown, model: unknown) => unknown;

/** What a round boundary needs from the run. */
export interface RoundBoundaryOptions {
  peer: RpcPeer;
  /** The request for this round, read at the moment it opens. */
  request: () => RefreshRequest;
  /** Told the content hashes a round carried, so none is injected twice. */
  remember: (hashes: readonly string[]) => void;
  /** Told when the endpoint changes, so the compactor summarises against it. */
  onRebind?: (endpoint: RoundEndpoint) => void;
  log?: (line: string) => void;
}

/**
 * One run's rounds: opened before compaction, counted on the wire, settled at
 * the end of the stream.
 */
export class RoundBoundary {
  readonly #options: RoundBoundaryOptions;
  readonly #log: (line: string) => void;
  #latest: ContextRefresh | undefined;
  #endpoint: RoundEndpoint | undefined;
  /** Whether a round has been opened and not yet sent. */
  #open = false;
  #calls = 0;

  constructor(options: RoundBoundaryOptions) {
    this.#options = options;
    this.#log = options.log ?? stderrLog;
  }

  /** The endpoint of the latest round, when Rust bound one. */
  get endpoint(): RoundEndpoint | undefined {
    return this.#endpoint;
  }

  /** How many model calls this run has made through the boundary. */
  get calls(): number {
    return this.#calls;
  }

  /**
   * Opens a round: the lease, the endpoint, the context. Throws
   * {@link RoundRefused} when Rust refused it.
   */
  async open(): Promise<ContextRefresh | undefined> {
    const request = this.#options.request();
    const refreshed = await refreshContext(this.#options.peer, request, this.#log);
    this.#latest = refreshed;
    this.#open = true;
    const endpoint = refreshed?.endpoint ?? undefined;
    if (endpoint) {
      const changed =
        !this.#endpoint ||
        this.#endpoint.baseUrl !== endpoint.baseUrl ||
        this.#endpoint.servedWindow !== endpoint.servedWindow;
      this.#endpoint = endpoint;
      if (changed) {
        if (endpoint.restarted) {
          this.#log(
            `[context] run=${request.runId} the model's server was started again ` +
              `(${endpoint.baseUrl}, ${endpoint.servedWindow}-token window); its cache is cold ` +
              `and nothing was carried across`,
          );
        }
        this.#options.onRebind?.(endpoint);
      }
    }
    return refreshed;
  }

  /** Wraps `transformContext` so the round is opened before compaction runs. */
  transform<M>(
    inner: (messages: M[], signal?: AbortSignal) => Promise<M[]>,
  ): (messages: M[], signal?: AbortSignal) => Promise<M[]> {
    return async (messages, signal) => {
      await this.open();
      return inner(messages, signal);
    };
  }

  /**
   * Wraps a stream function: the round's blocks prepended, the re-bound
   * endpoint used, the exact payload counted, and the round settled when the
   * stream ends.
   */
  stream<Fn extends (...args: never[]) => unknown>(inner: Fn): Fn {
    const wrapped = async (...raw: unknown[]): Promise<unknown> => {
      // A call nothing opened a round for — the loop calling the stream without
      // its transform, say — opens one here rather than going out unleased.
      if (!this.#open) await this.open();
      this.#open = false;
      const callIndex = this.#calls;
      this.#calls += 1;

      const args = [...raw] as StreamArgs;
      const request = this.#options.request();
      const refreshed = this.#latest;

      if (refreshed && refreshed.blocks.length > 0) {
        const rendered = renderContextBlocks(refreshed.blocks);
        // Found by shape rather than by position: `streamFn` has been called
        // with different arities by different versions of agent-core.
        const carrier = args.find(
          (arg): arg is { messages: unknown[] } =>
            typeof arg === "object" &&
            arg !== null &&
            Array.isArray((arg as { messages?: unknown }).messages),
        );
        if (rendered && carrier) {
          // Prepended, so it is read before the conversation rather than after
          // it. Not written into the transcript: the transcript is what was
          // said, and this is what Rust authorised for this round.
          carrier.messages.unshift({ role: "system", content: rendered });
        }
        this.#options.remember(refreshed.contentHashes);
      }

      // The endpoint Rust bound for this round, never the one the run started
      // with. A server stopped for a child and started again is on a new port.
      const endpoint = this.#endpoint;
      if (endpoint && typeof args[0] === "object" && args[0] !== null) {
        args[0] = {
          ...(args[0] as object),
          baseUrl: endpoint.baseUrl,
          contextWindow: endpoint.servedWindow,
        };
      }

      // The exact outgoing body, counted by the served model's own template
      // and tokenizer before it is sent. Chained after the existing hook, so
      // what is counted is what goes out.
      const options = (args[2] ?? {}) as { onPayload?: PayloadHook };
      const original = options.onPayload;
      args[2] = {
        ...options,
        onPayload: async (payload: unknown, model: unknown) => {
          const patched = (original ? await original(payload, model) : undefined) ?? payload;
          await this.#count(request.runId, patched, callIndex);
          return patched;
        },
      };

      let response: unknown;
      try {
        response = await (inner as unknown as (...rest: unknown[]) => unknown)(...args);
      } catch (cause) {
        this.#settle(request.runId, undefined);
        throw cause;
      }
      const result = (response as { result?: () => Promise<unknown> } | undefined)?.result;
      if (typeof result === "function") {
        result
          .call(response)
          .then((message) => this.#settle(request.runId, message))
          .catch(() => this.#settle(request.runId, undefined));
      } else {
        this.#settle(request.runId, undefined);
      }
      return response;
    };
    return wrapped as unknown as Fn;
  }

  async #count(runId: string, payload: unknown, callIndex: number): Promise<void> {
    try {
      const counted = (await this.#options.peer.request("context.count", {
        runId,
        payload,
        callIndex,
      })) as { countedBy?: string; inputTokens?: number; fit?: { fit?: string } };
      if (counted?.fit?.fit === "uncounted") {
        this.#log(
          `[context] run=${runId} the request could not be counted by the model's own tokenizer; ` +
            `it went out on the compactor's estimate`,
        );
      }
    } catch (cause) {
      if (cause instanceof RpcError && cause.code === ROUND_REFUSED) {
        throw new RoundRefused(cause.message);
      }
      this.#log(`[context] run=${runId} the request was not counted (${describe(cause)})`);
    }
  }

  #settle(runId: string, message: unknown): void {
    const usage = (message as { usage?: { input?: number; output?: number } } | undefined)?.usage;
    const stopReason = (message as { stopReason?: string } | undefined)?.stopReason;
    this.#options.peer
      .request("context.settle", {
        runId,
        inputTokens: usage?.input && usage.input > 0 ? usage.input : undefined,
        outputTokens: usage?.output && usage.output > 0 ? usage.output : undefined,
        stopReason,
      })
      .catch((cause: unknown) => {
        this.#log(`[context] run=${runId} the round was not settled (${describe(cause)})`);
      });
  }
}
