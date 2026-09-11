/**
 * Tools the model may request, and the single point at which that request is
 * authorised.
 *
 * ## Where the boundary is
 *
 * Nothing in this file decides anything. Every tool here is a stub that forwards
 * to the Rust core, and every call is put through `authorizeToolCall` first,
 * which asks the core for a verdict from `orchestrator::gateway::ToolGateway`.
 * The model can request; only Rust decides. That split is the product's central
 * claim, so it is worth being precise about how it is held:
 *
 * 1. **Authorisation is a loop hook, not a tool method.** It runs in
 *    `beforeToolCall`, which agent-core applies to every call uniformly. A tool
 *    added in a later phase cannot forget to authorise itself, because it is not
 *    the tool's job.
 *
 * 2. **A verdict is a single-use grant, not a boolean.** Rust replies to an
 *    allow with an opaque token bound to that exact call and consumes it on
 *    execution. So this side cannot cache a verdict, replay one, or authorise
 *    cheap arguments and execute expensive ones -- not because it is careful,
 *    but because the token would not match. The check is structural.
 *
 * 3. **Rust re-checks anyway.** `tool.execute` validates independently of the
 *    grant. Two independent refusals beat one, and the grant protects against a
 *    compromised runtime while the re-check protects against a bug in the grant.
 */

import type { AgentTool, BeforeToolCallContext, BeforeToolCallResult } from "@openclaw/agent-core";
import { ErrorCode, type ErrorCodeValue } from "./protocol.js";
import { definitionFor, TOOL_DEFINITIONS, type ToolDefinition } from "./catalogue.js";
import { RpcError, type RpcPeer } from "./peer.js";

/** What the Rust gateway replies to `tool.authorize`. Mirrors `GatewayVerdict`. */
export type Verdict =
  | { outcome: "allow"; tool: string; grant: string; resolvedPath?: string | null }
  | { outcome: "needsApproval"; tool: string; summary: string; resolvedPath?: string | null }
  | {
      outcome: "refuse";
      reason: string;
      /**
       * The plan has halted and every later call will be refused the same way.
       *
       * Set by the core when the budget is spent, the run was caught looping,
       * or its time is up. Without it a refusal was an ordinary error tool
       * result the model was free to retry — and every retry emitted events,
       * which rearmed the four-minute stall guard, so a run with nothing left
       * to spend ran to its thirty-minute deadline producing refusals.
       */
      terminal?: boolean;
    };

/** A single image attached to a tool result.

The agent-core API accepts image content parts in the same way
OpenAI does — the runtime translates the in-memory representation
into the chat schema the model is currently speaking. Keeping the
field as a base64 data URI rather than a path means the
representation is portable: it round-trips through any serialiser
the runtime speaks, and the bytes are not silently fetched from a
URL a model could not have verified.
*/
export interface ToolResultImage {
  /** MIME type, e.g. "image/png". */
  mime: string;
  /** Base64-encoded image bytes. */
  data: string;
  /** Optional caption: a textual proxy the model can cite. */
  caption?: string;
  /** Optional bounding box, as fractions of the source page (0.0–1.0). */
  bbox?: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
}

/** A single table attached to a tool result.

Preserved as structure, never flattened. The model sees the columns
and rows in the format its input layer accepts, and a downstream
verifier can check that a citation the model wrote actually exists
in the table — which is what a table that arrived as prose cannot
do.
*/
export interface ToolResultTable {
  /** Column headers, in order. Same length as every row. */
  headers: string[];
  /** Rows, each the same length as `headers`. */
  rows: string[][];
  /** Where on the page the table sits, as fractions (0.0–1.0). */
  bbox?: {
    left: number;
    top: number;
    right: number;
    bottom: number;
  };
  /** Citation the model can read. The page number at minimum. */
  citation: string;
}

/** What Rust returns from `tool.execute`. */
export interface ToolExecution {
  /** What the model sees. Always present — multimodal results still
   *  carry a prose rendering so a model that does not understand
   *  the structured fields gets a useful answer anyway. */
  text: string;
  /** Optional images attached to the result. The agent-core content
   *  array carries them as image parts alongside the text. */
  images?: ToolResultImage[];
  /** Optional tables attached to the result. The model sees the
   *  prose rendering in `text` and the structured form here. */
  tables?: ToolResultTable[];
  /** Structured detail for the audit record and the UI. Never shown to the model. */
  details?: unknown;
}

/**
 * Grants held between authorisation and execution, keyed by tool-call id.
 *
 * Scoped per run and cleared when it ends, so a grant cannot outlive the run
 * that earned it even if Rust's own expiry were to fail.
 */
export class GrantLedger {
  readonly #grants = new Map<string, string>();

  put(toolCallId: string, grant: string): void {
    this.#grants.set(toolCallId, grant);
  }

  /** Reads and removes. A grant is good for exactly one execution. */
  take(toolCallId: string): string | undefined {
    const grant = this.#grants.get(toolCallId);
    this.#grants.delete(toolCallId);
    return grant;
  }

  clear(): void {
    this.#grants.clear();
  }

  get size(): number {
    return this.#grants.size;
  }
}

/**
 * How long the gateway gets to answer whether a call may proceed.
 *
 * ## Why this is measured in minutes and not seconds
 *
 * It was thirty seconds, on the reasoning that deciding is a policy lookup and
 * therefore quick. That holds right up until the answer is "a person must
 * approve this", and then the Rust side waits on that person for up to
 * `approval::WAIT_LIMIT`, which is fifteen minutes.
 *
 * The two numbers did not agree, and the gap was not cosmetic. Every approval
 * a reviewer took longer than half a minute over ran like this: the model was
 * told the tool was unavailable and moved on, the person pressed Approve and
 * nothing happened, Rust issued a grant nothing ever redeemed, and the plan
 * lease it held sat orphaned for `LEASE_TTL`. The step showed as unfinished,
 * so `completion::verify` reported a run that did exactly what was asked as
 * failed.
 *
 * A minute past the Rust limit, so the Rust side is always the one that decides
 * a reviewer has not answered — it is the side that knows what it is waiting
 * for and can say so in words the model can act on. This side timing out first
 * can only produce "authorisation is unavailable", which is untrue: it was
 * available, and somebody was thinking about it.
 *
 * Safe only because the core no longer blocks its reader while it waits — see
 * the note on `Frame::Request` in `agent_runtime::mod`. While that was serial,
 * a long authorise deafened the whole channel, including to Stop.
 */
const AUTHORIZE_TIMEOUT_MS = 16 * 60 * 1000;

/**
 * Asks Rust whether a call may proceed, and records the grant if it may.
 *
 * Returns a `BeforeToolCallResult` for agent-core: `{ block: true, reason }`
 * turns into an error tool result the model reads and can recover from, which is
 * the behaviour we want -- a refusal is information, not a crash.
 */
export async function authorizeToolCall(
  peer: RpcPeer,
  ledger: GrantLedger,
  runId: string,
  context: BeforeToolCallContext,
  signal?: AbortSignal,
  /**
   * Called when the core reports that the plan has halted for good.
   *
   * The refusal is still returned as a blocked tool call, so the model is told
   * why in words it can report — but the run also stops, rather than retrying
   * into the same wall until the deadline.
   */
  onPlanHalted?: (reason: string) => void,
): Promise<BeforeToolCallResult | undefined> {
  const { toolCall, args } = context;

  // Checked before the request rather than only after it. A run stopped while
  // the previous step was running would otherwise open one more authorisation
  // — and if that one is the call that waits on a person, the turn the user
  // just cancelled goes on to hold the gateway for as long as the approval
  // allows.
  if (signal?.aborted) {
    return {
      block: true,
      reason: "The task was stopped before this call was authorised, so it did not run.",
    };
  }

  let verdict: Verdict;
  try {
    verdict = (await withTimeout(
      peer.request("tool.authorize", {
        runId,
        toolCallId: toolCall.id,
        tool: toolCall.name,
        args,
      }),
      AUTHORIZE_TIMEOUT_MS,
      `authorize:${toolCall.name}`,
    )) as Verdict;
  } catch (error) {
    // A gateway that cannot be reached is a gateway that did not say yes.
    // Failing closed is the only safe reading of silence here.
    const message = error instanceof Error ? error.message : String(error);
    return { block: true, reason: `Tool authorisation is unavailable, so the call was not made: ${message}` };
  }

  switch (verdict.outcome) {
    case "allow":
      ledger.put(toolCall.id, verdict.grant);
      return undefined;
    case "needsApproval":
      // Phase 1 ships only tools the gateway marks `needs_approval: false`, so
      // this is unreachable today. It blocks rather than assuming consent
      // because the wrong default here is the one that cannot be undone; the
      // approval queue is wired in Phase 4.
      return {
        block: true,
        reason: `${verdict.summary}\n\nThis action needs a person to approve it, and approval is not yet wired into this runtime.`,
      };
    case "refuse":
      if (verdict.terminal) onPlanHalted?.(verdict.reason);
      return { block: true, reason: verdict.reason };
  }
}

/** Builds one tool whose execution is performed by the Rust core. */
/**
 * The default ceiling for a tool that did not declare one.
 *
 * Twenty seconds rather than none. A catalogue entry without a timeout is a
 * defect, but the failure it produces should be a message a person can read,
 * not a run that never returns.
 */
const FALLBACK_TIMEOUT_MS = 20_000;

/**
 * Fails a call that does not come back.
 *
 * `ToolSpec::timeout` has existed on the Rust side all along and was carried
 * across the wire as `timeoutSeconds` - into a field nothing read. Every
 * generator was therefore unbounded: a subprocess that wedged, a headless step
 * that never exited, or a handler that blocked took the whole run with it, and
 * the person watching saw a spinner rather than an error.
 *
 * Bounded here rather than in each generator because this is the one place
 * every tool call passes through. A ceiling added per-generator is a ceiling
 * somebody forgets on the next one.
 *
 * The timer is cleared on both paths: leaving it pending would hold the process
 * open for as long as the longest timeout in the catalogue.
 */
/**
 * Races `work` against its timeout, and against the run being stopped.
 *
 * The abort arm is what makes Stop mean *stop*. Without it the only way out of
 * an in-flight tool call was its own ceiling — up to two minutes for a document
 * — during which the loop could not proceed and the person watched a turn they
 * had already cancelled.
 *
 * What this does **not** do is cancel the work on the other side. `Promise.race`
 * abandons the loser; it does not reach into Rust and stop the tool. That is
 * deliberate and is why the refusal below says the call "may already have
 * happened": for a side-effecting tool the file may genuinely be written, and
 * telling the model it definitely was not would be the more dangerous lie.
 */
async function withTimeout<T>(
  work: Promise<T>,
  milliseconds: number,
  tool: string,
  signal?: AbortSignal,
): Promise<T> {
  let timer: ReturnType<typeof setTimeout> | undefined;
  let onAbort: (() => void) | undefined;
  try {
    return await Promise.race([
      work,
      new Promise<never>((_resolve, reject) => {
        if (!signal) return;
        onAbort = () =>
          reject(
            new RpcError(
              ErrorCode.Refused,
              `The task was stopped while ${tool} was running. It may already have happened, ` +
                `so do not assume it did or did not — say that it was interrupted.`,
            ),
          );
        if (signal.aborted) onAbort();
        else signal.addEventListener("abort", onAbort, { once: true });
      }),
      new Promise<never>((_resolve, reject) => {
        timer = setTimeout(() => {
          reject(
            new RpcError(
              ErrorCode.ToolFailed,
              `${tool} did not finish within ${Math.round(milliseconds / 1000)}s and was ` +
                `stopped. Nothing it may have produced can be relied on. Say so rather than ` +
                `describing what it would have returned.`,
            ),
          );
        }, milliseconds);
      }),
    ]);
  } finally {
    if (timer) clearTimeout(timer);
    if (signal && onAbort) signal.removeEventListener("abort", onAbort);
  }
}

function hostTool(options: {
  name: string;
  label: string;
  description: string;
  parameters: ToolDefinition["parameters"];
  peer: RpcPeer;
  ledger: GrantLedger;
  runId: string;
  modelId: string;
  /**
   * Whether this tool may run alongside others in the same turn.
   *
   * Read-only tools are parallel: several searches at once cost the operator
   * the slowest rather than the sum, and one search cannot affect what another
   * returns. Anything that writes, produces a file or runs code is sequential —
   * two writes to the same path in one turn have an order, and it should not be
   * whichever finished first.
   */
  executionMode: "parallel" | "sequential";
  /** Wall-clock ceiling for one call, from the catalogue entry. */
  timeoutMs: number;
  /**
   * Told what each call produced, so the run's notes can be kept current.
   *
   * Called with the text the *model* is about to read, not with the structured
   * detail beside it. That is deliberate: the notes exist to record what the
   * model was told, and a marker the model never saw is one it cannot cite.
   */
  observe?: (observation: { tool: string; args: unknown; text: string }) => void;
}): AgentTool {
  const {
    name,
    label,
    description,
    timeoutMs,
    parameters,
    peer,
    ledger,
    runId,
    modelId,
    executionMode,
    observe,
  } = options;
  return {
    name,
    label,
    description,
    parameters,
    executionMode,
    async execute(toolCallId, params, signal) {
      // `signal` is the third argument agent-core has always passed and this
      // function has always ignored. Ignoring it meant `run.abort` — the
      // operator's Stop button — could not touch a call that was already in
      // flight: the loop sat in `Promise.all` over the launched calls until each
      // one's own timeout expired, which for `create_docx` is two minutes.
      if (signal?.aborted) {
        throw new RpcError(
          ErrorCode.Refused,
          `The task was stopped before ${name} ran, so it did not run.`,
        );
      }
      const grant = ledger.take(toolCallId);
      if (!grant) {
        // Reached only if the loop skipped `beforeToolCall` or a grant was
        // consumed twice. Either is a defect in this runtime, and the honest
        // response is to refuse and say so rather than try the call anyway.
        throw new RpcError(
          ErrorCode.Refused,
          `No authorisation grant for ${name}. The call was not put through the gateway.`,
        );
      }
      const execution = (await withTimeout(
        peer.request("tool.execute", {
        runId,
        toolCallId,
        tool: name,
        args: params,
        grant,
        // Stamped onto anything this call produces, so a reader of the
        // document knows which model wrote it.
        model: modelId,
        }),
        timeoutMs,
        name,
        signal,
      )) as ToolExecution;
      // After the call has actually succeeded. Recording an effect before the
      // gateway and the tool have both agreed to it would tell a resumed run
      // not to repeat something that never happened.
      //
      // Best-effort: a note that could not be taken costs the next attempt some
      // context, and throwing here would cost this attempt the tool result it
      // has already paid for.
      try {
        observe?.({ tool: name, args: params, text: execution.text });
      } catch {
        // Deliberately swallowed. See above.
      }

      // Build the content array the agent-core API expects. Text first
      // so a model that only reads `content[0].text` (a behaviour some
      // older agent-core versions fall back to) still gets the answer;
      // then the structured multimodal parts in declaration order, so
      // a model that walks the array sees them grouped with their
      // textual context.
      // A table is rendered into the text, not emitted as a block of its own.
      //
      // `{ type: "table", headers, rows, citation }` was a shape nothing
      // anywhere understands. The transport has no table type, so the provider
      // emitters skipped it; `getCompactionContentBlockText` returns `""` for
      // it, so it estimated at nothing; and the chat surface renders tables
      // from markdown in the text, so it never reached a person either. The
      // rows were being dropped in three places at once, silently.
      //
      // Markdown, because that is what the model reads and what `Markdown.tsx`
      // already renders. The citation rides with it so a figure taken from the
      // table can still be traced.
      const renderedTables = (execution.tables ?? []).map((table) => {
        const header = `| ${table.headers.join(" | ")} |`;
        const rule = `| ${table.headers.map(() => "---").join(" | ")} |`;
        const body = table.rows.map((row) => `| ${row.join(" | ")} |`).join("\n");
        return `${table.citation}\n${header}\n${rule}\n${body}`;
      });

      const text = renderedTables.length > 0
        ? [execution.text, ...renderedTables].join("\n\n")
        : execution.text;

      const content: Array<
        | { type: "text"; text: string }
        | {
            /**
             * The shape the transport actually reads.
             *
             * This was `{ source: { type: "base64", media_type, data } }` — the
             * Anthropic block shape — and `llm-core`'s `ImageContent` wants
             * `data` and `mimeType` at the top level. `isImageWithMediaPayload`
             * checks `block.data`, so every image failed it: the provider
             * emitter skipped it, and `describeToolResultMediaPlaceholder`
             * failed the same check, so the model was not even told an image
             * existed. Meanwhile the compaction estimator matches on
             * `block.type === "image"` alone and charged 2,000 tokens for each
             * one — a page of a scanned drawing cost a window it never reached.
             */
            type: "image";
            data: string;
            mimeType: string;
            // OpenAI-compatible vision input — some models accept a
            // caption as a marker. The runtime's translator is free to
            // ignore it on runtimes that do not.
            caption?: string;
          }
      > = [{ type: "text", text }];

      for (const image of execution.images ?? []) {
        content.push({
          type: "image",
          data: image.data,
          mimeType: image.mime,
          ...(image.caption !== undefined ? { caption: image.caption } : {}),
        });
      }

      return {
        content,
        details: execution.details ?? null,
      };
    },
  } as AgentTool;
}

/**
 * One tool's eligibility, as Rust reported it.
 *
 * Metadata only — no parameter schema. That is the whole point of asking: the
 * schemas are the second largest fixed thing in the context window after the
 * system prompt, and loading one for a tool this run may not call spends window
 * on a definition whose only use is to have the gateway refuse it.
 */
export interface EligibleTool {
  name: string;
  summary: string;
  /** Whether the call only reads. Decides whether it may run beside another. */
  readOnly: boolean;
  approvalClass: string;
  network: string;
  maxResponseBytes: number;
  timeoutSeconds: number;
}

/** What `tool.catalogue` answers. */
export interface Catalogue {
  tools: EligibleTool[];
  mode: string;
  /**
   * Why the catalogue could not be read, when it could not be.
   *
   * Set only on the failure path. An empty `tools` means two different things
   * — "the plan permits none" and "nobody could be asked" — and they were
   * indistinguishable, so a transport fault produced a run that answered from
   * the model's weights with no tools and nothing on screen to say why.
   */
  unavailable?: string;
}

/**
 * Asks Rust which tools this run may be offered.
 *
 * ## Why this side does not decide
 *
 * Eligibility depends on the run's plan and the machine's operating mode,
 * neither of which is on this side of the wire. Deciding here would mean
 * re-deriving them from something the child process can see, and the child
 * process is the part of the system that is deliberately not trusted with that.
 *
 * ## Why a failure means no tools rather than all of them
 *
 * Failing closed. A gateway that cannot be reached has not said which tools are
 * eligible, and reading silence as "all of them" would mean a transport fault
 * widening the surface a model can reach — the one direction a fault must never
 * move things in. A run with no tools can still answer from what it was told,
 * and says plainly that it could not use any.
 */
/**
 * How long the gateway has to answer with the catalogue.
 *
 * This call had no bound at all, and it happens *before* the run is
 * registered, before the deadline timer and before the stall guard — so a
 * gateway that accepted the frame and never answered left a turn that could not
 * be stopped by anything. `run.abort` found no active run and replied
 * `{aborted: false}`; the core's own timeout fired, aborted nothing, and the
 * child sat on the pending request until the pipe closed.
 *
 * Generous, because the gateway is doing real work — reading the plan, the
 * entitlements and the skills — and short enough that a wedged one is a failed
 * run rather than a hung one.
 */
const CATALOGUE_TIMEOUT_MS = 30_000;

export async function fetchCatalogue(peer: RpcPeer, runId: string): Promise<Catalogue> {
  try {
    const answer = (await withTimeout(
      peer.request("tool.catalogue", { runId }),
      CATALOGUE_TIMEOUT_MS,
      "tool.catalogue",
    )) as Catalogue;
    return {
      tools: Array.isArray(answer?.tools) ? answer.tools : [],
      mode: typeof answer?.mode === "string" ? answer.mode : "unknown",
    };
  } catch (error) {
    // Still failing closed — see the note above. What changed is that the
    // reason travels with the empty list instead of being swallowed, so the
    // run can say it had no tools *because nobody answered*, rather than
    // looking like a run whose plan permitted none.
    return {
      tools: [],
      mode: "unknown",
      unavailable: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Builds the tools this run may actually use.
 *
 * ## Deferred loading, in two steps
 *
 * `eligible` is the metadata Rust returned. Only names in it get their schema
 * loaded and handed to the model. A name Rust offered that this runtime has no
 * definition for is skipped rather than guessed at — the two lists are kept in
 * agreement by a test, and a runtime inventing a schema for a name it does not
 * know would be inventing an interface to a tool it cannot call.
 *
 * Passing `undefined` builds the whole catalogue. That is for the health probe,
 * which belongs to no run and has no plan to narrow against.
 *
 * ## Why execution mode is derived rather than declared
 *
 * A tool's `readOnly` flag decides it, in one place. Declaring the two
 * separately is how a tool ends up marked read-only and running sequentially,
 * or — much worse — marked as writing and running in parallel with a second
 * write to the same path. The order of two writes should not be whichever
 * finished first.
 *
 * Rust's answer wins where both have an opinion: it is the side that also
 * enforces the consequence.
 */
export function buildTools(
  peer: RpcPeer,
  ledger: GrantLedger,
  runId: string,
  modelId: string,
  observe?: (observation: { tool: string; args: unknown; text: string }) => void,
  eligible?: readonly EligibleTool[],
): AgentTool[] {
  type Entry = { definition: ToolDefinition; readOnly: boolean; timeoutMs: number };
  const definitions: Entry[] =
    eligible === undefined
      ? TOOL_DEFINITIONS.map((definition) => ({
          definition,
          readOnly: definition.readOnly,
          timeoutMs: FALLBACK_TIMEOUT_MS,
        }))
      : eligible
          .map((entry) => {
            const definition = definitionFor(entry.name);
            return definition
              ? {
                  definition,
                  readOnly: entry.readOnly,
                  // The catalogue's own figure. A missing or nonsensical one
                  // falls back rather than becoming an unbounded call.
                  timeoutMs:
                    Number.isFinite(entry.timeoutSeconds) && entry.timeoutSeconds > 0
                      ? entry.timeoutSeconds * 1000
                      : FALLBACK_TIMEOUT_MS,
                }
              : undefined;
          })
          .filter((entry): entry is Entry => entry !== undefined);

  return definitions.map(({ definition, readOnly, timeoutMs }) =>
    hostTool({
      peer,
      ledger,
      runId,
      modelId,
      observe,
      name: definition.name,
      label: definition.label,
      description: definition.description,
      parameters: definition.parameters,
      timeoutMs,
      // Reads may overlap: one cannot change what another returns, so several
      // at once cost the operator the slowest rather than the sum. Everything
      // that writes, produces a file, runs code or asks a person is serialised.
      executionMode: readOnly ? "parallel" : "sequential",
    }),
  );
}

export const toolErrorCode: ErrorCodeValue = ErrorCode.ToolFailed;
