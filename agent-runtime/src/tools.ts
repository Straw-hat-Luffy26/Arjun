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
type RequestPeer = Pick<RpcPeer, "request">;

/** What the Rust gateway replies to `tool.authorize`. Mirrors `GatewayVerdict`. */
export type Verdict =
  | { outcome: "allow"; tool: string; grant: string; resolvedPath?: string | null }
  | { outcome: "needsApproval"; tool: string; summary: string; resolvedPath?: string | null }
  | { outcome: "refuse"; reason: string };

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
 * Asks Rust whether a call may proceed, and records the grant if it may.
 *
 * Returns a `BeforeToolCallResult` for agent-core: `{ block: true, reason }`
 * turns into an error tool result the model reads and can recover from, which is
 * the behaviour we want -- a refusal is information, not a crash.
 */
export async function authorizeToolCall(
  peer: RequestPeer,
  ledger: GrantLedger,
  runId: string,
  context: BeforeToolCallContext,
): Promise<BeforeToolCallResult | undefined> {
  const { toolCall, args } = context;
  let verdict: Verdict;
  try {
    verdict = (await peer.request("tool.authorize", {
      runId,
      toolCallId: toolCall.id,
      tool: toolCall.name,
      args,
    })) as Verdict;
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
      return { block: true, reason: verdict.reason };
  }
}

/** Builds one tool whose execution is performed by the Rust core. */
function hostTool(options: {
  name: string;
  label: string;
  description: string;
  parameters: ToolDefinition["parameters"];
  peer: RequestPeer;
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
    async execute(toolCallId, params) {
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
      const execution = (await peer.request("tool.execute", {
        runId,
        toolCallId,
        tool: name,
        args: params,
        grant,
        // Stamped onto anything this call produces, so a reader of the
        // document knows which model wrote it.
        model: modelId,
      })) as ToolExecution;
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
      const content: Array<
        | { type: "text"; text: string }
        | {
            type: "image";
            source: { type: "base64"; media_type: string; data: string };
            // OpenAI-compatible vision input — some models accept a
            // caption as a marker. The runtime's translator is free to
            // ignore it on runtimes that do not.
            caption?: string;
          }
        | {
            type: "table";
            headers: string[];
            rows: string[][];
            citation: string;
          }
      > = [{ type: "text", text: execution.text }];

      for (const image of execution.images ?? []) {
        content.push({
          type: "image",
          source: { type: "base64", media_type: image.mime, data: image.data },
          ...(image.caption !== undefined ? { caption: image.caption } : {}),
        });
      }
      for (const table of execution.tables ?? []) {
        content.push({
          type: "table",
          headers: table.headers,
          rows: table.rows,
          citation: table.citation,
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
export async function fetchCatalogue(peer: RequestPeer, runId: string): Promise<Catalogue> {
  try {
    const answer = (await peer.request("tool.catalogue", { runId })) as Catalogue;
    return {
      tools: Array.isArray(answer?.tools) ? answer.tools : [],
      mode: typeof answer?.mode === "string" ? answer.mode : "unknown",
    };
  } catch {
    return { tools: [], mode: "unknown" };
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
  peer: RequestPeer,
  ledger: GrantLedger,
  runId: string,
  modelId: string,
  observe?: (observation: { tool: string; args: unknown; text: string }) => void,
  eligible?: readonly EligibleTool[],
): AgentTool[] {
  const definitions =
    eligible === undefined
      ? TOOL_DEFINITIONS.map((definition) => ({ definition, readOnly: definition.readOnly }))
      : eligible
          .map((entry) => {
            const definition = definitionFor(entry.name);
            return definition ? { definition, readOnly: entry.readOnly } : undefined;
          })
          .filter((entry): entry is { definition: ToolDefinition; readOnly: boolean } =>
            entry !== undefined,
          );

  return definitions.map(({ definition, readOnly }) =>
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
      // Reads may overlap: one cannot change what another returns, so several
      // at once cost the operator the slowest rather than the sum. Everything
      // that writes, produces a file, runs code or asks a person is serialised.
      executionMode: readOnly ? "parallel" : "sequential",
    }),
  );
}

export const toolErrorCode: ErrorCodeValue = ErrorCode.ToolFailed;
