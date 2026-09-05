/** Versioned, acknowledged checkpoints over the existing private stdio channel. */
import { randomUUID } from "node:crypto";
import type { AgentMessage } from "@openclaw/agent-core";
import type { RpcPeer } from "./peer.js";
import type { ContextLedgerSnapshot } from "./context-ledger.js";
import type { WorkingNotesState } from "./working-notes.js";
import { pendingToolCalls } from "./tool-recovery.js";

export type RequestPeer = Pick<RpcPeer, "request">;
export interface ExecutionIdentity { protocolVersion: 1; attemptId: string; fenceToken: number }
export type ContextPhase = "observed" | "modelReady" | "beforeTool" | "afterTool" | "compactionStarted" | "compactionCompleted" | "finished" | "paused";
export interface ContextView {
  protocolVersion: 1;
  runId: string;
  revision: number;
  checkpointId: string;
  rawSeq: number;
  projectionSeq: number;
  phase: ContextPhase;
  messages: AgentMessage[];
  notes: WorkingNotesState;
  ledger: ContextLedgerSnapshot | null;
  pendingApprovals: string[];
  unsettledEffects: string[];
}
export interface ContextCommit {
  protocolVersion: 1;
  runId: string;
  attemptId: string;
  fenceToken: number;
  expectedRevision: number;
  commitId: string;
  phase: ContextPhase;
  entries: { entryId: string; message: AgentMessage }[];
  projection: AgentMessage[] | null;
  notes: WorkingNotesState;
  ledger: ContextLedgerSnapshot | null;
}

export class DurableContextError extends Error {
  constructor(detail: string) { super(detail); this.name = "DurableContextError"; }
}

function viewOf(value: unknown, runId: string): ContextView {
  const view = value as Partial<ContextView> | null;
  if (!view || view.protocolVersion !== 1 || view.runId !== runId || !Number.isSafeInteger(view.revision)
      || (view.revision ?? 0) < 1 || !Number.isSafeInteger(view.rawSeq) || !Number.isSafeInteger(view.projectionSeq)
      || !Array.isArray(view.messages) || !view.notes || typeof view.notes.goal !== "string"
      || !Array.isArray(view.pendingApprovals) || !Array.isArray(view.unsettledEffects)) {
    throw new DurableContextError("The core returned an invalid durable context checkpoint.");
  }
  return view as ContextView;
}

/** Each instance captures an immutable attempt. Never look up a newer worker's
 * identity from a shared mutable run-id map when sending an old call. */
export function scopedPeer(peer: RequestPeer, runId: string, identity: ExecutionIdentity): RequestPeer {
  return { request: (method, params) => peer.request(method, {
    ...(params as Record<string, unknown> | undefined), runId,
    attemptId: identity.attemptId, fenceToken: identity.fenceToken,
  }) };
}

export class DurableContext {
  readonly #peer: RequestPeer;
  readonly #identity: ExecutionIdentity;
  readonly #runId: string;
  #revision = 0;
  #rawSeq = 0;
  #queue: Promise<unknown> = Promise.resolve();
  #failure: unknown;
  readonly #toolSequences = new Map<string, number>();

  constructor(peer: RequestPeer, runId: string, identity: ExecutionIdentity) {
    this.#peer = peer; this.#runId = runId; this.#identity = identity;
  }

  toolPeer(peer: RequestPeer): RequestPeer {
    return { request: (method, params) => {
      if (method !== "tool.authorize" && method !== "tool.execute") return peer.request(method, params);
      const values = params as Record<string, unknown>;
      const seq = this.#toolSequences.get(String(values.toolCallId));
      if (!seq) return Promise.reject(new DurableContextError("The tool's assistant request has not been durably acknowledged."));
      return peer.request(method, { ...values, operationSeq: seq });
    } };
  }

  #rememberToolSequences(message: AgentMessage, seq: number): void {
    if (message.role !== "assistant") return;
    this.#toolSequences.clear();
    for (const block of message.content) if (block.type === "toolCall") this.#toolSequences.set(block.id, seq);
  }

  async load(): Promise<{ view: ContextView | null; messages: AgentMessage[] }> {
    const result = await this.#peer.request("context.load", { runId: this.#runId }) as {
      protocolVersion?: number; view?: unknown; tail?: { seq: number; entryId: string; message: AgentMessage }[];
    };
    if (result?.protocolVersion !== 1 || !Array.isArray(result.tail)) throw new DurableContextError("The core does not support durable context recovery.");
    if (result.view == null) return { view: null, messages: [] };
    const view = viewOf(result.view, this.#runId);
    if (view.unsettledEffects.length) throw new DurableContextError("Unsettled effects require reconciliation before this run continues.");
    this.#revision = view.revision; this.#rawSeq = view.rawSeq;
    // The current notes are re-injected from structured state. Do not carry an
    // older rendered copy alongside them after restarting.
    const messages = view.messages.filter((message) => !(message as unknown as { arjunContextState?: boolean }).arjunContextState);
    let expected = view.projectionSeq;
    for (const entry of result.tail) {
      if (entry.seq !== ++expected) throw new DurableContextError("The recovery transcript has a gap.");
      messages.push(entry.message);
      (entry.message as AgentMessage & { arjunRawSeq: number }).arjunRawSeq = entry.seq;
      this.#rememberToolSequences(entry.message, entry.seq);
    }
    if (expected !== view.rawSeq) {
      throw new DurableContextError("The recovery transcript needs tool reconciliation before it is provider-valid.");
    }
    try { pendingToolCalls(messages); } catch { throw new DurableContextError("The recovery transcript contains an invalid tool batch."); }
    return { view, messages };
  }

  commit(phase: ContextPhase, notes: WorkingNotesState, ledger: ContextLedgerSnapshot | null,
      options: { message?: AgentMessage; projection?: AgentMessage[] } = {}): Promise<ContextView> {
    // Snapshot mutable inputs at the boundary, before waiting behind another
    // tool in a parallel batch. A later note update cannot rewrite this event.
    const captured = structuredClone({ notes, ledger, options });
    const commitId = randomUUID();
    const queued = this.#queue.then(async () => {
      if (this.#failure) throw this.#failure;
      const request: ContextCommit = {
        protocolVersion: 1, runId: this.#runId, attemptId: this.#identity.attemptId,
        fenceToken: this.#identity.fenceToken, expectedRevision: this.#revision, commitId, phase,
        entries: captured.options.message ? [{ entryId: `message:${commitId}`, message: captured.options.message }] : [],
        projection: captured.options.projection ?? null, notes: captured.notes, ledger: captured.ledger,
      };
      const next = viewOf(await this.#peer.request("context.commit", request), this.#runId);
      if (next.revision !== this.#revision + 1 || next.rawSeq < this.#rawSeq
          || next.checkpointId !== commitId || next.projectionSeq > next.rawSeq) {
        throw new DurableContextError("The durable context acknowledgment did not match this boundary.");
      }
      this.#revision = next.revision; this.#rawSeq = next.rawSeq;
      if (captured.options.message) this.#rememberToolSequences(captured.options.message, next.rawSeq);
      if (options.message) (options.message as AgentMessage & { arjunRawSeq: number }).arjunRawSeq = next.rawSeq;
      return next;
    }).catch((error: unknown) => {
      this.#failure = error instanceof DurableContextError ? error : new DurableContextError("The next step was refused because its context checkpoint could not be committed.");
      throw this.#failure;
    });
    // Keep a rejection handler attached even while the caller is unwinding; the
    // remembered failure makes every later boundary fail closed, not disappear.
    this.#queue = queued.catch(() => undefined);
    return queued;
  }
}
