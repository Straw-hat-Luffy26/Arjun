/**
 * Entry point for the sovereign agent runtime.
 *
 * Spawned by the Rust core as a child process. Speaks JSON-RPC over stdio and
 * opens nothing else -- no listening socket, no outbound connection except to
 * the loopback inference endpoint the router chose (enforced in `run.ts`).
 */

import { RpcPeer, type PeerTransport } from "./peer.js";
import { ErrorCode } from "./protocol.js";
import { startRun, type ActiveRun, type RunRequest } from "./run.js";
import type { PreservedState } from "./compaction.js";
import type { WorkingNotesState } from "./working-notes.js";

/**
 * Makes stdout unusable for anything except frames.
 *
 * The framing is newline-delimited, so one stray `console.log` -- ours, or a
 * vendored dependency's, or a dependency's dependency's -- desynchronises the
 * channel and the failure surfaces far from its cause. Rather than grep for
 * violations and trust that none appear later, the console methods are rebound
 * to stderr and the real stdout writer is captured privately here.
 *
 * `process.stdout.write` itself is left alone: monkey-patching it would break
 * the very writer the transport needs, and anything reaching for it directly is
 * doing so deliberately.
 */
function installStdoutGuard(): (line: string) => void {
  const write = process.stdout.write.bind(process.stdout);
  const toStderr =
    (level: string) =>
    (...args: unknown[]): void => {
      const text = args
        .map((arg) => (typeof arg === "string" ? arg : safeInspect(arg)))
        .join(" ");
      process.stderr.write(`[agent-runtime:${level}] ${text}\n`);
    };
  console.log = toStderr("log");
  console.info = toStderr("info");
  console.warn = toStderr("warn");
  console.error = toStderr("error");
  console.debug = toStderr("debug");
  return (line: string) => {
    write(line);
  };
}

function safeInspect(value: unknown): string {
  try {
    return JSON.stringify(value) ?? String(value);
  } catch {
    return String(value);
  }
}

function stdioTransport(write: (line: string) => void): PeerTransport {
  return {
    write,
    onData(sink) {
      process.stdin.setEncoding("utf8");
      process.stdin.on("data", (chunk: string) => sink(chunk));
    },
    onClose(sink) {
      process.stdin.on("end", sink);
      process.stdin.on("close", sink);
    },
  };
}

function main(): void {
  const write = installStdoutGuard();
  const peer = new RpcPeer(stdioTransport(write));

  /**
   * Runs in flight, so an abort can reach one.
   *
   * A map rather than a single slot because the core may drive more than one
   * run at a time in a later phase; today it drives one, and the map costs
   * nothing while removing an assumption that would be awkward to unpick.
   */
  const active = new Map<string, ActiveRun>();
  const starting = new Set<string>();

  peer.onFatal((error) => {
    process.stderr.write(
      `[agent-runtime:fatal] channel desynchronised: ${error instanceof Error ? error.message : String(error)}\n`,
    );
    // Exit rather than continue: past this point neither end can trust what the
    // other said, and a runtime that keeps executing tool calls on a channel it
    // cannot parse is exactly the failure this product must not have.
    process.exit(70);
  });

  peer.handle("run.start", async (params) => {
    const request = params as RunRequest;
    if (
      !request?.runId ||
      typeof request.messageId !== "string" ||
      request.messageId.length === 0 ||
      typeof request.prompt !== "string" ||
      !request.model?.baseUrl || request.execution?.protocolVersion !== 1
      || typeof request.execution.attemptId !== "string" || request.execution.attemptId.length === 0
      || !Number.isSafeInteger(request.execution.fenceToken) || request.execution.fenceToken < 1
    ) {
      throw Object.assign(
        new Error("run.start needs runId, messageId, prompt and model.baseUrl"),
        { code: ErrorCode.BadParams },
      );
    }
    if (starting.has(request.runId) || active.has(request.runId)) {
      throw Object.assign(new Error("This run already has an active worker."), { code: ErrorCode.Refused });
    }
    starting.add(request.runId);
    try {
      return await startRun(peer, request, (run) => active.set(request.runId, run));
    } finally {
      starting.delete(request.runId);
      active.delete(request.runId);
    }
  });

  peer.handle("run.abort", (params) => {
    const { runId, reason } = (params ?? {}) as { runId?: string; reason?: string };
    if (!runId) {
      throw Object.assign(new Error("run.abort needs runId"), { code: ErrorCode.BadParams });
    }
    const run = active.get(runId);
    // Not an error: the run finishing just before the abort arrived is a normal
    // race, and reporting it as a failure would make operators doubt the button.
    run?.abort(reason ?? "aborted by operator");
    return { aborted: Boolean(run) };
  });

  peer.handle("run.steer", (params) => {
    const { runId, text } = (params ?? {}) as { runId?: string; text?: string };
    if (!runId || typeof text !== "string" || text.trim().length === 0) {
      throw Object.assign(new Error("run.steer needs runId and non-empty text"), {
        code: ErrorCode.BadParams,
      });
    }
    const run = active.get(runId);
    // Same reasoning as abort: the run finishing just as the correction arrives
    // is an ordinary race, and reporting it as a failure would make an operator
    // distrust the control.
    run?.steer(text);
    return { steered: Boolean(run) };
  });

  /**
   * Updates the state a run must carry across compaction.
   *
   * Notifications would be cheaper, but this is a request so the core learns
   * whether the update landed. A preserved approval that was silently dropped
   * because the run had already ended is the one case where "probably
   * delivered" is not good enough — the next thing the core does with that
   * belief is decide not to ask a person again.
   */
  peer.handle("run.note", (params) => {
    const { runId, preserved, notes } = (params ?? {}) as {
      runId?: string;
      preserved?: PreservedState;
      notes?: Partial<WorkingNotesState>;
    };
    if (!runId || (preserved === undefined && notes === undefined)) {
      throw Object.assign(new Error("run.note needs runId and preserved or notes"), {
        code: ErrorCode.BadParams,
      });
    }
    const run = active.get(runId);
    run?.note({ preserved, notes });
    return { noted: Boolean(run), notes: run?.notes.state };
  });

  peer.handle("health", () => ({ ready: true, pid: process.pid, node: process.version, contextProtocolVersion: 1 }));

  process.stdin.on("end", () => {
    // The core went away. Nothing left to serve, and lingering would leave an
    // orphan holding a model server connection.
    for (const run of active.values()) run.abort("core closed the channel");
    process.exit(0);
  });

  process.stderr.write(`[agent-runtime:log] ready pid=${process.pid} node=${process.version}\n`);
}

main();
