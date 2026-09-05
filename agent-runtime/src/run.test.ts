/**
 * The Phase 1 loop, end to end, against a real HTTP model server.
 *
 * This is the test that proves the parts fit: a prompt goes to an
 * OpenAI-compatible endpoint, the reply asks for a tool, the tool call is put
 * to the gateway, the gateway's grant is spent executing it, the result goes
 * back to the model, and the model answers. Everything except the model's
 * judgement and the Rust core is real — the server speaks genuine SSE and the
 * agent loop is OpenClaw's, unmodified.
 *
 * The two fakes are deliberate and opposite in kind. The model server is fake
 * because a real one would make this test need a GPU and a 5 GB download. The
 * Rust core is fake because its own behaviour is tested in Rust; what is under
 * test here is that this side asks it the right questions in the right order.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { startRun, terminationOf, type RunRequest } from "./run.js";
import { TOOL_DEFINITIONS } from "./catalogue.js";
import type { ContextCommit } from "./durable-context.js";

/** One SSE chunk in the shape an OpenAI-compatible server emits. */
function chunk(delta: unknown, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`;
}

/**
 * A local inference server that replies with a scripted turn each time.
 *
 * Requests are recorded so the test can assert what the model was actually
 * shown — which is the only way to know the tool result reached it.
 */
function modelServer(turns: string[][]): Promise<{
  baseUrl: string;
  requests: unknown[];
  close: () => Promise<void>;
}> {
  const requests: unknown[] = [];
  let turn = 0;

  const server: Server = createServer((req, res) => {
    let body = "";
    req.on("data", (c) => {
      body += c;
    });
    req.on("end", () => {
      requests.push(JSON.parse(body || "{}"));
      const script = turns[Math.min(turn, turns.length - 1)] ?? [];
      turn += 1;
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      for (const line of script) res.write(line);
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });

  return new Promise((resolve) => {
    // Port 0 so parallel test files cannot collide, and loopback because the
    // runtime refuses anything else.
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        requests,
        close: () => new Promise<void>((done) => server.close(() => done())),
      });
    });
  });
}

/**
 * The eligibility answer Rust gives a run with an ordinary plan.
 *
 * Built from this runtime's own catalogue so a tool added later is offered to
 * these tests automatically — a fixed list here would silently stop exercising
 * whatever was added after it was written.
 */
function eligibleTools() {
  return {
    tools: TOOL_DEFINITIONS.map((definition) => ({
      name: definition.name,
      summary: definition.label,
      readOnly: definition.readOnly,
      approvalClass: definition.readOnly ? "automatic" : "personBeforeEffect",
      network: "none",
      maxResponseBytes: 16 * 1024,
      timeoutSeconds: 30,
    })),
    mode: "Work",
  };
}

/** A peer standing in for the Rust core, scripted per method. */
function coreStub(handlers: Record<string, (params: unknown) => unknown>) {
  const calls: Array<{ method: string; params: unknown }> = [];
  const events: unknown[] = [];
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);

  peer.request = ((method: string, params: unknown) => {
    calls.push({ method, params });
    // Served by default so each test can be about its own property rather than
    // about the one-off eligibility fetch every run makes. A test that cares
    // what the catalogue said overrides it like any other handler.
    const handler = handlers[method] ?? (method === "tool.catalogue" ? eligibleTools : undefined);
    if (!handler) return Promise.reject(new Error(`core stub has no ${method}`));
    try {
      return Promise.resolve(handler(params));
    } catch (error) {
      return Promise.reject(error);
    }
  }) as RpcPeer["request"];

  peer.notify = ((method: string, params: unknown) => {
    if (method === "run.event") events.push(params);
  }) as RpcPeer["notify"];

  return {
    peer,
    calls,
    events,
    /**
     * The methods this run called about tools, without the eligibility fetch.
     *
     * `calls` still holds every request, so nothing is hidden. This is what a
     * test asserting a call *sequence* wants: the fetch happens once at start-up
     * and says nothing about whether authorisation preceded execution.
     */
    get toolMethods() {
      return calls.map((call) => call.method).filter((method) => method.startsWith("tool.") && method !== "tool.catalogue");
    },
  };
}

let server: Awaited<ReturnType<typeof modelServer>> | undefined;

afterEach(async () => {
  await server?.close();
  server = undefined;
});

function request(baseUrl: string, prompt = "What is the seal specification?"): RunRequest {
  return {
    runId: "run-1",
    messageId: "msg-1",
    prompt,
    systemPrompt: "Search before answering.",
    model: { id: "test-model", provider: "sovereign-local", baseUrl, contextWindow: 8192, maxTokens: 256 },
  };
}

describe("a run that uses a tool", () => {
  beforeEach(async () => {
    server = await modelServer([
      // Turn one: ask for the tool.
      [
        chunk({ role: "assistant", content: "" }),
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "knowledge.search_authorized", arguments: '{"query":"seal specification"}' },
            },
          ],
        }),
        chunk({}, "tool_calls"),
      ],
      // Turn two: answer from what the tool returned.
      [chunk({ role: "assistant", content: "" }), chunk({ content: "The seal is 9.0 mm." }), chunk({}, "stop")],
    ]);
  });

  it("commits the real model message before authorizing its tool and checkpoints its result", async () => {
    let revision = 0;
    let rawSeq = 0;
    const messages: unknown[] = [];
    let assistantDurable = false;
    const core = coreStub({
      "context.load": () => ({ protocolVersion: 1, view: null, tail: [] }),
      "context.commit": (value) => {
        const boundary = value as ContextCommit;
        expect(boundary.expectedRevision).toBe(revision);
        expect(boundary.attemptId).toBe("attempt-1");
        for (const entry of boundary.entries) {
          messages.push(entry.message);
          if (entry.message.role === "assistant") assistantDurable = true;
        }
        rawSeq += boundary.entries.length;
        return { protocolVersion: 1, runId: "run-1", revision: ++revision, checkpointId: boundary.commitId,
          rawSeq, projectionSeq: rawSeq, phase: boundary.phase, messages: boundary.projection ?? [],
          notes: boundary.notes, ledger: boundary.ledger, pendingApprovals: [], unsettledEffects: [] };
      },
      "tool.authorize": () => {
        expect(assistantDurable).toBe(true);
        return { outcome: "allow", grant: "durable-grant" };
      },
      "tool.execute": () => ({ text: "Maintenance SOP p.4: the seal is 9.0 mm." }),
    });
    const input = request(server!.baseUrl);
    input.execution = { protocolVersion: 1, attemptId: "attempt-1", fenceToken: 1 };
    const outcome = await startRun(core.peer, input, () => {});
    expect(outcome.outcome.kind).toBe("completed");
    expect(messages.map((message) => (message as { role: string }).role)).toEqual(["user", "assistant", "toolResult", "assistant"]);
    expect(core.calls.filter((call) => call.method === "context.commit").at(-1)?.params).toMatchObject({ phase: "finished" });
  });

  it("does not execute a tool when its model message cannot be durably recorded", async () => {
    let revision = 0;
    let rawSeq = 0;
    const core = coreStub({
      "context.load": () => ({ protocolVersion: 1, view: null, tail: [] }),
      "context.commit": (value) => {
        const boundary = value as ContextCommit;
        if (boundary.entries.some((entry) => entry.message.role === "assistant")) throw new Error("injected disk failure");
        rawSeq += boundary.entries.length;
        return { protocolVersion: 1, runId: "run-1", revision: ++revision, checkpointId: boundary.commitId,
          rawSeq, projectionSeq: rawSeq, phase: boundary.phase, messages: boundary.projection ?? [],
          notes: boundary.notes, ledger: boundary.ledger, pendingApprovals: [], unsettledEffects: [] };
      },
    });
    const input = request(server!.baseUrl);
    input.execution = { protocolVersion: 1, attemptId: "attempt-1", fenceToken: 1 };
    const outcome = await startRun(core.peer, input, () => {});
    expect(outcome.outcome.kind).toBe("needsReview");
    expect(core.toolMethods).toEqual([]);
    expect(server!.requests).toHaveLength(1);
  });

  it("recovers a tool receipt after losing the worker's result checkpoint, then continues the same conversation", async () => {
    let revision = 0;
    let rawSeq = 0;
    let projectionSeq = 0;
    let projection: ContextCommit["projection"] = [];
    let view: unknown = null;
    const history: { seq: number; entryId: string; message: ContextCommit["entries"][number]["message"] }[] = [];
    let loseResultCheckpoint = true;
    let effects = 0;
    let toolReceipt: unknown;
    const core = coreStub({
      "context.load": () => ({ protocolVersion: 1, view, tail: history.filter((entry) => entry.seq > projectionSeq) }),
      "context.commit": (value) => {
        const boundary = value as ContextCommit;
        expect(boundary.expectedRevision).toBe(revision);
        if (loseResultCheckpoint && boundary.phase === "afterTool") {
          loseResultCheckpoint = false;
          throw new Error("worker disappeared before receiving its checkpoint acknowledgment");
        }
        for (const entry of boundary.entries) history.push({ ...entry, seq: ++rawSeq });
        if (boundary.projection) { projection = boundary.projection; projectionSeq = rawSeq; }
        view = { protocolVersion: 1, runId: "run-1", revision: ++revision, checkpointId: boundary.commitId,
          rawSeq, projectionSeq, phase: boundary.phase, messages: projection,
          notes: boundary.notes, ledger: boundary.ledger, pendingApprovals: [], unsettledEffects: [] };
        return view;
      },
      "tool.authorize": (value) => {
        expect(value).toMatchObject({ operationSeq: 2, toolCallId: "call_1" });
        return { outcome: "allow", grant: "g" };
      },
      "tool.execute": () => {
        if (!toolReceipt) { effects++; toolReceipt = { text: "Source A-17: the seal is 9.0 mm." }; }
        return toolReceipt;
      },
    });
    const input = { ...request(server!.baseUrl), execution: { protocolVersion: 1 as const, attemptId: "first", fenceToken: 1 } };
    expect((await startRun(core.peer, input, () => {})).outcome.kind).toBe("needsReview");
    expect(history.map((entry) => entry.message.role)).toEqual(["user", "assistant"]);
    const resumed = await startRun(core.peer, { ...input, execution: { ...input.execution, attemptId: "second", fenceToken: 2 } }, () => {});
    expect(resumed.outcome.kind).toBe("completed");
    expect(resumed.text).toBe("The seal is 9.0 mm.");
    expect(effects).toBe(1);
    expect(history.map((entry) => entry.message.role)).toEqual(["user", "assistant", "toolResult", "assistant"]);
    expect(server!.requests).toHaveLength(2);
    expect(JSON.stringify(server!.requests[1])).toContain("Source A-17");
  });

  it("authorises before executing, and executes only with the grant it was given", async () => {
    const core = coreStub({
      "tool.authorize": () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" }),
      "tool.execute": () => ({ text: "1 passage found. Maintenance SOP p.4: seal 9.0 mm." }),
    });

    const outcome = await startRun(core.peer, request(server!.baseUrl), () => {});

    const methods = core.toolMethods;
    expect(methods).toEqual(["tool.authorize", "tool.execute"]);

    // The grant issued by the authorise step is the one spent executing.
    // Found by method rather than by index: `calls` records every request,
    // including the one-off `tool.catalogue` eligibility fetch at index 0, so
    // `calls[1]` is the authorise call - which correctly carries no grant,
    // because the grant is in its reply, not its request.
    const executeCall = core.calls.find((call) => call.method === "tool.execute");
    expect(executeCall?.params).toMatchObject({
      runId: "run-1",
      toolCallId: "call_1",
      tool: "knowledge.search_authorized",
      grant: "g-1",
    });
    expect(outcome.text).toBe("The seal is 9.0 mm.");
  });

  it("gives the model the tool result, so the answer is grounded in it", async () => {
    const core = coreStub({
      "tool.authorize": () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" }),
      "tool.execute": () => ({ text: "1 passage found. Maintenance SOP p.4: seal 9.0 mm." }),
    });

    await startRun(core.peer, request(server!.baseUrl), () => {});

    // Two turns means the tool result went back and the model spoke again.
    expect(server!.requests).toHaveLength(2);
    const secondTurn = JSON.stringify(server!.requests[1]);
    expect(secondTurn).toContain("Maintenance SOP p.4");
  });

  it("reports lifecycle to the operator without echoing tool arguments", async () => {
    const core = coreStub({
      "tool.authorize": () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" }),
      "tool.execute": () => ({ text: "1 passage found." }),
    });

    await startRun(core.peer, request(server!.baseUrl), () => {});

    const types = core.events.map((event) => (event as { event: { type: string } }).event.type);
    expect(types).toContain("agent_start");
    expect(types).toContain("tool_execution_start");
    expect(types).toContain("agent_end");

    // The arguments are in the audit record under access control; sending them
    // again over the event channel would put document text on a second path.
    const toolEvents = core.events.filter((event) =>
      (event as { event: { type: string } }).event.type.startsWith("tool_execution"),
    );
    expect(toolEvents.length).toBeGreaterThan(0);
    for (const event of toolEvents) {
      expect((event as { event: { args?: unknown } }).event.args).toBeUndefined();
    }
  });
});

describe("a run whose tool call is refused", () => {
  beforeEach(async () => {
    server = await modelServer([
      [
        chunk({ role: "assistant", content: "" }),
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "knowledge.search_authorized", arguments: '{"query":"salary list"}' },
            },
          ],
        }),
        chunk({}, "tool_calls"),
      ],
      [
        chunk({ role: "assistant", content: "" }),
        chunk({ content: "I am not permitted to search that." }),
        chunk({}, "stop"),
      ],
    ]);
  });

  it("never executes, and hands the refusal back as something the model can read", async () => {
    const core = coreStub({
      "tool.authorize": () => ({
        outcome: "refuse",
        reason: "You do not hold SearchKnowledge for that collection.",
      }),
      "tool.execute": () => {
        throw new Error("execute must not be reached after a refusal");
      },
    });

    const outcome = await startRun(core.peer, request(server!.baseUrl), () => {});

    expect(core.toolMethods).toEqual(["tool.authorize"]);
    // The refusal reaches the model as a tool result, so it can say so rather
    // than stall — the run completes.
    expect(JSON.stringify(server!.requests[1])).toContain("SearchKnowledge");
    expect(outcome.text).toContain("not permitted");
  });
});

describe("a run the gateway cannot be asked about", () => {
  beforeEach(async () => {
    server = await modelServer([
      [
        chunk({ role: "assistant", content: "" }),
        chunk({
          tool_calls: [
            {
              index: 0,
              id: "call_1",
              type: "function",
              function: { name: "knowledge.search_authorized", arguments: '{"query":"x"}' },
            },
          ],
        }),
        chunk({}, "tool_calls"),
      ],
      [chunk({ role: "assistant", content: "" }), chunk({ content: "I could not check." }), chunk({}, "stop")],
    ]);
  });

  it("fails closed rather than running the tool anyway", async () => {
    const core = coreStub({
      "tool.authorize": () => {
        throw new Error("core closed the channel");
      },
      "tool.execute": () => {
        throw new Error("execute must not be reached when authorisation failed");
      },
    });

    await startRun(core.peer, request(server!.baseUrl), () => {});

    expect(core.toolMethods).toEqual(["tool.authorize"]);
    expect(JSON.stringify(server!.requests[1])).toContain("authorisation is unavailable");
  });
});

describe("endpoint policy", () => {
  it("refuses a model endpoint that is not loopback, before any socket opens", async () => {
    const core = coreStub({});
    await expect(
      startRun(
        core.peer,
        { ...request("https://api.openai.com/v1"), runId: "run-x" }, // arjun-egress-ok: a fixture proving this endpoint is refused, never reached
        () => {},
      ),
    ).rejects.toThrow(/not loopback/);
    expect(core.calls).toHaveLength(0);
  });

  it("refuses a private-network endpoint too, not just the public internet", async () => {
    const core = coreStub({});
    await expect(
      startRun(core.peer, request("http://192.168.1.50:8000/v1"), () => {}), // arjun-egress-ok: a fixture proving a private-network endpoint is refused
    ).rejects.toThrow(/not loopback/);
  });

  it("accepts the loopback forms a local server actually binds to", async () => {
    // localhost, 127.x and ::1 are all legitimate llama-server/vLLM bindings.
    server = await modelServer([[chunk({ role: "assistant", content: "" }), chunk({ content: "ok" }), chunk({}, "stop")]]);
    const port = new URL(server.baseUrl).port;
    const core = coreStub({});
    const outcome = await startRun(core.peer, request(`http://localhost:${port}/v1`), () => {});
    expect(outcome.text).toBe("ok");
  });
});

describe("aborting", () => {
  it("registers a handle the core can use to stop the run", async () => {
    server = await modelServer([[chunk({ role: "assistant", content: "" }), chunk({ content: "done" }), chunk({}, "stop")]]);
    const core = coreStub({});
    let registered: { abort: (reason?: unknown) => void } | undefined;

    await startRun(core.peer, request(server.baseUrl), (run) => {
      registered = run;
    });

    expect(registered).toBeDefined();
    // Aborting a finished run is a normal race and must not throw.
    expect(() => registered!.abort("operator stopped it")).not.toThrow();
  });
});

/**
 * A model server that opens a stream and never finishes it.
 *
 * The only thing that can end a run against this server is the run's own
 * deadline or an operator, which is exactly what the two tests below are about.
 */
function stallingServer(): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const open: Array<{ end: () => void }> = [];
  const server: Server = createServer((req, res) => {
    req.on("data", () => {});
    req.on("end", () => {
      res.writeHead(200, {
        "content-type": "text/event-stream",
        "cache-control": "no-cache",
        connection: "keep-alive",
      });
      res.write(chunk({ role: "assistant", content: "" }));
      open.push(res);
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        close: () =>
          new Promise<void>((done) => {
            for (const res of open) res.end();
            server.close(() => done());
          }),
      });
    });
  });
}

/** A model server that answers every request with a provider error. */
function erroringServer(): Promise<{ baseUrl: string; close: () => Promise<void> }> {
  const server: Server = createServer((req, res) => {
    req.on("data", () => {});
    req.on("end", () => {
      res.writeHead(503, { "content-type": "application/json" });
      res.end(JSON.stringify({ error: { message: "the model server is loading a model" } }));
    });
  });
  return new Promise((resolve) => {
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      resolve({
        baseUrl: `http://127.0.0.1:${port}/v1`,
        close: () => new Promise<void>((done) => server.close(() => done())),
      });
    });
  });
}

/**
 * How a run ended, as the loop actually ended it.
 *
 * The defect these pin: the core used to read "the `run.start` request
 * resolved" as "the task completed". Every ordinary ending of an agent loop
 * resolves that request -- a stop button, a provider error, the model's output
 * cap -- so a truncated fragment, a stopped run and a finished answer were
 * recorded, listed and shown as the same thing.
 */
describe("terminationOf: the ending is read off the loop, not off the transport", () => {
  it("reports a clean stop as completed, with nothing to excuse", () => {
    expect(terminationOf({ finalAssistant: { stopReason: "stop" } })).toEqual({
      kind: "completed",
    });
  });

  it("reports a provider error as failed, carrying the provider's sentence", () => {
    const outcome = terminationOf({
      finalAssistant: { stopReason: "error", errorMessage: "the model server refused: 503" },
      errorMessage: "the model server refused: 503",
    });
    expect(outcome.kind).toBe("failed");
    expect(outcome.detail).toContain("503");
  });

  it("reports the output cap as lengthLimited, never as completed", () => {
    const outcome = terminationOf({ finalAssistant: { stopReason: "length" } });
    expect(outcome.kind).toBe("lengthLimited");
    // The distinction that matters: the answer is a fragment.
    expect(outcome.detail).toMatch(/cut off/i);
  });

  it("reports an operator stop as aborted, with the cause recorded at the stop", () => {
    const outcome = terminationOf({
      finalAssistant: { stopReason: "aborted" },
      abortCause: { kind: "aborted", detail: "Stopped: operator stopped it" },
    });
    expect(outcome).toEqual({ kind: "aborted", detail: "Stopped: operator stopped it" });
  });

  it("reports a deadline stop as budgetStopped, not as a plain abort", () => {
    // Same `stopReason` from the loop's point of view; different endings to a
    // person reading the run. Which one it was is knowable only where the
    // abort was asked for, so it is recorded there.
    const outcome = terminationOf({
      finalAssistant: { stopReason: "aborted" },
      abortCause: {
        kind: "budgetStopped",
        detail: "Stopped: it ran past the time its plan allowed.",
      },
    });
    expect(outcome.kind).toBe("budgetStopped");
  });

  it("falls back to a plain abort when nothing recorded who asked", () => {
    expect(terminationOf({ finalAssistant: { stopReason: "aborted" } })).toEqual({
      kind: "aborted",
      detail: "Stopped before it finished.",
    });
  });

  it("reports an error the loop recorded after the last turn as failed", () => {
    const outcome = terminationOf({
      finalAssistant: { stopReason: "stop" },
      errorMessage: "the stream ended before the turn did",
    });
    expect(outcome.kind).toBe("failed");
    expect(outcome.detail).toBe("the stream ended before the turn did");
  });

  it("treats a run with no assistant message as completed only if nothing failed", () => {
    expect(terminationOf({}).kind).toBe("completed");
    expect(terminationOf({ errorMessage: "spawn failed" }).kind).toBe("failed");
  });
});

describe("startRun: the outcome it returns", () => {
  it("is completed for a run that answered", async () => {
    server = await modelServer([
      [chunk({ role: "assistant", content: "" }), chunk({ content: "ok" }), chunk({}, "stop")],
    ]);
    const core = coreStub({});
    const outcome = await startRun(core.peer, request(server.baseUrl), () => {});
    expect(outcome.outcome).toEqual({ kind: "completed" });
  });

  it("is lengthLimited for a turn the model server cut at the output cap", async () => {
    server = await modelServer([
      [
        chunk({ role: "assistant", content: "" }),
        chunk({ content: "The seal specification is " }),
        chunk({}, "length"),
      ],
    ]);
    const core = coreStub({});
    const outcome = await startRun(core.peer, request(server.baseUrl), () => {});
    expect(outcome.outcome.kind).toBe("lengthLimited");
    // The fragment is still returned: a cut-off answer is worth showing, as
    // long as it is not called finished.
    expect(outcome.text).toContain("The seal specification is");
  });

  it("is budgetStopped for a run whose deadline expired mid-flight", async () => {
    const stalled = await stallingServer();
    try {
      const core = coreStub({});
      const outcome = await startRun(
        core.peer,
        { ...request(stalled.baseUrl), deadlineMs: Date.now() + 150 },
        () => {},
      );
      expect(outcome.outcome.kind).toBe("budgetStopped");
      expect(outcome.outcome.detail).toMatch(/time its plan allowed/);
    } finally {
      await stalled.close();
    }
  });

  it("is aborted for a run an operator stopped mid-flight", async () => {
    const stalled = await stallingServer();
    try {
      const core = coreStub({});
      const outcome = await startRun(core.peer, request(stalled.baseUrl), (handle) => {
        setTimeout(() => handle.abort("operator stopped it"), 60);
      });
      expect(outcome.outcome.kind).toBe("aborted");
      expect(outcome.outcome.detail).toContain("operator stopped it");
    } finally {
      await stalled.close();
    }
  });

  it("is failed, never completed, when the model server errors", async () => {
    const broken = await erroringServer();
    try {
      const core = coreStub({});
      // This is the case the whole typed outcome exists for. A provider that
      // refuses is an ordinary ending of an agent loop, so `agent.prompt`
      // *resolves* and the `run.start` request resolves with it. Reading that
      // resolution as success is what recorded a 503 as a finished task.
      const outcome = await startRun(core.peer, request(broken.baseUrl), () => {});
      expect(outcome.outcome.kind).toBe("failed");
      expect(outcome.outcome.detail).toBeTruthy();
    } finally {
      await broken.close();
    }
  });
});
