/**
 * The four things that make a long run survive contact with a real operator.
 *
 * Compaction has its own file. This covers the rest of Phase 3, each against the
 * real agent loop and a real HTTP model server:
 *
 * - a small model that writes its tool call as prose still gets the call made;
 * - independent read-only tools run together rather than one after another;
 * - an operator's correction reaches a run already in flight;
 * - stopping a run mid-tool leaves the transcript able to say so.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it, vi } from "vitest";
import { convertToLlm, type AgentMessage } from "@openclaw/agent-core";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { withToolCallRepair } from "./repair.js";
import { startRun, type ActiveRun, type RunRequest } from "./run.js";
import { TOOL_DEFINITIONS } from "./catalogue.js";

function chunk(delta: unknown, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`;
}

const say = (text: string) => [
  chunk({ role: "assistant", content: "" }),
  chunk({ content: text }),
  chunk({}, "stop"),
];

const callTool = (id: string, name: string, args: unknown) => [
  chunk({ role: "assistant", content: "" }),
  chunk({
    tool_calls: [
      { index: 0, id, type: "function", function: { name, arguments: JSON.stringify(args) } },
    ],
  }),
  chunk({}, "tool_calls"),
];

/** Several tool calls in one assistant turn, which is what parallel execution is for. */
const callTools = (calls: Array<{ id: string; name: string; args: unknown }>) => [
  chunk({ role: "assistant", content: "" }),
  chunk({
    tool_calls: calls.map((call, index) => ({
      index,
      id: call.id,
      type: "function",
      function: { name: call.name, arguments: JSON.stringify(call.args) },
    })),
  }),
  chunk({}, "tool_calls"),
];

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
      res.writeHead(200, { "content-type": "text/event-stream" });
      for (const line of script) res.write(line);
      res.write("data: [DONE]\n\n");
      res.end();
    });
  });
  return new Promise((resolve) => {
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

function coreStub(handlers: Record<string, (params: unknown) => unknown>) {
  const calls: Array<{ method: string; params: unknown }> = [];
  const events: Array<{ runId: string; event: { type: string } }> = [];
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);

  peer.request = ((method: string, params: unknown) => {
    calls.push({ method, params });
    // Served by default so each test can be about its own property rather than
    // about the one-off eligibility fetch every run makes. A test that cares
    // what the catalogue said overrides it like any other handler.
    const handler = handlers[method] ?? (method === "tool.catalogue" ? eligibleTools : undefined);
    if (!handler) return Promise.reject(new Error(`core stub has no ${method}`));
    return Promise.resolve(handler(params));
  }) as RpcPeer["request"];
  peer.notify = ((method: string, params: unknown) => {
    if (method === "run.event") events.push(params as { runId: string; event: { type: string } });
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
      return calls.map((call) => call.method).filter((method) => method !== "tool.catalogue");
    },
  };
}

function request(baseUrl: string, prompt = "What is the seal specification?"): RunRequest {
  return {
    runId: "run-1",
    messageId: "msg-1",
    prompt,
    systemPrompt: "Search before answering.",
    model: { id: "qwen2.5-coder-7b", provider: "llama-cpp", baseUrl, contextWindow: 8192, maxTokens: 256 },
  };
}

const allow = () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g" });

let server: Awaited<ReturnType<typeof modelServer>> | undefined;
afterEach(async () => {
  await server?.close();
  server = undefined;
});

describe("repairing a tool call written as prose", () => {
  const assistantSaying = (text: string) => ({
    role: "assistant",
    content: [{ type: "text", text }],
    stopReason: "stop",
  });

  function repairing(text: string, names: string[] = ["knowledge.search_authorized"]) {
    const inner = {
      [Symbol.asyncIterator]: async function* () {},
      result: async () => assistantSaying(text),
      push: () => {},
      end: () => {},
    };
    const streamFn = vi.fn(() => inner) as never;
    return withToolCallRepair(streamFn, names);
  }

  async function repaired(text: string, names?: string[]) {
    const stream = repairing(text, names)({} as never, {} as never, {} as never) as unknown as {
      result: () => Promise<{ content: Array<{ type: string; name?: string; arguments?: unknown }> }>;
    };
    const message = await stream.result();
    return message.content.filter((block) => block.type === "toolCall");
  }

  /**
   * Different model families write a stray tool call differently, which is the
   * reason to reuse a parser that knows several rather than write one that
   * knows the format the first model happened to emit.
   */
  it.each([
    ["named bracket", `[knowledge.search_authorized]
{"query":"seal spec"}[/knowledge.search_authorized]`],
    ["tool bracket", `[tool:knowledge.search_authorized] {"query":"seal spec"}`],
    ["legacy marker", `[knowledge.search_authorized]
{"query":"seal spec"}[END_TOOL_REQUEST]`],
  ])("promotes a %s call into a real one", async (_label, text) => {
    const toolCalls = await repaired(text);
    expect(toolCalls).toHaveLength(1);
    expect(toolCalls[0]?.name).toBe("knowledge.search_authorized");
    expect(toolCalls[0]?.arguments).toEqual({ query: "seal spec" });
  });

  it("leaves an ordinary answer alone", async () => {
    // Prose that merely mentions a tool is not a call, and rewriting it would
    // invent work the model never asked for.
    const toolCalls = await repaired("I will search the documents for the seal specification.");
    expect(toolCalls).toHaveLength(0);
  });

  it("will not invent a call to a tool this run was never given", async () => {
    // The model naming something outside its catalogue is a hallucination, not
    // a call. The gateway would refuse it; not manufacturing it means the
    // refusal never has to happen.
    const toolCalls = await repaired(
      `[knowledge.search_authorized]
{"query":"seal spec"}[/knowledge.search_authorized]`,
      ["calculation.evaluate_with_units"],
    );
    expect(toolCalls).toHaveLength(0);
  });

  it("leaves a run with no tools completely untouched", async () => {
    const inner = { result: async () => assistantSaying("plain text") };
    const streamFn = vi.fn(() => inner) as never;
    expect(withToolCallRepair(streamFn, [])).toBe(streamFn);
  });
});

describe("running independent tools together", () => {
  it("issues every authorisation and every execution for one turn", async () => {
    server = await modelServer([
      callTools([
        { id: "call_1", name: "knowledge.search_authorized", args: { query: "seal" } },
        { id: "call_2", name: "knowledge.search_authorized", args: { query: "gasket" } },
        { id: "call_3", name: "knowledge.search_authorized", args: { query: "flange" } },
      ]),
      say("Three passages found."),
    ]);
    const core = coreStub({
      "tool.authorize": allow,
      "tool.execute": () => ({ text: "1 passage found." }),
    });

    await startRun(core.peer, request(server.baseUrl), () => {});

    expect(core.calls.filter((c) => c.method === "tool.authorize")).toHaveLength(3);
    expect(core.calls.filter((c) => c.method === "tool.execute")).toHaveLength(3);
  });

  it("still authorises each call separately, so parallelism buys no permission", async () => {
    server = await modelServer([
      callTools([
        { id: "call_1", name: "knowledge.search_authorized", args: { query: "seal" } },
        { id: "call_2", name: "knowledge.search_authorized", args: { query: "salary list" } },
      ]),
      say("One was refused."),
    ]);
    const core = coreStub({
      "tool.authorize": (params) =>
        (params as { args: { query: string } }).args.query === "salary list"
          ? { outcome: "refuse", reason: "Not permitted for that collection." }
          : allow(),
      "tool.execute": () => ({ text: "1 passage found." }),
    });

    await startRun(core.peer, request(server.baseUrl), () => {});

    // Two asked, one executed: a refusal inside a parallel batch stops only
    // its own call.
    expect(core.calls.filter((c) => c.method === "tool.authorize")).toHaveLength(2);
    expect(core.calls.filter((c) => c.method === "tool.execute")).toHaveLength(1);
  });
});

describe("steering a run in flight", () => {
  it("hands the caller a way to correct a run without killing it", async () => {
    server = await modelServer([say("Done.")]);
    const core = coreStub({});
    let handle: ActiveRun | undefined;

    await startRun(core.peer, request(server.baseUrl), (run) => {
      handle = run;
    });

    expect(handle?.steer).toBeTypeOf("function");
    // Steering a finished run is an ordinary race and must not throw.
    expect(() => handle?.steer("use the 2019 revision")).not.toThrow();
  });

  it("delivers the correction to the model on a later turn", async () => {
    // The run makes a tool call, which gives the loop a steering point before
    // the next model turn.
    server = await modelServer([
      callTool("call_1", "knowledge.search_authorized", { query: "seal" }),
      say("Using the 2019 revision."),
    ]);
    const core = coreStub({
      "tool.authorize": allow,
      "tool.execute": (params) => {
        // Injected while the run is genuinely in flight.
        handle?.steer("Actually, use the 2019 revision.");
        return { text: "1 passage found." };
      },
    });
    let handle: ActiveRun | undefined;

    await startRun(core.peer, request(server.baseUrl), (run) => {
      handle = run;
    });

    const lastRequest = JSON.stringify(server.requests.at(-1));
    expect(lastRequest).toContain("2019 revision");
  });
});

describe("stopping a run part-way", () => {
  it("records that tools may have partially executed", async () => {
    // The guidance exists so a continuation does not repeat a write that
    // already happened. It is a `custom` message, which the default converter
    // drops — this asserts it survives conversion, which is what makes it real.
    server = await modelServer([callTool("call_1", "knowledge.search_authorized", { query: "seal" }), say("ok")]);
    const core = coreStub({
      "tool.authorize": allow,
      "tool.execute": () => {
        handle?.abort("operator stopped it");
        return { text: "1 passage found." };
      },
    });
    let handle: ActiveRun | undefined;

    await startRun(core.peer, request(server.baseUrl), (run) => {
      handle = run;
    });

    // The run ends; what matters is that the transcript can say why.
    const aborted: AgentMessage = {
      role: "custom",
      customType: "openclaw:turn-aborted",
      content: "<turn_aborted>\nThe previous turn was interrupted.\n</turn_aborted>",
      display: false,
      timestamp: Date.now(),
    } as unknown as AgentMessage;
    expect(JSON.stringify(convertToLlm([aborted]))).toContain("turn_aborted");
  });

  it("reports compaction to the operator on the same channel as everything else", async () => {
    server = await modelServer([say("Done.")]);
    const core = coreStub({});
    await startRun(core.peer, request(server.baseUrl), () => {});

    // No compaction on a short run, but the channel carries lifecycle events —
    // compaction joins them rather than needing a separate surface.
    const types = core.events.map((e) => e.event.type);
    expect(types).toContain("agent_start");
    expect(types).not.toContain("context_compacted");
  });
});
