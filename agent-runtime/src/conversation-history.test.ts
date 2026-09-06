/**
 * The conversation a fresh turn starts from.
 *
 * Two properties are under test, and they are the two the audit named as the
 * ways a naive fix goes wrong:
 *
 *  1. **The new question is asked exactly once.** Prior turns are seeded as the
 *     loop's starting transcript and the question goes through `agent.prompt`.
 *     One append, one question — never both, and never neither.
 *  2. **Generation always happens.** A seeded transcript ending on an assistant
 *     answer must not let the loop conclude the work is done and return without
 *     calling the model.
 *
 * Both are asserted against what the *model server actually received*, because
 * that is the only place the question can be answered. Asserting on what this
 * side thinks it sent would pass for a run that assembled the messages
 * correctly and then failed to send them, which is the failure being fixed.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { seedMessages, startRun, type RunRequest } from "./run.js";

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

/** The messages one recorded request carried, in order. */
function messagesOf(request: unknown): { role: string; content: unknown }[] {
  const body = request as { messages?: { role: string; content: unknown }[] };
  return body.messages ?? [];
}

/** Every text block of a recorded message, flattened to one string. */
function textOf(message: { content: unknown }): string {
  if (typeof message.content === "string") return message.content;
  if (!Array.isArray(message.content)) return "";
  return message.content
    .map((block) =>
      typeof block === "object" &&
      block !== null &&
      typeof (block as { text?: unknown }).text === "string"
        ? (block as { text: string }).text
        : "",
    )
    .join("");
}

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

/** A core stub that offers no tools, so a turn is one model call. */
function coreStub() {
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);
  peer.request = ((method: string) =>
    method === "tool.catalogue"
      ? Promise.resolve({ tools: [], mode: "Work" })
      : Promise.reject(new Error(`core stub has no ${method}`))) as RpcPeer["request"];
  peer.notify = (() => {}) as RpcPeer["notify"];
  return peer;
}

function request(baseUrl: string, over: Partial<RunRequest> = {}): RunRequest {
  return {
    runId: "run-1",
    messageId: "msg-1",
    prompt: "And what about the flange?",
    systemPrompt: "Answer from what you were given.",
    model: { id: "test-model", provider: "sovereign-local", baseUrl, maxTokens: 256 },
    ...over,
  };
}

/** A single-turn answer, so a run is exactly one model call. */
const ANSWERS = [
  [chunk({ role: "assistant", content: "" }), chunk({ content: "Class 300." }), chunk({}, "stop")],
];

let server: Awaited<ReturnType<typeof modelServer>> | undefined;

afterEach(async () => {
  await server?.close();
  server = undefined;
});

describe("a turn that continues a conversation", () => {
  it("sends the earlier messages to the model", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub(),
      request(server.baseUrl, {
        history: [
          { role: "user", content: "What is the rating?" },
          { role: "assistant", content: "Class 300 throughout." },
        ],
      }),
      () => {},
    );

    // The whole point. Before this existed, the first request a second turn
    // made carried the new question and nothing else.
    const sent = messagesOf(server.requests[0]).map(textOf).join("\n");
    expect(sent).toContain("What is the rating?");
    expect(sent).toContain("Class 300 throughout.");
  });

  it("asks the new question exactly once", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub(),
      request(server.baseUrl, {
        prompt: "And what about the flange?",
        history: [
          { role: "user", content: "What is the rating?" },
          { role: "assistant", content: "Class 300 throughout." },
        ],
      }),
      () => {},
    );

    const asked = messagesOf(server.requests[0]).filter((message) =>
      textOf(message).includes("And what about the flange?"),
    );
    expect(asked).toHaveLength(1);
  });

  it("puts the new question last, after the history", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub(),
      request(server.baseUrl, {
        prompt: "And what about the flange?",
        history: [
          { role: "user", content: "What is the rating?" },
          { role: "assistant", content: "Class 300 throughout." },
        ],
      }),
      () => {},
    );

    const messages = messagesOf(server.requests[0]);
    const last = messages[messages.length - 1]!;
    expect(last.role).toBe("user");
    expect(textOf(last)).toContain("And what about the flange?");
  });

  /**
   * The failure mode the audit named first: a transcript ending on an assistant
   * answer, read by the loop as work already done.
   */
  it("still calls the model when the history ends on an assistant answer", async () => {
    server = await modelServer(ANSWERS);
    const outcome = await startRun(
      coreStub(),
      request(server.baseUrl, {
        history: [
          { role: "user", content: "What is the rating?" },
          { role: "assistant", content: "Class 300 throughout." },
        ],
      }),
      () => {},
    );

    expect(server.requests.length).toBeGreaterThan(0);
    expect(outcome.text).toContain("Class 300.");
    expect(outcome.outcome.kind).toBe("completed");
  });

  it("keeps the roles the conversation had", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub(),
      request(server.baseUrl, {
        history: [
          { role: "user", content: "first question" },
          { role: "assistant", content: "first answer" },
          { role: "user", content: "second question" },
          { role: "assistant", content: "second answer" },
        ],
      }),
      () => {},
    );

    const messages = messagesOf(server.requests[0]).filter((m) => m.role !== "system");
    const conversation = messages.map((m) => `${m.role}:${textOf(m)}`);
    expect(conversation).toEqual([
      "user:first question",
      "assistant:first answer",
      "user:second question",
      "assistant:second answer",
      "user:And what about the flange?",
    ]);
  });

  it("behaves exactly as before when there is no history", async () => {
    server = await modelServer(ANSWERS);
    await startRun(coreStub(), request(server.baseUrl, { prompt: "The only question." }), () => {});

    const messages = messagesOf(server.requests[0]).filter((m) => m.role !== "system");
    expect(messages).toHaveLength(1);
    expect(textOf(messages[0]!)).toContain("The only question.");
  });

  it("says so when the window could not hold the whole conversation", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub(),
      request(server.baseUrl, {
        history: [{ role: "assistant", content: "the part that fitted" }],
        historyDropped: 4,
      }),
      () => {},
    );

    // A model handed a truncated conversation with nothing marking the
    // truncation answers as though it remembers the whole thing.
    const sent = messagesOf(server.requests[0]).map(textOf).join("\n");
    expect(sent).toContain("4 earlier message(s)");
    expect(sent).toContain("do not have it");
  });
});

describe("seedMessages", () => {
  const model = {
    id: "test-model",
    api: "openai-completions",
    provider: "sovereign-local",
  } as const;

  it("carries nothing for a first turn", () => {
    expect(seedMessages(undefined, model)).toEqual([]);
    expect(seedMessages([], model)).toEqual([]);
  });

  it("drops an entry whose role the loop does not understand", () => {
    const seeded = seedMessages(
      [
        { role: "user", content: "kept" },
        // A role from a future schema, or a malformed frame. Dropped rather
        // than rewritten to `user`: putting the model's words in the person's
        // mouth is worse than a shorter history.
        { role: "toolResult", content: "not a conversation turn" } as never,
        { role: "assistant", content: "also kept" },
      ],
      model,
    );
    expect(seeded).toHaveLength(2);
    expect(seeded.map((m) => m.role)).toEqual(["user", "assistant"]);
  });

  it("drops an empty or whitespace-only turn", () => {
    const seeded = seedMessages(
      [
        { role: "user", content: "   " },
        { role: "assistant", content: "" },
        { role: "user", content: "real" },
      ],
      model,
    );
    expect(seeded).toHaveLength(1);
  });

  it("orders the seeded messages by their timestamps", () => {
    const seeded = seedMessages(
      [
        { role: "user", content: "one" },
        { role: "assistant", content: "two" },
        { role: "user", content: "three" },
      ],
      model,
    );
    const stamps = seeded.map((m) => (m as { timestamp: number }).timestamp);
    expect([...stamps].sort((a, b) => a - b)).toEqual(stamps);
  });

  /**
   * A replayed assistant turn is not a measurement, and must not be reported as
   * one. The context meter shows what a turn cost; a fabricated count here
   * would appear there as tokens somebody was charged.
   */
  it("reports no usage for a replayed assistant turn", () => {
    const [seeded] = seedMessages([{ role: "assistant", content: "answered earlier" }], model);
    const usage = (seeded as { usage?: { input: number; output: number; totalTokens: number } })
      .usage;
    expect(usage).toEqual(expect.objectContaining({ input: 0, output: 0, totalTokens: 0 }));
  });

  it("marks a replayed assistant turn as one that finished", () => {
    const [seeded] = seedMessages([{ role: "assistant", content: "answered earlier" }], model);
    // Only turns that completed are eligible on the Rust side, so anything
    // else here would tell the model the thread is full of failed attempts.
    expect((seeded as { stopReason?: string }).stopReason).toBe("stop");
  });

  it("adds the truncation marker only when something was dropped", () => {
    const none = seedMessages([{ role: "user", content: "q" }], model, 0);
    expect(none).toHaveLength(1);

    const some = seedMessages([{ role: "user", content: "q" }], model, 3);
    expect(some).toHaveLength(2);
    expect((some[0] as { content: { text: string }[] }).content[0]!.text).toContain(
      "3 earlier message(s)",
    );
  });
});
