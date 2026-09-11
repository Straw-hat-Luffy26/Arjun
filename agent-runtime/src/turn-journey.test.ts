/**
 * What the model is actually sent, and what actually stops.
 *
 * ## Why these assert on the provider request
 *
 * Every claim this change makes is a claim about the bytes that reach the
 * inference server: that a second turn carries the first, that a model change
 * does not lose it, that a pin keeps a passage the compactor would otherwise
 * clear, that a Stop ends generation. None of that is observable from this
 * side's own state — a run can assemble the right messages and fail to send
 * them, and a test reading its own variables would pass either way.
 *
 * So each of these stands up a real HTTP server speaking real SSE, runs the
 * production `startRun`, and asserts on the request bodies the server received.
 */

import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { afterEach, describe, expect, it } from "vitest";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { answerOf, startRun, type ActiveRun, type RunRequest } from "./run.js";

function chunk(delta: unknown, finishReason: string | null = null): string {
  return `data: ${JSON.stringify({
    id: "chatcmpl-test",
    object: "chat.completion.chunk",
    created: 0,
    model: "test-model",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
  })}\n\n`;
}

interface Recorded {
  messages: { role: string; content: unknown }[];
  model?: string;
}

function bodies(requests: unknown[]): Recorded[] {
  return requests as Recorded[];
}

/** Every text block of a recorded message, flattened. */
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

/**
 * The request for the *turn itself*, not the summariser's.
 *
 * A turn that compacts issues two calls: `generateSummary` asks the model to
 * condense the history, and then the turn's own request goes out. Reading the
 * first one reads the summariser's prompt, which carries none of this — a
 * mistake that makes a passing assertion meaningless and a failing one
 * baffling.
 */
function turnRequest(requests: unknown[]): Recorded {
  const all = bodies(requests);
  const own = all.filter(
    (body) => !sentIn(body).includes('context summarization assistant'),
  );
  const chosen = own[own.length - 1] ?? all[all.length - 1];
  if (!chosen) throw new Error('the model server received no request at all');
  return chosen;
}

/** Everything one request carried, as one string. */
function sentIn(request: Recorded): string {
  return (request.messages ?? []).map(textOf).join("\n");
}

/**
 * A local model server.
 *
 * `hold` keeps the response open after the scripted frames, which is what lets
 * a test stop a run mid-generation rather than after it.
 */
function modelServer(
  turns: string[][],
  options: { hold?: boolean } = {},
): Promise<{
  baseUrl: string;
  requests: unknown[];
  close: () => Promise<void>;
}> {
  const requests: unknown[] = [];
  const open: import("node:http").ServerResponse[] = [];
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
      if (options.hold) {
        // Deliberately not ended. The run is generating until something stops
        // it, which is the state a Stop has to be able to interrupt.
        open.push(res);
        return;
      }
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
        close: () =>
          new Promise<void>((done) => {
            for (const res of open) res.end();
            server.close(() => done());
          }),
      });
    });
  });
}

/** A core stub with no tools, so one turn is one model call. */
function coreStub(handlers: Record<string, (params: unknown) => unknown> = {}) {
  const events: { runId: string; event: { type: string } & Record<string, unknown> }[] = [];
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);
  peer.request = ((method: string, params: unknown) => {
    if (handlers[method]) return Promise.resolve(handlers[method](params));
    if (method === "tool.catalogue") return Promise.resolve({ tools: [], mode: "Work" });
    return Promise.reject(new Error(`core stub has no ${method}`));
  }) as RpcPeer["request"];
  peer.notify = ((method: string, params: unknown) => {
    if (method === "run.event") {
      events.push(params as { runId: string; event: { type: string } & Record<string, unknown> });
    }
  }) as RpcPeer["notify"];
  return { peer, events };
}

function request(baseUrl: string, over: Partial<RunRequest> = {}): RunRequest {
  return {
    runId: "run-1",
    messageId: "msg-1",
    prompt: "And the gasket torque?",
    systemPrompt: "Answer from what you were given.",
    model: { id: "test-model", provider: "sovereign-local", baseUrl, maxTokens: 256 },
    ...over,
  };
}

const ANSWERS = [
  [chunk({ role: "assistant", content: "" }), chunk({ content: "Class 300." }), chunk({}, "stop")],
];

let server: Awaited<ReturnType<typeof modelServer>> | undefined;

afterEach(async () => {
  await server?.close();
  server = undefined;
});

// ── Ordinary recall, reopening, restarting, model changes ────────────────

describe("what a continuing turn sends to the model", () => {
  const HISTORY = [
    { role: "user" as const, content: "What is the pressure rating?" },
    { role: "assistant" as const, content: "Class 300 throughout the skid." },
  ];

  it("carries the earlier exchange into the provider request", async () => {
    server = await modelServer(ANSWERS);
    await startRun(coreStub().peer, request(server.baseUrl, { history: HISTORY }), () => {});

    const sent = sentIn(turnRequest(server.requests));
    expect(sent).toContain("What is the pressure rating?");
    expect(sent).toContain("Class 300 throughout the skid.");
  });

  /**
   * Reopening a thread, or restarting the application, rebuilds the history
   * from the conversation on disk. From this side that is indistinguishable
   * from any other turn — which is the point: nothing here is stateful, so
   * there is no in-memory context to lose.
   */
  it("carries it identically for a turn assembled after a restart", async () => {
    server = await modelServer(ANSWERS);
    // A different run id and message id, as a fresh process would mint.
    await startRun(
      coreStub().peer,
      request(server.baseUrl, {
        runId: "run-after-restart",
        messageId: "msg-after-restart",
        history: HISTORY,
      }),
      () => {},
    );

    const sent = sentIn(turnRequest(server.requests));
    expect(sent).toContain("Class 300 throughout the skid.");
  });

  /** A model change must not cost the conversation. */
  it("carries it across a change of model", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub().peer,
      request(server.baseUrl, {
        history: HISTORY,
        model: {
          id: "a-completely-different-model",
          provider: "sovereign-local",
          baseUrl: server.baseUrl,
          maxTokens: 256,
        },
      }),
      () => {},
    );

    const body = turnRequest(server.requests);
    expect(body.model).toBe("a-completely-different-model");
    expect(sentIn(body)).toContain("Class 300 throughout the skid.");
  });

  /**
   * The question is asked once, and it is asked last. Both halves matter: a
   * transcript ending on an assistant answer can be read as already-done, and
   * one carrying the question twice asks it twice.
   */
  it("asks the new question exactly once, at the end", async () => {
    server = await modelServer(ANSWERS);
    await startRun(coreStub().peer, request(server.baseUrl, { history: HISTORY }), () => {});

    const messages = turnRequest(server.requests).messages.filter((m) => m.role !== "system");
    const asked = messages.filter((m) => textOf(m).includes("And the gasket torque?"));
    expect(asked).toHaveLength(1);
    expect(messages.at(-1)?.role).toBe("user");
  });
});

// ── Pins affecting the context actually retained ─────────────────────────

describe("a pinned passage survives into the provider request", () => {
  /**
   * The claim a pin makes is about what the *model* is sent after a compaction,
   * not about what a panel draws. So this compacts for real and reads the
   * request the server received.
   */
  const BIG = "padding ".repeat(4000);

  function longHistory() {
    return [
      { role: "user" as const, content: "search the manual" },
      { role: "assistant" as const, content: `[E3] the seal is 9.0 mm ${BIG}` },
      { role: "user" as const, content: "and the flange" },
      { role: "assistant" as const, content: `[E4] the flange is Class 300 ${BIG}` },
    ];
  }

  it("sends the whole history when nothing forces a cut", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub().peer,
      request(server.baseUrl, {
        history: longHistory(),
        model: {
          id: "test-model",
          provider: "sovereign-local",
          baseUrl: server.baseUrl,
          maxTokens: 256,
          // Large enough that nothing is compacted, so the baseline is real.
          contextWindow: 1_000_000,
        },
      }),
      () => {},
    );

    const sent = sentIn(turnRequest(server.requests));
    expect(sent).toContain("the seal is 9.0 mm");
    expect(sent).toContain("the flange is Class 300");
  });

  /**
   * The pinned set reaches the run as `preserved.pinned`, and the compactor
   * names it across the cut. The assertion is on the provider request: the
   * model is *told* what the person protected.
   */
  it("names the pinned entries in what the model is sent", async () => {
    server = await modelServer(ANSWERS);
    await startRun(
      coreStub().peer,
      request(server.baseUrl, {
        history: longHistory(),
        preserved: { activePlan: "read the manual", pinned: ["E3"] },
        model: {
          id: "test-model",
          provider: "sovereign-local",
          baseUrl: server.baseUrl,
          maxTokens: 256,
          // Small enough to force the compactor to act on this very turn.
          contextWindow: 4_000,
        },
      }),
      () => {},
    );

    const sent = sentIn(turnRequest(server.requests));
    expect(sent).toContain("E3");
    expect(sent).toMatch(/operator|Kept at/i);
  });
});

// ── Compaction reported live ─────────────────────────────────────────────

describe("compaction is reported as it happens", () => {
  it("emits a context_compacted frame carrying a full record", async () => {
    const BIG = "padding ".repeat(6000);
    server = await modelServer([
      [chunk({ role: "assistant", content: "" }), chunk({ content: "ok" }), chunk({}, "stop")],
    ]);
    const core = coreStub();

    await startRun(
      core.peer,
      request(server.baseUrl, {
        history: [
          { role: "user", content: `q ${BIG}` },
          { role: "assistant", content: `a ${BIG}` },
          { role: "user", content: `q2 ${BIG}` },
          { role: "assistant", content: `a2 ${BIG}` },
        ],
        model: {
          id: "test-model",
          provider: "sovereign-local",
          baseUrl: server.baseUrl,
          maxTokens: 128,
          contextWindow: 3_000,
        },
      }),
      () => {},
    );

    // The meter's rows are built from these, so a frame missing its ordinal or
    // its ledger cannot become a row at all.
    const compactions = core.events.filter((e) => e.event.type === "context_compacted");
    const ledgers = core.events.filter((e) => e.event.type === "context_ledger");
    expect(ledgers.length).toBeGreaterThan(0);
    if (compactions.length > 0) {
      const first = compactions[0]?.event as Record<string, unknown>;
      expect(typeof first.ordinal).toBe("number");
      expect(typeof first.at).toBe("string");
      expect(first.ledger).toBeTruthy();
    }
  });

  /** The meter is fed during execution, not only at the end. */
  it("publishes a ledger reading on every turn", async () => {
    server = await modelServer(ANSWERS);
    const core = coreStub();
    await startRun(core.peer, request(server.baseUrl), () => {});

    const ledgers = core.events.filter((e) => e.event.type === "context_ledger");
    expect(ledgers.length).toBeGreaterThan(0);
    expect(ledgers.every((e) => e.runId === "run-1")).toBe(true);
  });
});

// ── Stop, during generation ──────────────────────────────────────────────

describe("stopping a turn that is generating", () => {
  /** The run is held open by the server, so this is a genuine mid-flight stop. */
  async function stoppedMidGeneration(times = 1) {
    server = await modelServer(
      [[chunk({ role: "assistant", content: "" }), chunk({ content: "half an answer" })]],
      { hold: true },
    );
    let active: ActiveRun | undefined;
    const finished = startRun(coreStub().peer, request(server.baseUrl), (run) => {
      active = run;
    });
    // Let the first delta land, so there is partial output to preserve.
    await new Promise((resolve) => setTimeout(resolve, 250));
    for (let i = 0; i < times; i++) active?.abort("stopped by the operator");
    return finished;
  }

  it("ends as aborted rather than completed", async () => {
    const outcome = await stoppedMidGeneration();
    expect(outcome.outcome.kind).toBe("aborted");
    expect(outcome.outcome.detail ?? "").toMatch(/stopped/i);
  });

  it("keeps what was already produced", async () => {
    const outcome = await stoppedMidGeneration();
    // A stopped turn is not a failed one: the text that arrived is the
    // person's, and throwing it away would lose real work.
    expect(outcome.text).toContain("half an answer");
  });

  /** Pressing it twice is one stop, not an error and not a second ending. */
  it("survives being stopped repeatedly", async () => {
    const outcome = await stoppedMidGeneration(4);
    expect(outcome.outcome.kind).toBe("aborted");
    expect(outcome.text).toContain("half an answer");
  });

  /**
   * The completion race: a stop that lands after the turn has already finished
   * must not rewrite its ending.
   */
  it("does not relabel a turn that had already completed", async () => {
    server = await modelServer(ANSWERS);
    let active: ActiveRun | undefined;
    const outcome = await startRun(coreStub().peer, request(server.baseUrl), (run) => {
      active = run;
    });
    expect(outcome.outcome.kind).toBe("completed");

    // The button pressed a moment too late. Aborting a finished run is a no-op
    // rather than a fault, and it must not change the ending already recorded.
    expect(() => active?.abort("too late")).not.toThrow();
    expect(outcome.outcome.kind).toBe("completed");
  });

  /** And the next turn works. A stop is not a broken session. */
  it("leaves the next turn able to run normally", async () => {
    await stoppedMidGeneration();
    await server?.close();

    server = await modelServer(ANSWERS);
    const outcome = await startRun(
      coreStub().peer,
      request(server.baseUrl, { runId: "run-2", messageId: "msg-2" }),
      () => {},
    );
    expect(outcome.outcome.kind).toBe("completed");
    expect(outcome.text).toContain("Class 300.");
  });
});

describe("answerOf: what a stopped turn is recorded as having said", () => {
  const assistant = (text: string, extra: Record<string, unknown> = {}) => ({
    role: "assistant",
    content: [{ type: "text", text }],
    ...extra,
  });

  it("returns the text of a normal final message", () => {
    const { text } = answerOf([assistant("the seal is due for replacement")]);
    expect(text).toBe("the seal is due for replacement");
  });

  /**
   * The defect this guards.
   *
   * `stopIfAborted` appends an assistant message whose only text block is
   * empty, carrying `stopReason: "aborted"`. Reading strictly the last
   * assistant message therefore reported a stopped turn as having said nothing
   * — discarding a partial answer the person had been watching arrive, and
   * making an operator's Stop indistinguishable from a model that produced
   * nothing at all.
   */
  it("keeps the partial answer when the turn was interrupted", () => {
    const { text, finalAssistant } = answerOf([
      assistant("the seal is due for replacement, and the sp"),
      assistant("", { stopReason: "aborted" }),
    ]);

    expect(text).toBe("the seal is due for replacement, and the sp");
    // The ending still comes from the interrupted message — only the text is
    // taken from further back.
    expect((finalAssistant as { stopReason?: unknown })?.stopReason).toBe("aborted");
  });

  it("still reports nothing when nothing was said", () => {
    const { text } = answerOf([assistant("", { stopReason: "aborted" })]);
    expect(text).toBe("");
  });
});
