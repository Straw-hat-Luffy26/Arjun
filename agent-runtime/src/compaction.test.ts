import { describe, expect, it, vi } from "vitest";
import { convertToLlm, estimateContextTokens, type AgentMessage } from "@openclaw/agent-core";
import type { Model } from "@openclaw/ai";
import { RunCompactor, settingsForWindow } from "./compaction.js";

/** A local model with a small window, which is the case that matters. */
function model(contextWindow: number): Model {
  return {
    id: "qwen2.5-coder-7b",
    name: "Qwen2.5 Coder 7B",
    api: "openai-completions",
    provider: "llama-cpp",
    baseUrl: "http://127.0.0.1:8080/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: 2048,
  } as Model;
}

function user(text: string): AgentMessage {
  return { role: "user", content: [{ type: "text", text }], timestamp: 1 } as AgentMessage;
}

function assistant(text: string): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: "openai-completions",
    provider: "llama-cpp",
    model: "qwen2.5-coder-7b",
    stopReason: "stop",
    timestamp: 1,
  } as unknown as AgentMessage;
}

/** A transcript long enough to overflow the given window. */
function longTranscript(pairs: number, charsEach: number): AgentMessage[] {
  const messages: AgentMessage[] = [];
  for (let i = 0; i < pairs; i++) {
    messages.push(user(`question ${i} ${"x".repeat(charsEach)}`));
    messages.push(assistant(`answer ${i} ${"y".repeat(charsEach)}`));
  }
  return messages;
}

/** A summariser that answers without a model server. */
function summariser(text = "Earlier: the operator asked about pump seals.") {
  return {
    completeSimple: vi.fn(async () => ({
      role: "assistant",
      content: [{ type: "text", text }],
      api: "openai-completions",
      provider: "llama-cpp",
      model: "qwen2.5-coder-7b",
      stopReason: "stop",
      timestamp: 1,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    })),
  } as never;
}

describe("settingsForWindow", () => {
  it("scales to the window instead of demanding more than the model has", () => {
    // Upstream's fixed defaults reserve 16k and keep 20k — both larger than an
    // 8k window entirely, which would make compaction impossible exactly where
    // it is needed most.
    const small = settingsForWindow(8_192);
    expect(small.reserveTokens).toBeLessThan(8_192);
    expect(small.keepRecentTokens).toBeLessThan(8_192);
    expect(small.reserveTokens + small.keepRecentTokens).toBeLessThan(8_192);
    expect(small.enabled).toBe(true);
  });

  it("stays proportionate on a large window too", () => {
    const large = settingsForWindow(200_000);
    expect(large.reserveTokens).toBe(40_000);
    expect(large.keepRecentTokens).toBe(80_000);
  });

  it("disables itself when the window is unknown rather than guessing", () => {
    for (const window of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      expect(settingsForWindow(window).enabled).toBe(false);
    }
  });
});

/**
 * The prefix is charged once, not twice.
 *
 * `estimateContextTokens` has two modes: with no provider usage to read it sums
 * the messages, and the system prompt and tool schemas are genuinely not in
 * that figure — so the ledger's `fixed()` has to be added. Once a turn has
 * completed it short-circuits to the provider's own `usage.input`, which
 * counted the *whole* request and already contains them.
 *
 * Adding `fixed()` in both cases inflated every turn after the first by the
 * size of the catalogue plus the system prompt. The run compacted earlier than
 * it needed to, and the ceiling pass dropped history that would have fitted —
 * on a small window with a full catalogue, thousands of tokens of it.
 */
describe("the fixed prefix is charged once", () => {
  /** An assistant turn carrying a provider usage record, as a real one does. */
  function answered(text: string, inputTokens: number): AgentMessage {
    return {
      role: "assistant",
      content: [{ type: "text", text }],
      api: "openai-completions",
      provider: "llama-cpp",
      model: "qwen2.5-coder-7b",
      stopReason: "stop",
      timestamp: 1,
      usage: {
        input: inputTokens,
        output: 8,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: inputTokens + 8,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    } as unknown as AgentMessage;
  }

  it("does not compact a turn the provider says already fits", async () => {
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
    });
    // ~3,000 tokens of prefix: a tool catalogue and a system prompt, booked the
    // way a real run books them.
    compactor.ledger.setText("system", "s".repeat(6_000));
    compactor.ledger.setText("toolSchema", "t".repeat(6_000));

    // Long enough that there *is* something older than `keepRecentTokens` to
    // summarise — otherwise nothing would compact whatever the arithmetic said,
    // and the test would pass for the wrong reason.
    //
    // The final turn carries the provider's own count for the whole request,
    // prefix included: 4,000 against a trigger at 8,192 − 1,638 = 6,554, which
    // fits with room to spare. Adding the prefix a second time makes it 7,000
    // and compacts a turn that did not need it.
    const messages = [...longTranscript(20, 400), answered("short answer", 4_000)];

    await compactor.transform(messages);

    expect(compactor.compactions).toBe(0);
  });

  it("still adds the prefix before any turn has completed", async () => {
    // The other mode, and the reason `fixed()` is added at all: with no usage
    // to read, the estimate is messages only. An 8k window told that 2,700
    // tokens of conversation fitted, while the request around it came to 9,238,
    // compacted nothing and died at the provider.
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
    });
    // ~3,000 tokens of prefix.
    compactor.ledger.setText("system", "s".repeat(6_000));
    compactor.ledger.setText("toolSchema", "t".repeat(6_000));

    // ~4,000 tokens of conversation: comfortably under the 6,554 trigger on its
    // own, and over it once the prefix is added. Long enough that there is
    // something older than `keepRecentTokens` to summarise, which is what makes
    // this a compaction rather than a ceiling trim.
    //
    // No assistant turn carries usage here, so the estimator is in its
    // messages-only mode and the prefix genuinely has to be added.
    const messages = longTranscript(20, 400);

    await compactor.transform(messages);

    expect(compactor.compactions).toBe(1);
  });
});

describe("a run that outgrows its window", () => {
  it("leaves a short transcript completely alone", async () => {
    const runtime = summariser();
    const compactor = new RunCompactor({
      model: model(32_768),
      runtime,
      apiKey: "local",
    });
    const messages = [user("hello"), assistant("hi")];

    expect(await compactor.transform(messages)).toEqual(messages);
    expect(compactor.compactions).toBe(0);
  });

  it("compacts rather than letting the request exceed the window", async () => {
    // This is the regression the Rust engine currently fails: a prompt at or
    // over the window is refused outright, so a long run stops instead of
    // degrading.
    const runtime = summariser();
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime,
      apiKey: "local",
    });
    const messages = longTranscript(40, 800);

    const projected = await compactor.transform(messages);

    expect(compactor.compactions).toBe(1);
    expect(projected.length).toBeLessThan(messages.length);
    // The transcript itself is untouched — the audit record keeps everything.
    expect(messages).toHaveLength(80);
  });

  it("puts the summary where the model will actually read it", async () => {
    const runtime = summariser("Earlier: pump seal specification was 9.0 mm.");
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime,
      apiKey: "local",
    });

    const projected = await compactor.transform(longTranscript(40, 800));
    const llm = convertToLlm(projected);

    // Converted, not merely present: the default converter would have dropped
    // the summary silently, which is the failure this guards.
    expect(JSON.stringify(llm)).toContain("9.0 mm");
  });

  it("reports what it did, so an operator is not surprised by a shorter context", async () => {
    const seen: unknown[] = [];
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
      onCompacted: (event) => seen.push(event),
    });

    await compactor.transform(longTranscript(40, 800));

    expect(seen).toHaveLength(1);
    const event = seen[0] as { tokensBefore: number; tokensAfter: number; messagesSummarised: number };
    expect(event.tokensAfter).toBeLessThan(event.tokensBefore);
    expect(event.messagesSummarised).toBeGreaterThan(0);
  });

  it("does not compact again on the next turn just because the transcript is still long", async () => {
    // The trap: measuring the raw transcript rather than what is actually sent
    // means every subsequent turn looks over budget and re-summarises forever.
    const runtime = summariser();
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime,
      apiKey: "local",
    });
    const messages = longTranscript(40, 800);

    await compactor.transform(messages);
    const afterFirst = compactor.compactions;
    await compactor.transform(messages);

    expect(compactor.compactions).toBe(afterFirst);
  });

  it("extends the previous summary rather than summarising a summary", async () => {
    const runtime = summariser();
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime,
      apiKey: "local",
    });

    await compactor.transform(longTranscript(40, 800));
    await compactor.transform(longTranscript(90, 800));

    expect(compactor.compactions).toBe(2);
    // The second call must carry the first summary forward, so the prompt is an
    // update rather than a fresh summarisation of already-summarised text.
    const prompts = (runtime as unknown as { completeSimple: { mock: { calls: unknown[][] } } })
      .completeSimple.mock.calls;
    expect(JSON.stringify(prompts[1])).toContain("previous-summary");
  });

  it("still makes the context fit when summarisation fails", async () => {
    // A failed summary must not fail the task, and it must not be allowed to
    // send an over-long request either.
    //
    // This used to return the transcript untouched and leave the size to the
    // provider, on the reasoning that its refusal named the problem better than
    // a summary of nothing would. It named it accurately and uselessly: the
    // person saw `400 request (…) exceeds the available context size` and had
    // no action available, and a model server that cannot write a summary is in
    // no better position to accept an over-long request than to shorten one. So
    // the ceiling pass drops messages instead — mechanically, needing nothing
    // from the model that has just declined to answer.
    const failing = {
      completeSimple: vi.fn(async () => {
        throw new Error("the summariser is unavailable");
      }),
    } as never;
    const window = 8_192;
    const compactor = new RunCompactor({
      model: model(window),
      runtime: failing,
      apiKey: "local",
    });
    const messages = longTranscript(40, 800);

    const projected = await compactor.transform(messages);

    expect(compactor.compactions).toBe(0);
    // Shorter than what it was given, and inside what the model will accept.
    expect(projected.length).toBeLessThan(messages.length);
    expect(estimateContextTokens(projected).tokens).toBeLessThanOrEqual(
      window - settingsForWindow(window).reserveTokens,
    );
    // The question is the one message that may never be dropped: an answer to a
    // request that fits because the question was removed is an answer about
    // nothing.
    expect(projected[projected.length - 1]).toEqual(messages[messages.length - 1]);
    // And the loss is stated in the context rather than left for the model to
    // discover by contradicting itself.
    expect(JSON.stringify(projected)).toContain("Context notice");
  });

  it("never cuts between a tool call and its result", async () => {
    // Splitting the pair produces a transcript the provider rejects as
    // malformed, which surfaces as a mysterious loop failure.
    const messages: AgentMessage[] = [];
    for (let i = 0; i < 30; i++) {
      messages.push(user(`ask ${i} ${"x".repeat(600)}`));
      messages.push({
        role: "assistant",
        content: [
          { type: "text", text: "searching" },
          { type: "toolCall", id: `call_${i}`, name: "search_documents", arguments: { query: "q" } },
        ],
        api: "openai-completions",
        provider: "llama-cpp",
        model: "qwen2.5-coder-7b",
        stopReason: "toolUse",
        timestamp: 1,
      } as unknown as AgentMessage);
      messages.push({
        role: "toolResult",
        toolCallId: `call_${i}`,
        toolName: "search_documents",
        content: [{ type: "text", text: `result ${i} ${"y".repeat(600)}` }],
        isError: false,
        timestamp: 1,
      } as unknown as AgentMessage);
    }

    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
    });
    const projected = await compactor.transform(messages);

    // Every retained tool result must still have its calling assistant message.
    const kept = projected.filter((m) => m.role === "assistant" || m.role === "toolResult");
    const calledIds = new Set(
      kept
        .filter((m) => m.role === "assistant")
        .flatMap((m) =>
          (Array.isArray(m.content) ? m.content : [])
            .filter((b) => b.type === "toolCall")
            .map((b) => (b as { id: string }).id),
        ),
    );
    for (const message of kept) {
      if (message.role === "toolResult") {
        expect(calledIds.has((message as unknown as { toolCallId: string }).toolCallId)).toBe(true);
      }
    }
  });
});

/** A transcript of tool round trips, each call with its result. */
function toolTranscript(rounds: number, charsEach: number): AgentMessage[] {
  const messages: AgentMessage[] = [];
  for (let i = 0; i < rounds; i++) {
    messages.push(user(`ask ${i} ${"x".repeat(charsEach)}`));
    messages.push({
      role: "assistant",
      content: [
        { type: "text", text: "searching" },
        { type: "toolCall", id: `call_${i}`, name: "search_documents", arguments: { query: "q" } },
      ],
      api: "openai-completions",
      provider: "llama-cpp",
      model: "qwen2.5-coder-7b",
      stopReason: "toolUse",
      timestamp: 1,
    } as unknown as AgentMessage);
    messages.push({
      role: "toolResult",
      toolCallId: `call_${i}`,
      toolName: "search_documents",
      content: [{ type: "text", text: `result ${i} ${"y".repeat(charsEach)}` }],
      isError: false,
      timestamp: 1,
    } as unknown as AgentMessage);
  }
  messages.push(user("And the flange rating?"));
  return messages;
}

describe("what compaction says it stands in for", () => {
  it("names the range and the tool calls a summary replaced", async () => {
    const events: Array<{ sourceRange: { from: number; to: number; toolCalls: string[] } }> = [];
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
      preserved: () => ({ activePlan: "draft, then review" }),
      onCompacted: (event) => events.push(event),
    });
    const projected = await compactor.transform(toolTranscript(30, 600));

    expect(compactor.compactions).toBe(1);
    const range = events[0]!.sourceRange;
    expect(range.from).toBe(0);
    expect(range.to).toBeGreaterThan(0);
    expect(range.toolCalls.length).toBeGreaterThan(0);
    expect(range.toolCalls[0]).toBe("call_0");
    // Said to the model too, in the carried state beside the summary.
    const text = JSON.stringify(projected);
    expect(text).toContain(`stands in for the first ${range.to} message(s)`);
    expect(text).toContain("call_0");
    expect(text).toContain("kept in the run's record, not deleted");
  });

  it("names the tool calls the ceiling pass dropped", async () => {
    const failing = {
      completeSimple: vi.fn(async () => {
        throw new Error("the summariser is unavailable");
      }),
    } as never;
    const compactor = new RunCompactor({ model: model(8_192), runtime: failing, apiKey: "local" });
    const projected = await compactor.transform(toolTranscript(30, 600));
    const notice = JSON.stringify(projected);
    expect(notice).toContain("Context notice");
    expect(notice).toContain("The removed messages included tool calls call_0");
    // Whole groups went, rather than every message being cut to fit.
    expect(notice).not.toMatch(/\d+ messages were shortened/);
  });

  it("drops a tool call together with its results, and never orphans either", async () => {
    const failing = {
      completeSimple: vi.fn(async () => {
        throw new Error("the summariser is unavailable");
      }),
    } as never;
    const compactor = new RunCompactor({ model: model(8_192), runtime: failing, apiKey: "local" });
    const messages = toolTranscript(30, 600);
    const projected = await compactor.transform(messages);
    expect(projected.length).toBeLessThan(messages.length);
    const kept = projected.filter((m) => m.role === "assistant" || m.role === "toolResult");
    const called = new Set(
      kept.flatMap((m) =>
        m.role === "assistant" && Array.isArray(m.content)
          ? m.content.filter((b) => b.type === "toolCall").map((b) => (b as { id: string }).id)
          : [],
      ),
    );
    const answered = new Set(
      kept
        .filter((m) => m.role === "toolResult")
        .map((m) => (m as unknown as { toolCallId: string }).toolCallId),
    );
    expect([...answered].every((id) => called.has(id))).toBe(true);
    expect([...called].every((id) => answered.has(id))).toBe(true);
  });
});

describe("a compactor following a re-bound endpoint", () => {
  it("fits the next projection to the smaller window the server came back with", async () => {
    // Started against a 32k window; the server was stopped for a child and
    // came back with 8k. The compactor must stop certifying 32k-sized
    // requests the moment it is told.
    const compactor = new RunCompactor({
      model: model(32_768),
      runtime: summariser(),
      apiKey: "local",
    });
    const messages = longTranscript(20, 1_600);
    const roomy = await compactor.transform(messages);
    expect(compactor.compactions).toBe(0);
    expect(roomy.length).toBe(messages.length);

    compactor.rebind({ ...model(8_192), baseUrl: "http://127.0.0.1:41001/v1" });
    const tight = await compactor.transform(messages);
    expect(estimateContextTokens(tight).tokens).toBeLessThanOrEqual(
      8_192 - settingsForWindow(8_192, 2048).reserveTokens,
    );
    expect(compactor.ledger.snapshot().window).toBe(8_192);
  });

  it("asks the re-bound server for its summary, not the one the run started on", async () => {
    const runtime = summariser();
    const compactor = new RunCompactor({ model: model(8_192), runtime, apiKey: "local" });
    compactor.rebind({ ...model(8_192), baseUrl: "http://127.0.0.1:41002/v1" });
    await compactor.transform(longTranscript(40, 800));
    const asked = (runtime as unknown as { completeSimple: { mock: { calls: unknown[][] } } })
      .completeSimple.mock.calls[0]?.[0] as { baseUrl?: string } | undefined;
    expect(asked?.baseUrl).toBe("http://127.0.0.1:41002/v1");
  });
});
