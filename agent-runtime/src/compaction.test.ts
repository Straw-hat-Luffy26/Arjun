import { describe, expect, it, vi } from "vitest";
import { convertToLlm, type AgentMessage } from "@openclaw/agent-core";
import type { Model } from "@openclaw/ai";
import { RunCompactor, settingsForWindow, trimLargeSavedToolResults } from "./compaction.js";
import { ContextLedger } from "./context-ledger.js";

it("shortens only durably saved tool bodies and retains an exact retrieval reference", () => {
  const body="Source A-17: " + "large output ".repeat(2000);
  const saved = { role: "toolResult", toolCallId: "call-1", toolName: "workspace.read_text",
    content: [{ type: "text", text: body }], isError: false, timestamp: 1, arjunRawSeq: 3,
  } satisfies AgentMessage & { arjunRawSeq: number };
  const { arjunRawSeq: _sequence, ...unsaved } = saved;
  const trimmed=trimLargeSavedToolResults([saved,unsaved],500);
  expect(trimmed.cleared).toBe(1);
  expect(JSON.stringify(trimmed.messages[0])).toContain("transcriptSeq:3");
  expect(JSON.stringify(trimmed.messages[0]).length).toBeLessThan(1100);
  expect(JSON.stringify(saved)).toContain(body);
  expect(trimmed.messages[1]).toBe(unsaved);
});

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
      onCompacted: (event) => { seen.push(event); },
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

  it("refuses an oversized model call when summarisation fails", async () => {
    const failing = {
      completeSimple: vi.fn(async () => {
        throw new Error("the summariser is unavailable");
      }),
    } as never;
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: failing,
      apiKey: "local",
    });
    const messages = longTranscript(40, 800);

    await expect(compactor.transform(messages)).rejects.toThrow(/context budget/i);
    expect(compactor.compactions).toBe(0);
  });

  it("includes fixed system and tool schemas in admission", async () => {
    const ledger = new ContextLedger(8192);
    ledger.set("system", 5000);
    ledger.set("toolSchema", 2500);
    const runtime = summariser() as unknown as { completeSimple: ReturnType<typeof vi.fn> };
    const compactor = new RunCompactor({ model: model(8192), runtime: runtime as never, apiKey: "local", ledger });
    await expect(compactor.transform([user("hello")])).rejects.toThrow(/context budget/i);
    expect(runtime.completeSimple).not.toHaveBeenCalled();
  });

  it("refuses an unknown context limit rather than sending an unbounded request", async () => {
    const compactor = new RunCompactor({ model: model(0), runtime: summariser(), apiKey: "local" });
    await expect(compactor.transform([user("hello")])).rejects.toThrow(/context budget/i);
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
