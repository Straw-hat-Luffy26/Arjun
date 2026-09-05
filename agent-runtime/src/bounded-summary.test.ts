import { describe, expect, it } from "vitest";
import type { AgentCoreCompletionRuntimeDeps, AgentMessage } from "@openclaw/agent-core";
import type { Model } from "@openclaw/ai";
import { generateBoundedSummary } from "./bounded-summary.js";
import { inputBudget, projectedTokens } from "./context-budget.js";

const model = { id: "local", contextWindow: 8192, maxTokens: 2048 } as Model;
describe("bounded summary requests", () => {
  it("chunks oversized history without sending an oversized summarizer request", async () => {
    const messages = [{ role: "user", content: "A-17 https://example.invalid/spec#A-17 " + "large output ".repeat(4000), timestamp: 0 }] as AgentMessage[];
    const original = JSON.stringify(messages);
    let calls = 0;
    const runtime: AgentCoreCompletionRuntimeDeps = { completeSimple: async (_model, context) => {
      calls += 1;
      expect(projectedTokens(context.messages as AgentMessage[]) + projectedTokens([{ role: "user", content: context.systemPrompt ?? "", timestamp: 0 }])).toBeLessThan(inputBudget(8192, 2048));
      return { role: "assistant", content: [{ type: "text", text: "Next: inspect source A-17 at https://example.invalid/spec#A-17." }], stopReason: "stop" } as never;
    } };
    const summary = await generateBoundedSummary({ model, messages, runtime, apiKey: "local" });
    expect(calls).toBeGreaterThan(1);
    expect(calls).toBeLessThanOrEqual(32);
    expect(summary).toContain("https://example.invalid/spec#A-17");
    expect(JSON.stringify(messages)).toBe(original);
  });

  it("stops when a provider exceeds the summary output bound", async () => {
    const runtime: AgentCoreCompletionRuntimeDeps = { completeSimple: async () => ({ role: "assistant", content: [{ type: "text", text: "x".repeat(60000) }], stopReason: "stop" } as never) };
    await expect(generateBoundedSummary({ model, runtime, messages: [{ role: "user", content: "hello", timestamp: 0 }], apiKey: "local" })).rejects.toThrow(/output allowance/);
  });
});
