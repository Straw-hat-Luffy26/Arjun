import { describe, expect, it } from "vitest";
import type { AgentMessage } from "@openclaw/agent-core";
import { buildDestinationContext, transitionPins, type ModelContext } from "./destination-context.js";
import { WorkingNotes } from "./working-notes.js";
import { pendingToolCalls } from "./tool-recovery.js";
import { inputBudget } from "./context-budget.js";

const destination = (window: number): ModelContext => ({ modelId: `target-${window}`, servedModelId: `target-${window}`,
  provider: "sovereign-local", contextWindow: window, maxTokens: 256, input: ["text"] });
const assistant = (content: unknown[]): AgentMessage => ({ role: "assistant", content, model: "source-8192",
  provider: "source-provider", api: "openai-completions", timestamp: 1, stopReason: "toolUse",
  usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0,
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } } }) as AgentMessage;
const call = (id: string, name = "workspace.read_text", args = { path: "DOC-REV-007.txt" }) => ({ type: "toolCall", id, name, arguments: args });
const result = (id: string, text: string): AgentMessage => ({ role: "toolResult", toolCallId: id,
  toolName: "workspace.read_text", content: [{ type: "text", text }], isError: false, timestamp: 2 });
function source(): AgentMessage[] {
  const messages: AgentMessage[] = [{ role: "user", content: "Inspect PUMP-A71 at https://plant.invalid/spec#DOC-REV-006", timestamp: 0 }];
  for (let i = 0; i < 6; i++) messages.push(assistant([call(`source-call-${i}`)]), result(`source-call-${i}`, `EQUIP-Z009 [E1] ${"ordinary observation; ".repeat(300)}`));
  messages.push({ role: "user", content: "Correction: use PUMP-A17 and DOC-REV-007, never PUMP-A71. Preserve 000123 exactly.", timestamp: 3 });
  messages.push(assistant([{ type: "thinking", thinking: "opaque source reasoning", thinkingSignature: "source-only" },
    call("pending-read-0007"), call("pending-write-0008", "workspace.write_text", { path: "PUMP-A17.txt" })]), result("pending-read-0007", "Exact partial batch result DOC-REV-007"));
  return messages.map((m, i) => ({ ...m, arjunRawSeq: i + 1 }) as unknown as AgentMessage);
}

describe("destination context admission", () => {
  for (const window of [4096, 32768]) it(`transitions from 8192 to ${window} with exact corrections, identifiers and a pending batch`, () => {
    const messages = source();
    const original = structuredClone(messages);
    const notes = new WorkingNotes(); notes.setGoal("Inspect PUMP-A17 using DOC-REV-007");
    const built = buildDestinationContext({ messages, destination: destination(window), notes, fixedTokens: 220,
      preserved: { ...transitionPins(messages), policyDecisions: ['{"id":"approval-00042","status":"pending","tool":"workspace.write_text"}'] } });
    const text = JSON.stringify(built.projection);
    for (const exact of ["PUMP-A17", "DOC-REV-007", "000123", "EQUIP-Z009", "[E1]", "approval-00042", "https://plant.invalid/spec#DOC-REV-006"]) expect(text).toContain(exact);
    expect(text).toContain("Correction: use PUMP-A17 and DOC-REV-007, never PUMP-A71.");
    expect(text).not.toContain("opaque source reasoning");
    expect(text).not.toContain("source-only");
    expect(pendingToolCalls(built.projection).map((p) => p.toolCall)).toEqual(pendingToolCalls(messages).map((p) => p.toolCall));
    expect(text).toContain("Exact partial batch result DOC-REV-007");
    expect(built.estimatedInput).toBeLessThanOrEqual(inputBudget(window, 256));
    expect(messages).toEqual(original);
    if (window > 8192) expect(text).toContain("ordinary observation; ".repeat(300));
    else expect(text).toContain("Exact saved result:");
  });

  it("refuses an unrepresentable pending action instead of trimming its arguments", () => {
    const messages = [assistant([call("write-001", "workspace.write_text", { path: "x".repeat(18000) })])];
    expect(() => buildDestinationContext({ messages, destination: destination(2048), notes: new WorkingNotes(), fixedTokens: 200,
      preserved: {} })).toThrow(/safe limit/);
    expect(pendingToolCalls(messages)[0]!.toolCall.arguments.path).toHaveLength(18000);
  });

  it("refuses loss of an unsupported image", () => {
    const messages = [{ role: "user", content: [{ type: "image", data: "AA==", mimeType: "image/png" }], timestamp: 0 }] as AgentMessage[];
    expect(() => buildDestinationContext({ messages, destination: destination(4096), notes: new WorkingNotes(), fixedTokens: 100, preserved: {} })).toThrow(/saved image/);
  });

  it("rejects orphan results before attempting compression", () => {
    expect(() => buildDestinationContext({ messages: [result("missing", "result")], destination: destination(4096),
      notes: new WorkingNotes(), fixedTokens: 100, preserved: {} })).toThrow(/orphan/);
  });
});
