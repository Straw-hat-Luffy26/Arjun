/** Deterministic destination admission. Raw history is never edited; model-
 * specific reasoning is excluded and pending action IDs/arguments remain exact. */
import { createCompactionSummaryMessage, type AgentMessage } from "@openclaw/agent-core";
import { ContextBudgetExceeded, admitProjection, inputBudget, projectedTokens } from "./context-budget.js";
import { preservedMessage, type PreservedState } from "./compaction.js";
import { pendingToolCalls } from "./tool-recovery.js";
import type { WorkingNotes } from "./working-notes.js";

export interface ModelContext {
  modelId: string;
  servedModelId: string;
  provider: string;
  contextWindow: number;
  maxTokens: number;
  input: string[];
}

export function sameModelContext(a: ModelContext, b: ModelContext): boolean {
  return a.modelId === b.modelId && a.servedModelId === b.servedModelId && a.provider === b.provider
    && a.contextWindow === b.contextWindow && a.maxTokens === b.maxTokens
    && [...a.input].sort().join(",") === [...b.input].sort().join(",");
}

function textOf(message: AgentMessage): string {
  const content = (message as { content?: unknown }).content;
  if (typeof content === "string") return content;
  if (!Array.isArray(content)) return "";
  return content.flatMap((b) => b.type === "text" ? [b.text] : b.type === "toolCall" ? [JSON.stringify(b.arguments)] : []).join("\n");
}

/** User instructions are retained verbatim and in order, including superseded
 * wording. The latest correction must not be reconstructed from a summary. */
export function transitionPins(messages: AgentMessage[]): Pick<PreservedState, "exactInstructions" | "exactIdentifiers"> {
  const exactInstructions = messages.filter((m) => m.role === "user"
    && !(m as { arjunContextState?: boolean }).arjunContextState).map(textOf);
  const identifiers = new Set<string>();
  for (const message of messages) {
    for (const match of textOf(message).matchAll(/https?:\/\/[^\s"<>]+|\[E\d+\]|\b[A-Za-z0-9]+(?:[-_:/.][A-Za-z0-9]+)+\b/g)) identifiers.add(match[0]);
  }
  return { exactInstructions, exactIdentifiers: [...identifiers] };
}

export function buildDestinationContext(options: {
  messages: AgentMessage[]; destination: ModelContext; notes: WorkingNotes;
  preserved: PreservedState; fixedTokens: number;
}): { messages: AgentMessage[]; projection: AgentMessage[]; estimatedInput: number; omittedGroups: number } {
  const { destination, notes, preserved, fixedTokens } = options;
  const budget = inputBudget(destination.contextWindow, destination.maxTokens);
  const fingerprint = (messages: AgentMessage[]) => pendingToolCalls(messages).map(({ toolCall: p }) => ({ id: p.id, name: p.name, arguments: p.arguments }));
  const originalPending = fingerprint(options.messages);
  // Whole pending batches (including already-recorded results) cannot be cut.
  const pendingStart = originalPending.length
    ? options.messages.findLastIndex((m) => m.role === "assistant") : options.messages.length;
  let messages = options.messages.map((original): AgentMessage => {
    const message = structuredClone(original);
    if (Array.isArray((message as { content?: unknown }).content)
      && (message as { content: { type: string }[] }).content.some((b) => b.type === "image")
      && !destination.input.includes("image")) {
      throw new ContextBudgetExceeded("the destination cannot represent a saved image; an authorized text extraction is required.");
    }
    if (message.role !== "assistant") return message;
    // Do not replay opaque thinking, signatures or provider response handles.
    const content = message.content.flatMap<(typeof message.content)[number]>((block) => {
      if (block.type === "thinking") return [];
      if (block.type === "text") return [{ type: "text" as const, text: block.text }];
      if (block.id !== block.id.trim()) throw new ContextBudgetExceeded("a saved action ID cannot be represented exactly by the destination.");
      return [{ type: "toolCall" as const, id: block.id, name: block.name, arguments: structuredClone(block.arguments) }];
    });
    return { ...message, content, model: destination.servedModelId, provider: destination.provider,
      api: "openai-completions", responseModel: undefined };
  });
  const carried = preservedMessage(preserved, notes, 0);
  const omitted: number[] = [];
  const omittedMessage = () => createCompactionSummaryMessage(
    `Earlier completed history omitted for this destination window. Exact raw transcript sequences: ${omitted.join(", ")}. Retrieve with memory.recall_authorized and transcriptSeq before relying on omitted details.`,
    0, new Date(0).toISOString(),
  ) as AgentMessage;
  const projection = () => [...(omitted.length ? [omittedMessage()] : []), ...(carried ? [carried] : []), ...messages];
  // Already carried exactly, outside any subsequent summary. Removing these
  // duplicates never splits a tool batch (validated above).
  const indices = messages.map((_, index) => index);
  const keep = (m: AgentMessage) => m.role !== "user" || !(preserved.exactInstructions ?? []).includes(textOf(m));
  const candidates = indices.filter((i) => i < pendingStart && keep(messages[i]!));
  messages = messages.filter(keep);
  let omittedGroups = 0;
  // First shorten large, durable completed tool observations. Every preview
  // retains an exact retrieval reference; arbitrary unsaved content is not cut.
  for (const i of candidates) {
    if (fixedTokens + projectedTokens(projection()) <= budget) break;
    const original = options.messages[i]!;
    const seq = (original as { arjunRawSeq?: number }).arjunRawSeq;
    const index = messages.findIndex((m) => (m as { arjunRawSeq?: number }).arjunRawSeq === seq);
    if (!seq || index < 0 || messages[index]!.role !== "toolResult" || textOf(messages[index]!).length < 512) continue;
    const message = messages[index]! as Extract<AgentMessage, { role: "toolResult" }>;
    messages[index] = { ...message, content: [{ type: "text", text: `${textOf(message).slice(0, 256)}\n[Preview. Exact saved result: memory.recall_authorized({scope:"run",transcriptSeq:${seq},offsetChars:0,limitChars:1536}).]` }] };
  }
  // If previews still do not fit, omit only whole completed groups. The pinned
  // task state remains exact; the marker explicitly says the history is omitted.
  while (fixedTokens + projectedTokens(projection()) > budget && messages.length) {
    const head = messages[0];
    if (!head) break;
    const count = head.role === "assistant" ? 1 + messages.slice(1).findIndex((m) => m.role !== "toolResult") : 1;
    const groupSize = count === 0 ? messages.length : count;
    const group = messages.slice(0, groupSize);
    if (originalPending.length && group.some((m) => m.role === "assistant" && m.content.some((b) => b.type === "toolCall" && originalPending.some((p) => p.id === b.id)))) break;
    const seqs = group.map((m) => (m as { arjunRawSeq?: number }).arjunRawSeq);
    if (seqs.some((seq) => !seq)) break;
    omitted.push(...seqs as number[]);
    omittedGroups++;
    messages.splice(0, groupSize);
  }
  if (JSON.stringify(fingerprint(messages)) !== JSON.stringify(originalPending)) {
    throw new ContextBudgetExceeded("destination construction changed the pending action batch.");
  }
  const admitted = admitProjection(projection(), fixedTokens, budget);
  return { messages: [...(omitted.length ? [omittedMessage()] : []), ...messages], projection: admitted, estimatedInput: fixedTokens + projectedTokens(admitted), omittedGroups };
}
