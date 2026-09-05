import type { AgentMessage } from "@openclaw/agent-core";
import type { AssistantMessage, ToolCall } from "@openclaw/ai";

/** Only the final batch may be incomplete. Missing results in older turns,
 * duplicate results and unknown call IDs are corruption, not recovery work. */
export function pendingToolCalls(messages: AgentMessage[]): { assistantMessage: AssistantMessage; toolCall: ToolCall }[] {
  const pending = new Map<string, { assistantMessage: AssistantMessage; toolCall: ToolCall }>();
  for (const message of messages) {
    if (message.role === "toolResult") {
      if (!pending.delete(message.toolCallId)) throw new Error("The saved transcript has an orphan or duplicate tool result.");
      continue;
    }
    if (pending.size) throw new Error("The saved transcript interrupts an unfinished tool batch.");
    if (message.role === "assistant") {
      for (const block of message.content) {
        if (block.type !== "toolCall") continue;
        if (!block.id || pending.has(block.id)) throw new Error("The saved tool batch has duplicate or missing action IDs.");
        pending.set(block.id, { assistantMessage: message, toolCall: block });
      }
    }
  }
  return [...pending.values()];
}
