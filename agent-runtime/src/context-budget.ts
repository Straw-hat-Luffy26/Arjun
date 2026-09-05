/** Admission is separate from compression: a failed compression never permits
 * an oversized request. Counts are conservative estimates, not tokenizer claims.
 * The raw transcript is not modified by any function in this module. */
import { estimateTokens, type AgentMessage } from "@openclaw/agent-core";

export class ContextBudgetExceeded extends Error {
  readonly code = "context_budget_exceeded";
  constructor(detail: string) {
    super(`Context budget: ${detail}`);
    this.name = "ContextBudgetExceeded";
  }
}

export function inputBudget(window: number, outputTokens: number): number {
  if (!Number.isFinite(window) || window < 512 || !Number.isFinite(outputTokens) || outputTokens < 0) {
    throw new ContextBudgetExceeded("a valid model context limit and output reservation are required.");
  }
  const available = Math.min(Math.floor(window * 0.7), window - Math.ceil(outputTokens) - 256);
  if (available < 256) throw new ContextBudgetExceeded("the configured reply leaves no safe input capacity.");
  return available;
}

/** Do not use historical provider usage as the size of a newly cut projection. */
export function projectedTokens(messages: readonly AgentMessage[]): number {
  return messages.reduce((sum, message) => sum + estimateTokens(message) + 16, 0);
}

export function admitProjection(messages: AgentMessage[], fixedTokens: number, budget: number): AgentMessage[] {
  const estimated = fixedTokens + projectedTokens(messages);
  if (!Number.isFinite(estimated) || estimated > budget) {
    throw new ContextBudgetExceeded(`the next request needs about ${estimated} input tokens; the safe limit is ${budget}. Durable state must be retained before retrying with a smaller projection.`);
  }
  return messages;
}
