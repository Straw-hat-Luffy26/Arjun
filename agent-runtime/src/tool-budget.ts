/**
 * Fitting the tool catalogue into the window the server was actually started
 * with.
 *
 * ## The failure this removes
 *
 * Measured on this machine, on the catalogue as it stands: the 31 tool
 * definitions a planned run is offered serialise to 36,782 characters — about
 * **9,196 tokens**. `vram_planner` routinely serves a 32k-trained model at
 * 8,192 tokens to buy GPU layers, so the tool definitions alone were larger
 * than the entire context window before a system prompt, a conversation or a
 * question had been added to them.
 *
 * The consequence was not a degraded answer. It was:
 *
 * ```text
 * 400 request (9238 tokens) exceeds the available context size (8192 tokens)
 * ```
 *
 * on a turn whose question was five words long — and it was unfixable from
 * anywhere else in the pipeline, because every other budget in the product
 * (documents, history, the reply reserve) was dividing up a window that had
 * already been spent. `RunCompactor` could not help either: it summarises
 * *messages*, and not one of those 9,196 tokens is a message.
 *
 * ## Why compression rather than dropping tools
 *
 * A model calls a tool from its name and its parameter *shape*. It chooses
 * which tool to call from the prose. Those are different costs and only one of
 * them is load-bearing for correctness: a tool whose description has been cut
 * to its first two sentences is still callable and still correct when called,
 * while a tool that has been dropped is a capability the run no longer has.
 *
 * The catalogue's descriptions are written in six clauses in a fixed order —
 * what it does, when to use it, when not to, what it changes, what it costs,
 * what to do when it fails (see `catalogue.ts`). That order is a compression
 * plan: the clauses that decide *whether to call this tool at all* come first,
 * so cutting from the end removes the guidance a model needs least in order to
 * pick correctly.
 *
 * So pressure is applied in stages, and tools are only dropped when even the
 * smallest honest rendering of all of them will not fit. Each stage is
 * reported, because a run whose tools were quietly shortened is a run whose
 * tool choice may be worse for a reason nobody can see.
 *
 * ## Why this is not "just raise --ctx-size"
 *
 * Because the window is not free. `--ctx-size` and `--n-gpu-layers` are one
 * decision: the KV cache for a larger window comes out of the same VRAM as the
 * offloaded layers, so a bigger window is bought with a slower model. On the
 * hardware this product targets that trade is often not available at all. The
 * request has to fit the machine, not the other way round.
 */

import { estimateContextTokens, type AgentMessage } from "@openclaw/agent-core";

/** The tool shape this module needs. Structural, so it fits `AgentTool`. */
export interface BudgetableTool {
  name: string;
  description?: string;
  parameters?: unknown;
}

/**
 * Characters a provider spends per tool that are not the name, description or
 * schema: the `{"type":"function","function":{…}}` envelope and the separators
 * around it.
 *
 * Counted rather than ignored because it is charged once per tool, and a
 * catalogue of thirty is over a thousand characters of pure envelope.
 */
const TOOL_ENVELOPE_CHARS = 48;

/**
 * How the catalogue is rendered, in order of increasing pressure.
 *
 * Every stage keeps every tool callable: the name and the parameter *shape*
 * are never touched. What shrinks is prose — first the trailing clauses of a
 * tool's description, then the parameter descriptions, and only then the
 * leading clauses.
 */
export const COMPRESSION_STAGES = [
  { id: "full", toolDescriptionChars: Infinity, parameterDescriptionChars: Infinity },
  { id: "trimmedGuidance", toolDescriptionChars: 600, parameterDescriptionChars: 160 },
  { id: "shortGuidance", toolDescriptionChars: 280, parameterDescriptionChars: 90 },
  { id: "schemaOnly", toolDescriptionChars: 150, parameterDescriptionChars: 0 },
  { id: "minimal", toolDescriptionChars: 90, parameterDescriptionChars: 0 },
] as const;

export type CompressionStageId = (typeof COMPRESSION_STAGES)[number]["id"];

/**
 * What the catalogue costs per tool once it is compressed as far as it goes.
 *
 * A ceiling on the *average*, not on any single tool: the Rust side reserves
 * `eligible tools × this` before it budgets documents and history, and what it
 * needs to reserve is what the whole catalogue will cost, not what its largest
 * member does. Measured at 87.3 tokens per tool over the real catalogue
 * (2,707 tokens across 31 tools); 96 is that with room for a tool or two more.
 *
 * Asserted by `tool-budget.test.ts` against the real catalogue, and mirrored on
 * the Rust side as `commands::agent::TOOL_FLOOR_TOKENS_PER_TOOL`. The two
 * numbers have to agree or one side is dividing a window the other has already
 * spent, so this one is measured and that one is checked against it.
 */
export const MAX_MINIMAL_TOKENS_PER_TOOL = 96;

/** What the fitting did, in the terms an operator can act on. */
export interface ToolBudgetReport {
  /** How the catalogue was rendered in the end. */
  stage: CompressionStageId;
  /** Tokens the rendered catalogue occupies. */
  tokens: number;
  /** Tokens it would have occupied untouched. */
  tokensBefore: number;
  /** What it was allowed. `0` when no window was known. */
  budget: number;
  /** Tools left out entirely because even `minimal` would not fit. */
  dropped: string[];
  /** True when the catalogue still does not fit and nothing more can be cut. */
  overBudget: boolean;
}

export interface FittedTools<T extends BudgetableTool> {
  tools: T[];
  report: ToolBudgetReport;
}

/**
 * Loose text, measured by the same estimator the ledger uses.
 *
 * Exported because the callers that decide budgets must count with the same
 * ruler the ledger reports with. Two estimators would put the meter on screen
 * and the decision behind it into permanent, invisible disagreement.
 */
export function estimateTextTokens(text: string): number {
  if (!text) return 0;
  const message = {
    role: "user",
    content: [{ type: "text", text }],
    timestamp: 0,
  } as AgentMessage;
  return estimateContextTokens([message]).tokens;
}

/** What one tool costs on the wire: name, prose, schema and envelope. */
export function toolTokens(tool: BudgetableTool): number {
  const wire =
    tool.name +
    (tool.description ?? "") +
    JSON.stringify(tool.parameters ?? {}) +
    " ".repeat(TOOL_ENVELOPE_CHARS);
  return estimateTextTokens(wire);
}

/** What a whole catalogue costs on the wire. */
export function catalogueTokens(tools: readonly BudgetableTool[]): number {
  return tools.reduce((total, tool) => total + toolTokens(tool), 0);
}

/**
 * Cuts prose at a sentence boundary at or before `maxChars`.
 *
 * Sentence-aligned rather than character-aligned because the descriptions are
 * written as ordered clauses and a model reads them as sentences: a cut in the
 * middle of "Do not use it for" inverts the meaning of the clause it truncates,
 * which is worse than not carrying the clause at all.
 *
 * Falls back to a word-boundary cut with an ellipsis only when the first
 * sentence is itself longer than the cap, so the result is never longer than
 * asked for and never empty for a tool that had a description.
 */
export function capSentences(text: string, maxChars: number): string {
  const trimmed = text.trim();
  if (!Number.isFinite(maxChars)) return trimmed;
  if (maxChars <= 0) return "";
  if (trimmed.length <= maxChars) return trimmed;

  let cut = 0;
  // A sentence ends at ". ", "! " or "? ", or at the end of the string.
  const boundary = /[.!?](?:\s|$)/g;
  for (let match = boundary.exec(trimmed); match !== null; match = boundary.exec(trimmed)) {
    const end = match.index + 1;
    if (end > maxChars) break;
    cut = end;
  }
  if (cut === 0) {
    // One very long sentence. Cut it at a word boundary and mark the cut,
    // rather than returning nothing — a tool with no description at all is one
    // the model cannot tell apart from its neighbour.
    const hard = trimmed.slice(0, Math.max(1, maxChars - 1));
    const lastSpace = hard.lastIndexOf(" ");
    const body = lastSpace > maxChars / 2 ? hard.slice(0, lastSpace) : hard;
    return `${body.trimEnd()}…`;
  }
  return trimmed.slice(0, cut).trimEnd();
}

/**
 * Rewrites a JSON schema's `description` fields to fit `maxChars`.
 *
 * Deep and structural: `anyOf`, `items` and nested objects all carry
 * descriptions, and a pass that only looked at top-level properties would leave
 * most of the prose in place. Everything that is not a description — types,
 * enums, `required`, `minimum`, `additionalProperties` — is copied through
 * untouched, because that is the part the model builds a valid call from.
 *
 * A `maxChars` of `0` removes descriptions entirely.
 */
export function capSchemaDescriptions(schema: unknown, maxChars: number): unknown {
  if (!Number.isFinite(maxChars)) return schema;
  if (Array.isArray(schema)) {
    return schema.map((entry) => capSchemaDescriptions(entry, maxChars));
  }
  if (typeof schema !== "object" || schema === null) return schema;

  const out: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(schema as Record<string, unknown>)) {
    if (key === "description" && typeof value === "string") {
      const capped = capSentences(value, maxChars);
      if (capped) out[key] = capped;
      continue;
    }
    out[key] = capSchemaDescriptions(value, maxChars);
  }
  return out;
}

/** Renders one tool at one stage. Returns the tool itself at `full`. */
function render<T extends BudgetableTool>(
  tool: T,
  stage: (typeof COMPRESSION_STAGES)[number],
): T {
  if (stage.id === "full") return tool;
  return {
    ...tool,
    description: capSentences(tool.description ?? "", stage.toolDescriptionChars),
    parameters: capSchemaDescriptions(tool.parameters, stage.parameterDescriptionChars),
  };
}

/**
 * Fits the catalogue into `budgetTokens`, compressing before dropping.
 *
 * A `budgetTokens` of zero or less means the caller could not work one out —
 * an unknown window, most often. The catalogue is returned untouched and the
 * report says so: guessing a budget from no information would be a silent
 * degradation with nothing behind it, and the window is the one number this
 * decision cannot be made without.
 *
 * Dropping, when it happens, is from the end of the list. Rust hands the
 * eligible tools in the order the plan's steps need them, so the tail is the
 * work the run reaches last — and a run that gets three steps in before running
 * out of capability is strictly better than one that cannot take the first
 * step. Every dropped name is reported.
 */
export function fitToolsToBudget<T extends BudgetableTool>(
  tools: readonly T[],
  budgetTokens: number,
): FittedTools<T> {
  const tokensBefore = catalogueTokens(tools);

  if (!Number.isFinite(budgetTokens) || budgetTokens <= 0 || tools.length === 0) {
    return {
      tools: [...tools],
      report: {
        stage: "full",
        tokens: tokensBefore,
        tokensBefore,
        budget: Math.max(0, Math.floor(budgetTokens) || 0),
        dropped: [],
        overBudget: false,
      },
    };
  }

  const budget = Math.floor(budgetTokens);

  for (const stage of COMPRESSION_STAGES) {
    const rendered = tools.map((tool) => render(tool, stage));
    const tokens = catalogueTokens(rendered);
    if (tokens <= budget) {
      return {
        tools: rendered,
        report: {
          stage: stage.id,
          tokens,
          tokensBefore,
          budget,
          dropped: [],
          overBudget: false,
        },
      };
    }
  }

  // Even `minimal` does not fit. Drop from the tail until it does, keeping at
  // least one tool: a run offered nothing it can call has to answer a document
  // question from memory, which is the failure this product exists to prevent.
  const smallest = COMPRESSION_STAGES[COMPRESSION_STAGES.length - 1]!;
  const kept = tools.map((tool) => render(tool, smallest));
  const dropped: string[] = [];
  while (kept.length > 1 && catalogueTokens(kept) > budget) {
    const removed = kept.pop();
    if (removed) dropped.push(removed.name);
  }
  const tokens = catalogueTokens(kept);
  return {
    tools: kept,
    report: {
      stage: smallest.id,
      tokens,
      tokensBefore,
      budget,
      dropped: dropped.reverse(),
      overBudget: tokens > budget,
    },
  };
}
