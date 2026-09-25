/**
 * The catalogue has to fit the window the model was actually served with.
 *
 * These run against the *real* `TOOL_DEFINITIONS` rather than fixtures, because
 * the failure being guarded was a property of the real catalogue and no fixture
 * would have had it: 31 tools, 9,251 tokens, sent to a server started with
 * 8,192. A test built on three invented tools would have passed throughout.
 */

import { describe, expect, it } from "vitest";
import { TOOL_DEFINITIONS } from "./catalogue.js";
import { settingsForWindow } from "./compaction.js";
import { toolBudgetFor } from "./run.js";
import {
  capSchemaDescriptions,
  capSentences,
  catalogueTokens,
  COMPRESSION_STAGES,
  estimateTextTokens,
  fitToolsToBudget,
  MAX_MINIMAL_TOKENS_PER_TOOL,
  toolTokens,
  type BudgetableTool,
} from "./tool-budget.js";

/** The catalogue in the shape it reaches the wire in. */
function catalogue(): BudgetableTool[] {
  return TOOL_DEFINITIONS.map((definition) => ({
    name: definition.name,
    description: definition.description,
    parameters: definition.parameters,
  }));
}

/**
 * The catalogue rendered at maximum compression, with nothing dropped.
 *
 * A budget of `1` would also reach the smallest stage, but it reaches it by
 * amputating the catalogue down to one tool — which is a different question
 * from "how small can the whole catalogue be made". The budget has to sit
 * between the `minimal` size and the next stage up. Measured at P08 (60 tools)
 * as 5,755 tokens at `minimal` (95.9 per tool), so 5,800 still compresses to
 * `minimal` and drops nothing. At P07 (56 tools) it was 5,350 (95.5 per tool);
 * at P06 (52 tools) 4,990 / 5,619; it was
 * 4,800 at 48 tools (P05: 4,521 / 5,089),
 * 4,000 at 43 (P04: 3,902 / 4,399) and 3,000 at 33.
 */
function minimalCatalogue(): BudgetableTool[] {
  const fitted = fitToolsToBudget(catalogue(), 5_800);
  if (fitted.report.stage !== "minimal" || fitted.report.dropped.length > 0) {
    throw new Error(
      `expected the whole catalogue at the smallest stage, got ${fitted.report.stage} with ` +
        `${fitted.report.dropped.length} dropped`,
    );
  }
  return fitted.tools;
}

/** The ten tools P04 added, which plans list last and the fitter drops first. */
const P04_ARTIFACT_TOOLS: ReadonlySet<string> = new Set([
  "artifact.manifest",
  "artifact.read_version",
  "artifact.read_region",
  "artifact.list_templates",
  "artifact.validate",
  "artifact.render",
  "artifact.diff",
  "artifact.resolve_evidence",
  "artifact.register_version",
  "artifact.edit",
]);

/**
 * The four page tools P06 added, listed just before the artifact family (the
 * order a plan lists them in), so a small window drops the artifact family
 * first and these next. `media.extract_findings`, `document.read_pages` and
 * `document.search` sit earlier and keep the common case covered.
 */
const P06_PAGE_TOOLS: ReadonlySet<string> = new Set([
  "document.layout_map",
  "document.render_regions",
  "document.ocr_regions",
  "document.extract_tables",
]);

/**
 * The four retrieval tools P07 added, listed just before the P06 page tools.
 * `knowledge.search_authorized`, `knowledge.load_evidence_region` and
 * `knowledge.multimodal_retrieve` sit earlier and keep retrieval covered when
 * a window is too small for these.
 */
const P07_RETRIEVAL_TOOLS: ReadonlySet<string> = new Set([
  "knowledge.hybrid_search",
  "knowledge.source_version",
  "memory.neighbours",
  "knowledge.rerank",
]);

/**
 * The four calculation families P08 added, listed after the P07 tools and
 * before the page tools. `calculation.evaluate_with_units` sits in the core
 * and keeps every figure computable when a window is too small for these.
 */
const P08_CALCULATION_TOOLS: ReadonlySet<string> = new Set([
  "calculation.validate_dimensions",
  "calculation.compare",
  "calculation.solve",
  "calculation.sensitivity",
]);

/** Every `properties` key in a schema, at every depth. */
function propertyNames(schema: unknown, found: string[] = []): string[] {
  if (Array.isArray(schema)) {
    for (const entry of schema) propertyNames(entry, found);
    return found;
  }
  if (typeof schema !== "object" || schema === null) return found;
  for (const [key, value] of Object.entries(schema as Record<string, unknown>)) {
    if (key === "properties" && typeof value === "object" && value !== null) {
      found.push(...Object.keys(value as Record<string, unknown>));
    }
    propertyNames(value, found);
  }
  return found;
}

describe("the tool catalogue against a small window", () => {
  it("is larger than an 8k window on its own, which is the defect", () => {
    // The measurement this whole module exists for. If this ever stops being
    // true the compression is no longer load-bearing — but the assertion is
    // what makes that a decision somebody takes rather than a thing that drifts.
    expect(catalogueTokens(catalogue())).toBeGreaterThan(8_192);
  });

  /**
   * The core — every tool before P07's — at 8k. Measured at P07: the 8k tool
   * budget is 3,686 tokens and this core fills 3,611 of it at `minimal`, so no
   * further tool of any kind fits an 8k window whole. P08 widened the core's
   * `calculation.evaluate_with_units` (typed inputs, result unit, rounding)
   * and it still fits. P07's and P08's tools are kept whole at 10,240 (the
   * artifact family and two page tools drop); at 12,288 only the last two
   * artifact tools drop, and everything is kept from 14,336.
   */
  it("fits the budget an 8k model can afford, with every tool before P07's intact", () => {
    const budget = toolBudgetFor(8_192, "You are ARJUN.".repeat(80), "Write bubble sort");
    const core = catalogue().filter(
      (tool) =>
        !P04_ARTIFACT_TOOLS.has(tool.name) &&
        !P06_PAGE_TOOLS.has(tool.name) &&
        !P07_RETRIEVAL_TOOLS.has(tool.name) &&
        !P08_CALCULATION_TOOLS.has(tool.name),
    );
    const fitted = fitToolsToBudget(core, budget);

    expect(fitted.report.tokens).toBeLessThanOrEqual(budget);
    expect(fitted.report.overBudget).toBe(false);
    // Compression before amputation: nothing is dropped at this size.
    expect(fitted.tools).toHaveLength(core.length);
    expect(fitted.report.dropped).toEqual([]);
  });

  /**
   * P04-OBS-1. With the ten shared artifact tools the *whole* catalogue no
   * longer fits an 8k window even at `minimal` (3,902 tokens against a budget
   * of about 3,690). No plan offers the whole catalogue: `planning::derive`
   * permits the artifact family only for work on a deliverable, and lists it
   * last. So when a plan does permit everything and the window is 8k, what is
   * dropped must be that family, from the tail, and said — never a producer,
   * the sandbox or retrieval. Role-scoped loading (plan P03) is the remedy.
   *
   * P05 adds the five orchestrator tools *before* that family (4,521 tokens
   * at `minimal` for 48 tools); at 8k the fitter now drops nine of the ten
   * artifact tools and keeps every orchestrator tool, which this still pins.
   *
   * P06 adds four page tools between the orchestrator tools and that family
   * (4,990 tokens at `minimal` for 52 tools, 95.96 per tool against the 96
   * Rust reserves). Measured at 8k: the ten artifact tools and then the four
   * page tools are dropped, and every orchestrator tool, producer and reader
   * is kept.
   *
   * P07 adds four retrieval tools before the page tools (5,350 tokens at
   * `minimal` for 56 tools). Measured at 8k: the artifact family, the page
   * tools and then these four are dropped; the original search, page-region
   * and multimodal readers are kept.
   *
   * P08 adds four calculation families between the P07 tools and the page
   * tools (5,755 tokens for 60). At 8k they go after the page tools and
   * before the P07 tools; `calculation.evaluate_with_units` stays.
   */
  it("drops only the P04, P06, P07 and P08 additions, from the tail, when the whole catalogue meets an 8k window", () => {
    const budget = toolBudgetFor(8_192, "You are ARJUN.".repeat(80), "Write bubble sort");
    const fitted = fitToolsToBudget(catalogue(), budget);
    const droppable = (name: string) =>
      P04_ARTIFACT_TOOLS.has(name) ||
      P06_PAGE_TOOLS.has(name) ||
      P07_RETRIEVAL_TOOLS.has(name) ||
      P08_CALCULATION_TOOLS.has(name);

    expect(fitted.report.overBudget).toBe(false);
    expect(fitted.report.dropped.length).toBeGreaterThan(0);
    for (const name of fitted.report.dropped) {
      expect(droppable(name), `${name} was dropped`).toBe(true);
    }
    // From the tail: the whole artifact family goes before any page tool does.
    const dropped = fitted.report.dropped;
    const firstPageTool = dropped.findIndex((name) => P06_PAGE_TOOLS.has(name));
    if (firstPageTool >= 0) {
      for (const name of P04_ARTIFACT_TOOLS) {
        expect(dropped.includes(name), `${name} kept while a page tool was dropped`).toBe(true);
      }
    }
    // Every page tool goes before any P08 calculation tool does...
    if (dropped.some((name) => P08_CALCULATION_TOOLS.has(name))) {
      for (const name of P06_PAGE_TOOLS) {
        expect(dropped.includes(name), `${name} kept while a P08 tool was dropped`).toBe(true);
      }
    }
    // ...and every P08 tool before any P07 retrieval tool.
    const firstRetrievalTool = dropped.findIndex((name) => P07_RETRIEVAL_TOOLS.has(name));
    if (firstRetrievalTool >= 0) {
      for (const name of [...P06_PAGE_TOOLS, ...P08_CALCULATION_TOOLS]) {
        expect(dropped.includes(name), `${name} kept while a P07 tool was dropped`).toBe(true);
      }
    }
    expect(fitted.tools.some((kept) => kept.name === "calculation.evaluate_with_units")).toBe(true);
    for (const tool of catalogue().filter((t) => !droppable(t.name))) {
      expect(fitted.tools.some((kept) => kept.name === tool.name), `${tool.name} kept`).toBe(true);
    }
  });

  it("keeps every P07 and P08 tool at a 10k window, dropping only artifact and page tools from the tail", () => {
    const budget = toolBudgetFor(10_240, "You are ARJUN.".repeat(80), "Write bubble sort");
    const fitted = fitToolsToBudget(catalogue(), budget);
    for (const name of [...P07_RETRIEVAL_TOOLS, ...P08_CALCULATION_TOOLS]) {
      expect(fitted.tools.some((kept) => kept.name === name), `${name} kept`).toBe(true);
    }
    for (const name of fitted.report.dropped) {
      expect(P04_ARTIFACT_TOOLS.has(name) || P06_PAGE_TOOLS.has(name), `${name} was dropped`).toBe(true);
    }
    // Measured at P08: the ten artifact tools and the last two page tools.
    expect(fitted.report.dropped.filter((name) => P06_PAGE_TOOLS.has(name))).toEqual([
      "document.ocr_regions",
      "document.extract_tables",
    ]);
  });

  it("keeps every page tool at a 12k window and everything at 14k", () => {
    const twelve = fitToolsToBudget(catalogue(), toolBudgetFor(12_288, "You are ARJUN.".repeat(80), "Write bubble sort"));
    for (const name of twelve.report.dropped) {
      expect(P04_ARTIFACT_TOOLS.has(name), `${name} was dropped at 12k`).toBe(true);
    }
    const fourteen = fitToolsToBudget(catalogue(), toolBudgetFor(14_336, "You are ARJUN.".repeat(80), "Write bubble sort"));
    expect(fourteen.report.dropped).toEqual([]);
  });

  it("leaves the whole request inside the window on the case that failed", () => {
    // The reported failure, reconstructed: an 8,192-token server, a system
    // prompt of the size this product composes, and a five-word question.
    // Before this module the request came to 9,238 tokens and was refused.
    const window = 8_192;
    const systemPrompt = "You are ARJUN, an offline engineering assistant. ".repeat(30);
    const prompt = "Write the code for bubble sorting";

    const budget = toolBudgetFor(window, systemPrompt, prompt);
    const fitted = fitToolsToBudget(catalogue(), budget);

    const request =
      estimateTextTokens(systemPrompt) + estimateTextTokens(prompt) + fitted.report.tokens;
    // Inside the window with the reply's reserve still untouched, which is the
    // property the provider actually checks.
    expect(request).toBeLessThan(window - settingsForWindow(window).reserveTokens);
  });

  it("holds the per-tool floor the Rust side reserves against", () => {
    // `commands::agent::TOOL_FLOOR_TOKENS_PER_TOOL` reserves this much per
    // eligible tool before it budgets documents and history. If the compressed
    // catalogue costs more than that per tool, Rust is handing out a window
    // this side has already spent — and both sides believe they fitted.
    const minimal = minimalCatalogue();
    const perTool = catalogueTokens(minimal) / minimal.length;
    expect(perTool).toBeLessThanOrEqual(MAX_MINIMAL_TOKENS_PER_TOOL);
  });

  it("never changes what a call has to look like, only what is said about it", () => {
    // The line this module must not cross. A model builds a call from the
    // parameter names, types and `required` list; prose only helps it choose.
    // Compression that touched the shape would produce calls the gateway
    // refuses as malformed — trading a context error for a worse one.
    const before = catalogue();
    const after = minimalCatalogue();

    expect(after.map((tool) => tool.name)).toEqual(before.map((tool) => tool.name));
    for (let index = 0; index < before.length; index += 1) {
      const original = before[index]!.parameters as Record<string, unknown>;
      const compressed = after[index]!.parameters as Record<string, unknown>;
      expect(propertyNames(compressed)).toEqual(propertyNames(original));
      expect(compressed.required).toEqual(original.required);
      expect(compressed.additionalProperties).toEqual(original.additionalProperties);
      expect(compressed.type).toEqual(original.type);
    }
  });

  it("gets smaller at every stage, so pressure always buys something", () => {
    const tools = catalogue();
    let previous = Number.POSITIVE_INFINITY;
    for (const stage of COMPRESSION_STAGES) {
      const rendered = tools.map((tool) => ({
        ...tool,
        description: capSentences(tool.description ?? "", stage.toolDescriptionChars),
        parameters: capSchemaDescriptions(tool.parameters, stage.parameterDescriptionChars),
      }));
      const tokens = catalogueTokens(rendered);
      expect(tokens).toBeLessThan(previous);
      previous = tokens;
    }
  });

  it("drops tools only when compression has run out, and says which", () => {
    // A window so small that even the minimal rendering of everything will not
    // fit. Dropping is the honest last resort; doing it silently is not.
    const fitted = fitToolsToBudget(catalogue(), 600);

    expect(fitted.tools.length).toBeGreaterThan(0);
    expect(fitted.tools.length).toBeLessThan(TOOL_DEFINITIONS.length);
    expect(fitted.report.dropped.length).toBe(TOOL_DEFINITIONS.length - fitted.tools.length);
    // Dropped from the tail: the plan's earliest steps keep their tools.
    expect(fitted.tools[0]!.name).toBe(TOOL_DEFINITIONS[0]!.name);
  });

  it("keeps one tool rather than none, however small the window", () => {
    // A run offered nothing it can call has to answer from memory, which is the
    // failure this product exists to prevent. One tool beats none.
    const fitted = fitToolsToBudget(catalogue(), 1);
    expect(fitted.tools.length).toBeGreaterThanOrEqual(1);
    expect(fitted.report.overBudget).toBe(true);
  });

  it("leaves the catalogue alone when no window is known", () => {
    // An unknown window is not a reason to degrade. It is a reason to say the
    // budget could not be worked out, which the report does.
    const fitted = fitToolsToBudget(catalogue(), 0);
    expect(fitted.report.stage).toBe("full");
    expect(fitted.tools).toHaveLength(TOOL_DEFINITIONS.length);
  });
});

describe("cutting prose", () => {
  it("cuts at a sentence boundary, not mid-clause", () => {
    // "Do not use it for X" truncated to "Do not use it" inverts the sentence.
    const text = "Searches the documents. Do not use it for greetings or code. Effects: none.";
    expect(capSentences(text, 40)).toBe("Searches the documents.");
  });

  it("never returns more than it was asked for", () => {
    const text = "A".repeat(500);
    expect(capSentences(text, 60).length).toBeLessThanOrEqual(60);
  });

  it("keeps something rather than nothing when one sentence is too long", () => {
    const text = `${"word ".repeat(100)}end.`;
    const capped = capSentences(text, 50);
    expect(capped.length).toBeGreaterThan(0);
    expect(capped.endsWith("…")).toBe(true);
  });

  it("removes schema descriptions entirely at zero, keeping the shape", () => {
    const schema = {
      type: "object",
      properties: {
        query: { type: "string", description: "What to look for." },
        page: { type: "integer", description: "Which batch." },
      },
      required: ["query"],
    };
    const stripped = capSchemaDescriptions(schema, 0) as typeof schema;
    expect(stripped.required).toEqual(["query"]);
    expect(Object.keys(stripped.properties)).toEqual(["query", "page"]);
    expect(JSON.stringify(stripped)).not.toContain("What to look for");
  });

  it("reaches descriptions nested under anyOf and items", () => {
    // A pass that only walked top-level properties would leave most of the
    // catalogue's parameter prose in place — several tools carry unions.
    const schema = {
      anyOf: [{ type: "string", description: "one" }],
      items: { type: "object", properties: { a: { type: "string", description: "two" } } },
    };
    const stripped = JSON.stringify(capSchemaDescriptions(schema, 0));
    expect(stripped).not.toContain("one");
    expect(stripped).not.toContain("two");
  });
});

describe("the budget a window affords", () => {
  it("reports nothing rather than guessing when the window is unknown", () => {
    expect(toolBudgetFor(0, "system", "prompt")).toBe(0);
    expect(toolBudgetFor(Number.NaN, "system", "prompt")).toBe(0);
  });

  it("asks for maximum compression rather than none when the window is tiny", () => {
    // The distinction that matters: `0` means "no budget could be worked out"
    // and leaves the catalogue untouched. A known but hopeless window must not
    // land on that path — that is exactly how a 9,238-token request was sent.
    expect(toolBudgetFor(2_048, "x".repeat(8_000), "prompt")).toBe(1);
  });

  it("never spends more than a share of the window on tools", () => {
    // A large window is not a reason to spend half of it on prose the model
    // reads once and a document it cannot then hold.
    for (const window of [8_192, 32_768, 131_072]) {
      expect(toolBudgetFor(window, "", "")).toBeLessThanOrEqual(Math.floor(window * 0.45));
    }
  });

  it("leaves room for the reply and the conversation at every size", () => {
    for (const window of [4_096, 8_192, 32_768]) {
      const budget = toolBudgetFor(window, "system prompt", "question");
      const reserve = settingsForWindow(window).reserveTokens;
      expect(budget + reserve).toBeLessThan(window);
    }
  });
});

describe("measuring a tool", () => {
  it("charges the envelope, not only the text", () => {
    const bare: BudgetableTool = { name: "a.b", description: "", parameters: {} };
    expect(toolTokens(bare)).toBeGreaterThan(0);
  });
});
