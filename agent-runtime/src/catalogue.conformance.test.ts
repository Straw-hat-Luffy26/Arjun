/**
 * The catalogue on this side and `ToolName` on the Rust side must agree.
 *
 * ## Why this test exists
 *
 * Two lists kept in two languages that must say the same thing about the same
 * names will eventually disagree. The disagreement is silent: a name this side
 * declares that Rust does not know is refused by the gateway, and a name Rust
 * knows that this side does not declare is never offered. Neither crashes, and
 * the symptom — a tool the model can see but cannot use, or one it cannot see
 * but is entitled to — looks like a tuning problem rather than a code defect.
 *
 * The test below is the thing that turns that silent disagreement into a
 * failure that says which name is wrong, in the build that introduced it rather
 * than in a field report three months later.
 */

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { TOOL_DEFINITIONS, definitionFor, type ToolDefinition } from "./catalogue.js";
import {
  CANONICAL_TOOL_NAMES,
  LEGACY_TOOL_NAMES,
  canonicalToolName,
  isArtifactProducing,
  isCalculation,
  isCodeExecution,
  isEvidenceProducing,
  isSideEffecting,
} from "./tool-names.js";

/**
 * The authoritative wire names from Rust's `ToolName::as_str()`.
 *
 * Kept here rather than read from the Rust source, because these are the names
 * the protocol carries. Adding or removing one here is a deliberate act that
 * also changes the protocol — and the test that fails when they differ is the
 * one that makes sure the deliberate act happened on both sides.
 */
const RUST_WIRE_NAMES: ReadonlySet<string> = new Set([
  "knowledge.search_authorized",
  "knowledge.load_evidence_region",
  "knowledge.multimodal_retrieve",
  "media.extract_findings",
  "memory.recall_authorized",
  "memory.promote_approved",
  "workspace.read_text",
  "workspace.write_text",
  "calculation.evaluate_with_units",
  "artifact.create_approval_note",
  "artifact.create_calculation_workbook",
  "artifact.create_briefing_deck",
  "sandbox.run_code",
  "artifact.verify_docx",
  "capability.search",
  "agent.delegate_readonly",
  "sovereignty.get_evidence",
  "document.read_pages",
  "document.search",
  "knowledge.build_graph",
  "artifact.create_chart",
  "artifact.create_diagram",
  "artifact.create_pdf",
  "artifact.create_table",
  "notebook.list",
  "notebook.create",
  "notebook.rename",
  "notebook.delete",
  "notebook.list_sources",
  "notebook.add_source",
  "notebook.remove_source",
  // Missing from this list, and therefore from the catalogue, since they were
  // added to Rust. The published contract below is what found them.
  "artifact.list",
  "artifact.read",
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
  "task.plan_update",
  "agent.delegate",
  "agent.status",
  "agent.cancel",
  "task.request_review",
  "document.layout_map",
  "document.render_regions",
  "document.ocr_regions",
  "document.extract_tables",
]);

/**
 * Legacy names from `ToolName::legacy_str()` that old task records may hold.
 * These must resolve to the current name through `ToolName::from_str`.
 */
const LEGACY_NAMES: ReadonlyMap<string, string> = new Map([
  ["search_documents", "knowledge.search_authorized"],
  ["load_more_evidence", "knowledge.load_evidence_region"],
  ["memory_recall_authorized", "memory.recall_authorized"],
  ["memory_promote_approved", "memory.promote_approved"],
  ["read_scoped_file", "workspace.read_text"],
  ["write_scoped_file", "workspace.write_text"],
  ["run_calculation", "calculation.evaluate_with_units"],
  ["create_docx", "artifact.create_approval_note"],
  ["create_xlsx", "artifact.create_calculation_workbook"],
  // Missing from this list until the shared table was checked against it.
  // Rust has resolved it since the rename; this file simply never listed it,
  // which is the drift these tests exist to catch.
  ["create_pptx", "artifact.create_briefing_deck"],
  ["execute_code", "sandbox.run_code"],
  ["validate_artifact", "artifact.verify_docx"],
  // The four artifact tools that were namespaced from the start. Rust's
  // `legacy_str` resolves each bare spelling; this file did not list any of
  // them, which is why the table above read 12 where the shared one reads 17.
  ["create_diagram", "artifact.create_diagram"],
  ["create_pdf", "artifact.create_pdf"],
  ["create_chart", "artifact.create_chart"],
  ["create_table", "artifact.create_table"],
]);

/**
 * Aliases this side resolves that Rust does not, and deliberately so.
 *
 * `create_flowchart` was never the name of anything. It is what a model
 * reaches for when it has been asked for a flowchart, and folding it onto the
 * diagram tool here means the call succeeds instead of ending the step on an
 * unresolved name. Rust never sees the raw spelling — the runtime canonicalises
 * before the call crosses the wire — so `ToolName::legacy_str` has no reason to
 * carry it, and asserting parity without this exception would force one of the
 * two tables to be wrong.
 */
const RUNTIME_ONLY_ALIASES: ReadonlyMap<string, string> = new Map([
  ["create_flowchart", "artifact.create_diagram"],
]);

/**
 * The expected read-only status, mirroring Rust's `ToolName::is_read_only`.
 *
 * Held here so the test can check both directions: a tool this side marks
 * read-only that Rust does not is one that may run in parallel when Rust
 * expects it serialised, or worse, vice versa.
 */
const EXPECTED_READ_ONLY: ReadonlyMap<string, boolean> = new Map([
  ["knowledge.search_authorized", true],
  ["knowledge.load_evidence_region", true],
  ["knowledge.multimodal_retrieve", true],
  ["media.extract_findings", true],
  ["memory.recall_authorized", true],
  ["memory.promote_approved", false],
  ["workspace.read_text", true],
  ["workspace.write_text", false],
  ["calculation.evaluate_with_units", true],
  ["artifact.create_approval_note", false],
  ["artifact.create_calculation_workbook", false],
  ["artifact.create_briefing_deck", false],
  ["sandbox.run_code", false],
  ["artifact.verify_docx", true],
  ["capability.search", true],
  ["agent.delegate_readonly", true],
  ["sovereignty.get_evidence", true],
  ["document.read_pages", true],
  ["document.search", true],
  ["knowledge.build_graph", true],
  ["artifact.create_chart", false],
  ["artifact.create_diagram", false],
  ["artifact.create_pdf", false],
  ["artifact.create_table", false],
  ["notebook.list", true],
  ["notebook.create", false],
  ["notebook.rename", false],
  ["notebook.delete", false],
  ["notebook.list_sources", true],
  ["notebook.add_source", false],
  ["notebook.remove_source", false],
  ["artifact.list", true],
  ["artifact.read", true],
  ["artifact.manifest", true],
  ["artifact.read_version", true],
  ["artifact.read_region", true],
  ["artifact.list_templates", true],
  ["artifact.validate", true],
  ["artifact.render", true],
  ["artifact.diff", true],
  ["artifact.resolve_evidence", true],
  ["artifact.register_version", false],
  ["artifact.edit", false],
  ["task.plan_update", false],
  ["agent.delegate", false],
  ["agent.status", true],
  ["agent.cancel", false],
  ["task.request_review", false],
  ["document.layout_map", true],
  ["document.render_regions", true],
  ["document.ocr_regions", true],
  ["document.extract_tables", true],
]);

describe("the shared canonicalisation layer agrees with this file's tables", () => {
  // `tool-names.ts` is what production classifies by. This file is what pins
  // the protocol. They must be the same list, or the classification is being
  // done against names the protocol does not carry -- which is precisely the
  // defect that made every completed effect go unrecorded after the rename.
  it("declares exactly the wire names Rust does", () => {
    expect(new Set(CANONICAL_TOOL_NAMES)).toEqual(RUST_WIRE_NAMES);
  });

  it("declares exactly the legacy aliases Rust does, plus the runtime-only ones", () => {
    expect(new Map(LEGACY_TOOL_NAMES)).toEqual(
      new Map([...LEGACY_NAMES, ...RUNTIME_ONLY_ALIASES]),
    );
  });

  it("resolves every runtime-only alias to a tool Rust knows", () => {
    // The exception above is only safe while what it folds onto is real. An
    // alias pointing at a name Rust dropped would turn a working call into an
    // unresolved one at the gateway instead of in the runtime.
    for (const [alias, current] of RUNTIME_ONLY_ALIASES) {
      expect(canonicalToolName(alias)).toBe(current);
      expect(RUST_WIRE_NAMES).toContain(current);
    }
  });

  it("folds every legacy spelling onto its current name", () => {
    for (const [legacy, current] of LEGACY_NAMES) {
      expect(canonicalToolName(legacy)).toBe(current);
    }
  });

  it("folds every current spelling onto itself", () => {
    for (const name of RUST_WIRE_NAMES) {
      expect(canonicalToolName(name)).toBe(name);
    }
  });

  it("refuses to guess at a name neither table knows", () => {
    // Fail-closed. A caller that cannot tell what a name means must not be
    // handed a default, because the default that matters -- "not
    // side-effecting" -- is the dangerous one.
    expect(canonicalToolName("rm_rf")).toBeUndefined();
    expect(canonicalToolName("")).toBeUndefined();
  });
});

describe("catalogue \u2194 Rust conformance", () => {
  it("every TS tool name exists in Rust's ToolName enum", () => {
    for (const definition of TOOL_DEFINITIONS) {
      expect(RUST_WIRE_NAMES.has(definition.name)).toBe(true);
    }
  });

  it("every Rust tool name has a TS definition", () => {
    for (const name of RUST_WIRE_NAMES) {
      expect(definitionFor(name)).toBeDefined();
    }
  });

  it("readOnly agrees between TS and Rust for every tool", () => {
    for (const definition of TOOL_DEFINITIONS) {
      const expected = EXPECTED_READ_ONLY.get(definition.name);
      expect(expected).toBeDefined();
      expect(definition.readOnly).toBe(
        expected,
      );
    }
  });
});

describe("schema strictness", () => {
  it("every tool schema disallows additional properties", () => {
    for (const definition of TOOL_DEFINITIONS) {
      const schema = definition.parameters;
      expect(
        (schema as unknown as { additionalProperties?: boolean }).additionalProperties,
      ).toBe(false);
    }
  });
});

describe("description completeness", () => {
  /**
   * Six clauses appear in every description, in the same order. The first
   * word of each clause is enough to detect its presence without being
   * fragile against rewording.
   */
  const REQUIRED_CLAUSES = [
    // What to use it for
    { pattern: /\bUse it\b/i, label: "when to use" },
    // What not to use it for
    { pattern: /\bDo not use\b/i, label: "when not to use" },
    // What it changes
    { pattern: /\bEffects?\b/i, label: "side effects" },
    // What it costs
    { pattern: /\bLimits?\b/i, label: "limits" },
    // What to do when it fails
    { pattern: /\bIf it\b/i, label: "failure recovery" },
  ];

  for (const definition of TOOL_DEFINITIONS) {
    it(`${definition.name} contains all required description clauses`, () => {
      for (const { pattern, label } of REQUIRED_CLAUSES) {
        expect(
          pattern.test(definition.description),
        ).toBe(true);
      }
    });
  }
});

/**
 * The contract Rust publishes, generated from `orchestrator::contract` and held
 * byte-equal to it by a Rust test. Read here rather than typed here: the lists
 * above were typed, and drifted exactly where it mattered -- the gateway
 * required arguments this catalogue never offered, so four tools were in every
 * catalogue and could not be called, and two Rust tools were never offered.
 */
interface PublishedArgument {
  name: string;
  kind: "text" | "path" | "integer" | "object" | "list";
}
interface PublishedTool {
  name: string;
  aliases: string[];
  required: PublishedArgument[];
  optional: PublishedArgument[];
  readOnly: boolean;
  sideEffecting: boolean;
  output: "evidence" | "calculation" | "artifact" | "execution" | "childResult" | "text";
}
const PUBLISHED = JSON.parse(
  readFileSync(new URL("./tool-contract.json", import.meta.url), "utf8"),
) as { contractVersion: number; tools: PublishedTool[] };

/** The kind the Rust gateway checks, for one property of a TypeBox schema. */
function kindOf(property: Record<string, unknown>): PublishedArgument["kind"] | "unknown" {
  if (Array.isArray(property.anyOf)) {
    // A union of string literals is text to the gateway.
    const allText = (property.anyOf as Array<Record<string, unknown>>).every(
      (member) => member.type === "string" || typeof member.const === "string",
    );
    return allText ? "text" : "unknown";
  }
  switch (property.type) {
    case "string":
      return "text";
    case "integer":
      return "integer";
    case "object":
      return "object";
    case "array":
      return "list";
    default:
      return "unknown";
  }
}

describe("the published Rust tool contract", () => {
  it("names exactly the tools this runtime's tables name", () => {
    const published = new Set(PUBLISHED.tools.map((tool) => tool.name));
    expect(published).toEqual(RUST_WIRE_NAMES);
    expect(new Set(CANONICAL_TOOL_NAMES)).toEqual(published);
  });

  for (const tool of PUBLISHED.tools) {
    describe(tool.name, () => {
      const definition = definitionFor(tool.name);

      it("has a definition the model can be offered", () => {
        expect(definition).toBeDefined();
      });

      it("offers only arguments the gateway accepts, and requires none it does not", () => {
        const schema = definition!.parameters as unknown as {
          properties?: Record<string, Record<string, unknown>>;
          required?: string[];
        };
        const offered = Object.keys(schema.properties ?? {});
        const requiredHere = new Set(schema.required ?? []);
        const accepted = new Map(
          [...tool.required, ...tool.optional].map((argument) => [argument.name, argument.kind]),
        );

        // Every argument the model may send is one the gateway will accept.
        for (const name of offered) {
          expect(accepted.has(name), `${tool.name} offers ${name}, which the gateway refuses`).toBe(
            true,
          );
        }
        // The gateway never requires what the model's schema lets it omit.
        for (const argument of tool.required) {
          expect(
            requiredHere.has(argument.name),
            `${tool.name} requires ${argument.name} at the gateway and the schema makes it optional`,
          ).toBe(true);
        }
        // And each is the same kind on both sides. A path is text to a schema.
        for (const name of offered) {
          const expected = accepted.get(name);
          const here = kindOf(schema.properties?.[name] ?? {});
          expect(here, `${tool.name}.${name}`).toBe(expected === "path" ? "text" : expected);
        }
      });

      it("agrees about read-only, side effects and what it returns", () => {
        expect(definition!.readOnly).toBe(tool.readOnly);
        expect(isSideEffecting(tool.name)).toBe(tool.sideEffecting);
        expect(isEvidenceProducing(tool.name)).toBe(tool.output === "evidence");
        expect(isCalculation(tool.name)).toBe(tool.output === "calculation");
        expect(isArtifactProducing(tool.name)).toBe(tool.output === "artifact");
        expect(isCodeExecution(tool.name)).toBe(tool.output === "execution");
      });
    });
  }

  it("resolves every alias this runtime knows to the tool Rust resolves it to", () => {
    const owner = new Map<string, string>();
    for (const tool of PUBLISHED.tools) {
      for (const alias of tool.aliases) owner.set(alias, tool.name);
    }
    for (const [alias, current] of LEGACY_TOOL_NAMES) {
      expect(owner.get(alias), alias).toBe(current);
    }
  });
});

describe("legacy name compatibility", () => {
  it("every legacy name resolves to a current tool definition", () => {
    for (const [legacy, current] of LEGACY_NAMES) {
      const definition = definitionFor(current);
      expect(definition).toBeDefined();
      expect(definition!.name).toBe(current);
    }
  });

  it("no tool was introduced with both a legacy and a current name that differ", () => {
    // Tools introduced after the namespace rename have no legacy name.
    // Any name in LEGACY_NAMES must map to a name in RUST_WIRE_NAMES.
    for (const [, current] of LEGACY_NAMES) {
      expect(RUST_WIRE_NAMES.has(current)).toBe(true);
    }
  });
});
