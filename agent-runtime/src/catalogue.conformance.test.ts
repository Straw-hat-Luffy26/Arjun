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

import { describe, expect, it } from "vitest";
import { TOOL_DEFINITIONS, definitionFor, type ToolDefinition } from "./catalogue.js";
import {
  CANONICAL_TOOL_NAMES,
  LEGACY_TOOL_NAMES,
  canonicalToolName,
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
]);

describe("the shared canonicalisation layer agrees with this file's tables", () => {
  // `tool-names.ts` is what production classifies by. This file is what pins
  // the protocol. They must be the same list, or the classification is being
  // done against names the protocol does not carry -- which is precisely the
  // defect that made every completed effect go unrecorded after the rename.
  it("declares exactly the wire names Rust does", () => {
    expect(new Set(CANONICAL_TOOL_NAMES)).toEqual(RUST_WIRE_NAMES);
  });

  it("declares exactly the legacy aliases Rust does", () => {
    expect(new Map(LEGACY_TOOL_NAMES)).toEqual(new Map(LEGACY_NAMES));
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
