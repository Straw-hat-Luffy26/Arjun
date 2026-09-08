/**
 * One place that knows what a tool is called, and what kind of thing it does.
 *
 * ## Why this exists
 *
 * The tools were renamed into namespaces — `create_docx` became
 * `artifact.create_approval_note`, `search_documents` became
 * `knowledge.search_authorized`. Rust learned both spellings, in
 * `ToolName::as_str` and `ToolName::legacy_str`, and resolves either through
 * `ToolName::from_str`. Nothing on this side did.
 *
 * `note-taking.ts` matched tool names against three hard-coded sets, all of
 * them written in the old spelling, and the catalogue hands it the new one. So
 * every one of those comparisons was false, and had been since the rename:
 *
 * - a search returned `[E3]` and no evidence marker was recorded;
 * - a calculation ran and no calculation id was recorded;
 * - a document was produced and no artifact and **no completed effect** was
 *   recorded.
 *
 * The last is the serious one. A completed effect is what a resumed run reads
 * to find out what already happened. With none recorded, a run that produced an
 * approval note, lost its process, and resumed would produce the note again —
 * the exact failure the working notes exist to prevent, silently reintroduced
 * by a rename.
 *
 * So the names and the classifications live here, once, and everything that
 * needs either asks this module rather than writing a set of its own.
 *
 * ## What canonicalisation means here
 *
 * Accepting both spellings on the way *in* and speaking only the current one on
 * the way *out*. A task record written months ago holds `create_docx`; the
 * catalogue offers `artifact.create_approval_note`; both must fold to the same
 * thing, and what gets written down from here on is the current name.
 *
 * The tables are checked against Rust's by `catalogue.conformance.test.ts`. Two
 * lists in two languages that must agree will otherwise drift, and the drift is
 * silent — which is how this defect happened in the first place.
 */

/** The current wire name of every tool, as `ToolName::as_str` spells it. */
export const CANONICAL_TOOL_NAMES = [
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
  "notebook.add_source",
  "notebook.create",
  "notebook.delete",
  "notebook.list",
  "notebook.list_sources",
  "notebook.remove_source",
  "notebook.rename",
] as const;

export type CanonicalToolName = (typeof CANONICAL_TOOL_NAMES)[number];

const CANONICAL: ReadonlySet<string> = new Set(CANONICAL_TOOL_NAMES);

/**
 * The pre-namespace spelling of every tool that had one.
 *
 * Read-side only, mirroring `ToolName::legacy_str`. Records, audit lines and
 * working notes written before the rename hold these, and a reader that could
 * not resolve them would report a months-old run as having used a tool that
 * does not exist. Nothing writes these.
 */
export const LEGACY_TOOL_NAMES: ReadonlyMap<string, CanonicalToolName> = new Map([
  ["search_documents", "knowledge.search_authorized"],
  ["load_more_evidence", "knowledge.load_evidence_region"],
  ["memory_recall_authorized", "memory.recall_authorized"],
  ["memory_promote_approved", "memory.promote_approved"],
  ["read_scoped_file", "workspace.read_text"],
  ["write_scoped_file", "workspace.write_text"],
  ["run_calculation", "calculation.evaluate_with_units"],
  ["create_docx", "artifact.create_approval_note"],
  ["create_xlsx", "artifact.create_calculation_workbook"],
  ["create_pptx", "artifact.create_briefing_deck"],
  ["execute_code", "sandbox.run_code"],
  ["validate_artifact", "artifact.verify_docx"],
]);

/**
 * The current name for a tool, whichever spelling it arrived in.
 *
 * Returns `undefined` for a name neither table knows. That is deliberate and is
 * the fail-closed reading: a caller that cannot tell what a name means must not
 * guess, and a classification that answered "not side-effecting" for an
 * unrecognised tool would be exactly the wrong guess.
 */
export function canonicalToolName(raw: string): CanonicalToolName | undefined {
  if (CANONICAL.has(raw)) return raw as CanonicalToolName;
  return LEGACY_TOOL_NAMES.get(raw);
}

/**
 * Tools that return numbered evidence the answer can cite.
 *
 * A marker recorded here is what lets `pruneStaleToolResults` clear a passage
 * from the context while keeping the citation resolvable.
 */
const EVIDENCE_PRODUCING: ReadonlySet<CanonicalToolName> = new Set([
  "knowledge.search_authorized",
  "knowledge.load_evidence_region",
  "knowledge.multimodal_retrieve",
  "media.extract_findings",
]);

/** Tools whose result is a deterministic calculation with its working shown. */
const CALCULATION: ReadonlySet<CanonicalToolName> = new Set([
  "calculation.evaluate_with_units",
]);

/**
 * Tools that leave a file behind the run can be asked about afterwards.
 *
 * A subset of the side-effecting set: running code has an effect and produces
 * no artifact this run can name.
 */
const ARTIFACT_PRODUCING: ReadonlySet<CanonicalToolName> = new Set([
  "workspace.write_text",
  "artifact.create_approval_note",
  "artifact.create_calculation_workbook",
  "artifact.create_briefing_deck",
]);

/**
 * Tools whose success is a side effect a resumed run must not repeat.
 *
 * Mirrors `events::idempotency::is_side_effecting` in Rust. The briefing deck
 * belongs here for the same reason the other two documents do — it is a file
 * written to disk, and a resumption that wrote it twice would leave two.
 */
const SIDE_EFFECTING: ReadonlySet<CanonicalToolName> = new Set([
  "workspace.write_text",
  "artifact.create_approval_note",
  "artifact.create_calculation_workbook",
  "artifact.create_briefing_deck",
  "sandbox.run_code",
]);

/** Whether this tool returns numbered evidence. Accepts either spelling. */
export function isEvidenceProducing(tool: string): boolean {
  const name = canonicalToolName(tool);
  return name !== undefined && EVIDENCE_PRODUCING.has(name);
}

/** Whether this tool performs a calculation. Accepts either spelling. */
export function isCalculation(tool: string): boolean {
  const name = canonicalToolName(tool);
  return name !== undefined && CALCULATION.has(name);
}

/** Whether this tool leaves a named file behind. Accepts either spelling. */
export function isArtifactProducing(tool: string): boolean {
  const name = canonicalToolName(tool);
  return name !== undefined && ARTIFACT_PRODUCING.has(name);
}

/**
 * Whether repeating this tool would repeat a side effect. Accepts either
 * spelling.
 *
 * An unrecognised name answers `false`, which is the only answer available: a
 * tool this build has never heard of cannot be executed by it either, so there
 * is no effect to guard against.
 */
export function isSideEffecting(tool: string): boolean {
  const name = canonicalToolName(tool);
  return name !== undefined && SIDE_EFFECTING.has(name);
}

/** Whether this tool runs code rather than producing a document. */
export function isCodeExecution(tool: string): boolean {
  return canonicalToolName(tool) === "sandbox.run_code";
}
