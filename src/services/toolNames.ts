/**
 * What a tool is called, and how it reads on screen.
 *
 * ## Why this exists
 *
 * The tools were renamed into namespaces — `create_docx` became
 * `artifact.create_approval_note`. Rust resolves both spellings through
 * `ToolName::from_str`; the agent runtime resolves both through
 * `agent-runtime/src/tool-names.ts`. The surface resolved neither.
 *
 * It held two label maps — one in `useRun.ts`, one in `AssistantMessageCell.tsx`
 * — that were copies of each other, both keyed on the *old* spelling. Live
 * events carry the new one, so every activity row in the trace and every tool
 * step in the chat fell through to its fallback and displayed the raw wire
 * name: a person watching a run saw `artifact.create_approval_note` where the
 * design says "Producing a Word document".
 *
 * Records written before the rename hold the old spelling, so both must work.
 * That is what canonicalisation is for here: accept either on the way in, and
 * look the label up once.
 *
 * The table is checked against Rust's `ToolName` and against the runtime's copy
 * by `toolNames.test.ts`. Three lists in three languages that must agree will
 * otherwise drift, and the drift is silent — which is how this happened.
 */

/** The current wire name of every tool, as `ToolName::as_str` spells it. */
export const CANONICAL_TOOL_NAMES = [
  'knowledge.search_authorized',
  'knowledge.load_evidence_region',
  'knowledge.multimodal_retrieve',
  'media.extract_findings',
  'memory.recall_authorized',
  'memory.promote_approved',
  'workspace.read_text',
  'workspace.write_text',
  'calculation.evaluate_with_units',
  'artifact.create_approval_note',
  'artifact.create_calculation_workbook',
  'artifact.create_briefing_deck',
  'artifact.create_diagram',
  'artifact.create_pdf',
  'artifact.create_chart',
  'artifact.create_table',
  'sandbox.run_code',
  'artifact.verify_docx',
  'capability.search',
  'agent.delegate_readonly',
  'sovereignty.get_evidence',
] as const;

export type CanonicalToolName = (typeof CANONICAL_TOOL_NAMES)[number];

const CANONICAL: ReadonlySet<string> = new Set(CANONICAL_TOOL_NAMES);

/**
 * The pre-namespace spelling of every tool that had one.
 *
 * Read-side only, mirroring `ToolName::legacy_str`. Task records and audit
 * lines written before the rename hold these.
 */
export const LEGACY_TOOL_NAMES: ReadonlyMap<string, CanonicalToolName> = new Map([
  ['search_documents', 'knowledge.search_authorized'],
  ['load_more_evidence', 'knowledge.load_evidence_region'],
  ['memory_recall_authorized', 'memory.recall_authorized'],
  ['memory_promote_approved', 'memory.promote_approved'],
  ['read_scoped_file', 'workspace.read_text'],
  ['write_scoped_file', 'workspace.write_text'],
  ['run_calculation', 'calculation.evaluate_with_units'],
  ['create_docx', 'artifact.create_approval_note'],
  ['create_xlsx', 'artifact.create_calculation_workbook'],
  ['create_pptx', 'artifact.create_briefing_deck'],
  ['execute_code', 'sandbox.run_code'],
  ['validate_artifact', 'artifact.verify_docx'],
  // Never a former spelling of anything — these four are namespaced in every
  // record ever written. They are here because a model writes the bare name,
  // the runtime now resolves it, and a record of that call must read as the
  // tool it was rather than as an unknown wire name.
  ['create_diagram', 'artifact.create_diagram'],
  ['create_flowchart', 'artifact.create_diagram'],
  ['create_pdf', 'artifact.create_pdf'],
  ['create_chart', 'artifact.create_chart'],
  ['create_table', 'artifact.create_table'],
]);

/**
 * The current name for a tool, whichever spelling it arrived in.
 *
 * `undefined` for a name neither table knows — a tool added by a newer backend
 * than this build. The caller shows the raw string, which is honest, rather
 * than a label invented for a tool it does not know about.
 */
export function canonicalToolName(raw: string): CanonicalToolName | undefined {
  if (CANONICAL.has(raw)) return raw as CanonicalToolName;
  return LEGACY_TOOL_NAMES.get(raw);
}

/**
 * How each tool reads in a trace. Follows `ToolName::describe` in Rust.
 *
 * Present tense and specific: a person watching a run wants to know what is
 * happening, not which function is being called.
 */
const TOOL_LABELS: Readonly<Record<CanonicalToolName, string>> = {
  'knowledge.search_authorized': 'Searching the documents',
  'knowledge.load_evidence_region': 'Reading more of a document',
  'knowledge.multimodal_retrieve': 'Reading a drawing or table',
  'media.extract_findings': 'Reading a scanned page',
  'memory.recall_authorized': 'Recalling what this machine knows',
  'memory.promote_approved': 'Recording an approved fact',
  'workspace.read_text': 'Reading a file',
  'workspace.write_text': 'Writing a file',
  'calculation.evaluate_with_units': 'Calculating',
  'artifact.create_approval_note': 'Producing a Word document',
  'artifact.create_calculation_workbook': 'Producing a workbook',
  'artifact.create_briefing_deck': 'Producing a briefing deck',
  'artifact.create_diagram': 'Drawing a diagram',
  'artifact.create_pdf': 'Producing a PDF',
  'artifact.create_chart': 'Drawing a chart',
  'artifact.create_table': 'Producing a table',
  'sandbox.run_code': 'Running code',
  'artifact.verify_docx': 'Checking a produced file',
  'capability.search': 'Looking for a relevant skill',
  'agent.delegate_readonly': 'Asking a sub-task to read something',
  'sovereignty.get_evidence': 'Collecting sovereignty evidence',
};

/**
 * The label for a tool, in either spelling.
 *
 * Falls back to the raw name for a tool this build does not know. Showing the
 * wire name is not pretty and is at least true; inventing a label would be
 * neither.
 */
export function labelForTool(tool: string): string {
  const name = canonicalToolName(tool);
  return name ? TOOL_LABELS[name] : tool;
}

/** Which icon a tool's row carries in the trace. */
export function iconForTool(tool: string): 'search' | 'link' | 'tool' {
  const name = canonicalToolName(tool) ?? tool;
  if (name.includes('search') || name.includes('retrieve')) return 'search';
  if (name.includes('read') || name.includes('recall')) return 'link';
  return 'tool';
}
