/**
 * What kind of file a run produced, and how to draw it.
 *
 * ## The defect this exists to make impossible
 *
 * `Kind` in `src-tauri/src/agent_runtime/artifacts.rs` has four variants —
 * `Document`, `Workbook`, `Deck`, `Text` — and serialises them camel-cased over
 * the IPC boundary. The surface declared three of them, and keyed its icon
 * table on that short union:
 *
 * ```ts
 * const ARTIFACT_ICONS: Record<ArtifactReport['kind'], typeof FileText> = {
 *   document: FileText, workbook: FileSpreadsheet, text: FileText,
 * };
 * const Icon = ARTIFACT_ICONS[artifact.kind];
 * ```
 *
 * Every briefing deck therefore resolved to `undefined`, and `<Icon />` on
 * `undefined` is a React invariant violation rather than a missing picture. The
 * nearest error boundary is the one around `<Outlet />` in `AppShell`, so a
 * successfully produced PPTX replaced the whole page with a crash card. The
 * file was on disk and sound the entire time; only the drawing of it threw.
 *
 * TypeScript could not have caught it. `ArtifactReport` arrives as JSON from
 * Rust, so its declared type is a claim about a struct in another language and
 * nothing verifies the claim. `toolNames.ts` says this in its own header about
 * the tool table: lists in three languages that must agree will drift, and the
 * drift is silent. This is the same disease in a second place.
 *
 * ## Why a function and not a wider table
 *
 * Adding `deck:` to the table would fix today's crash and leave the shape that
 * caused it — a lookup that returns `undefined` for anything it has not been
 * told about. [`artifactPresentation`] is total: every string in, a drawable
 * presentation out. A fifth `Kind` added in Rust and shipped to an older
 * surface then renders as a plain file row carrying its own raw name, which is
 * how `labelForTool` already treats a tool it does not recognise.
 *
 * `artifactKind.test.ts` reads `artifacts.rs` and fails if a variant there has
 * no entry here, so the drift is caught in CI rather than on a screen.
 */

/**
 * The glyphs a produced file can be drawn with.
 *
 * A closed set on purpose: the component maps it with a `Record`, so a glyph
 * added here without a matching icon is a compile error rather than another
 * `undefined` component.
 */
export const ARTIFACT_GLYPHS = ['document', 'workbook', 'deck', 'file'] as const;

export type ArtifactGlyph = (typeof ARTIFACT_GLYPHS)[number];

export interface ArtifactPresentation {
  /** Which icon to draw. Always one of [`ARTIFACT_GLYPHS`]. */
  glyph: ArtifactGlyph;
  /** What to call this kind of file on screen. */
  label: string;
  /**
   * False when this build has never heard of the kind — a file produced by a
   * newer backend than the surface. The row still draws; it just does not
   * pretend to know what it is holding.
   */
  known: boolean;
}

/**
 * Every artifact kind Rust can send, as `Kind` serialises it.
 *
 * The labels are `Kind::label()` verbatim, so the two languages describe a file
 * with the same words. `artifactKind.test.ts` checks both halves against the
 * Rust source.
 */
const KNOWN_KINDS: ReadonlyMap<string, ArtifactPresentation> = new Map([
  ['document', { glyph: 'document', label: 'Word document', known: true }],
  ['workbook', { glyph: 'workbook', label: 'Workbook', known: true }],
  ['deck', { glyph: 'deck', label: 'Briefing deck', known: true }],
  ['text', { glyph: 'file', label: 'Text file', known: true }],
] as const);

/** The kinds this build recognises, for tests and for exhaustiveness checks. */
export const ARTIFACT_KINDS: readonly string[] = ['document', 'workbook', 'deck', 'text'];

/**
 * How to draw a produced file of this kind.
 *
 * Total. An unrecognised kind is drawn as a generic file labelled with the raw
 * string the backend sent, rather than resolving to `undefined` and taking the
 * page down with it. Showing "deck" where "Briefing deck" belongs is a cosmetic
 * miss; showing nothing at all was not.
 */
export function artifactPresentation(kind: string): ArtifactPresentation {
  const known = KNOWN_KINDS.get(kind);
  if (known) return known;
  return {
    glyph: 'file',
    // The raw name, for the same reason `labelForTool` shows a raw tool name:
    // inventing a friendly label for something this build cannot identify
    // would be a guess presented as knowledge.
    label: kind || 'File',
    known: false,
  };
}
