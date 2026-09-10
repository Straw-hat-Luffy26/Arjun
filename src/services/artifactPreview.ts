/**
 * Turning a preview of a produced file into something to draw.
 *
 * ## The defect this fixes
 *
 * `commands/artifact_preview.rs` serialises exactly four fields:
 *
 * ```json
 * { "kind": "docxBody", "text": "...", "truncated": false, "sizeBytes": 8321 }
 * ```
 *
 * The surface declared a different object — `mime`, and a body under `content`,
 * `dataUrl` or `reason` depending on the kind — and both preview panes read
 * those names:
 *
 * ```tsx
 * <pre>{preview.content}</pre>       // undefined, for every kind
 * <img src={preview.dataUrl} />      // undefined, for every image
 * ```
 *
 * The seven `kind` strings did line up, which is why the lane looked alive: the
 * request succeeded, the spinner cleared, the pane opened — and opened empty.
 * A preview that renders nothing is indistinguishable from a file with nothing
 * in it, so the failure read as a backend problem for as long as it went
 * unexamined. It was never a backend problem; `preview()` is covered by tests
 * that assert on `r.text`.
 *
 * Rust is authoritative: it is the producer, its shape is the wire format, and
 * it is the tested half. This module describes that shape and nothing else.
 *
 * ## Why the decision is a function rather than JSX
 *
 * The same pane existed twice, in `AssistantMessageCell.tsx` and in
 * `RunView.tsx`, as near-identical copies — so the broken field names existed
 * twice, and a fix to one would have left the other blank. What to show is
 * decided here, once, and tested without a DOM; the two components only draw
 * the answer. Same reason `mermaidParse`, `runProgress` and `markdownFence` are
 * their own modules.
 */

/** The preview kinds `PreviewKind` can serialise, camel-cased as serde emits. */
export const PREVIEW_KINDS = [
  'text',
  'markdown',
  'docxBody',
  'xlsxFirstSheet',
  'pptxSlideList',
  'image',
  'svg',
  'pdf',
  'unsupported',
] as const;

export type PreviewKind = (typeof PREVIEW_KINDS)[number];

/**
 * A preview exactly as `artifact_preview.rs` sends it.
 *
 * One flat object, not a discriminated union: the Rust struct is one struct,
 * and modelling it as a union was what let the field names drift unnoticed.
 */
export interface ArtifactPreview {
  kind: PreviewKind;
  /** The body — preview text, or a `data:` URL when `kind` is `image`. */
  text: string;
  /** True when the file was longer than the cap and the preview was cut. */
  truncated: boolean;
  /** Size of the whole file, not of the preview. */
  sizeBytes: number;
}

/** What the pane should draw. */
export type PreviewDisplay =
  /** An image, from a `data:` URL. */
  | { layout: 'image'; src: string; note: string | null }
  /** A block of text. `mono` when it is tabular or extracted, not prose. */
  | { layout: 'body'; body: string; mono: boolean; note: string | null }
  /** No preview to draw, and a sentence saying why. */
  | { layout: 'notice'; message: string };

const TRUNCATED_NOTE = 'Preview is truncated. Use the folder button for the full file.';

/**
 * Kinds whose body is not prose, and so is drawn monospaced: text files, an
 * extracted `.docx` body, and a sheet rendered as a markdown table.
 */
const MONOSPACED: ReadonlySet<string> = new Set([
  'text',
  'docxBody',
  'xlsxFirstSheet',
  // An SVG preview is its own source, which is markup and reads as code.
  'svg',
]);

/**
 * How to draw this preview.
 *
 * Total, and never returns something that would draw as an empty pane. A body
 * that came back empty is reported as empty in words, because an empty `<pre>`
 * is precisely the symptom this module was written to remove — the reader
 * cannot tell it from a preview that failed.
 */
export function previewDisplay(preview: ArtifactPreview): PreviewDisplay {
  if (preview.kind === 'unsupported') {
    return {
      layout: 'notice',
      message:
        'Preview is not available for this format. Use the folder button to open it in the file manager.',
    };
  }

  // A PDF is named rather than shrugged at. There is no PDF reader in the Rust
  // process — PDFs are read by the Python sidecar — so the backend sends the
  // kind and no body, and saying "PDF, open it to read it" is more use to
  // somebody looking at a file this app produced a moment ago than the generic
  // unsupported-format sentence.
  if (preview.kind === 'pdf') {
    return {
      layout: 'notice',
      message: 'This is a PDF. Use the folder button to open it.',
    };
  }

  const body = preview.text ?? '';

  if (preview.kind === 'image') {
    // A data URL is the only thing this can be, and an <img> with an empty src
    // resolves against the page and silently draws nothing.
    if (!body.startsWith('data:')) {
      return {
        layout: 'notice',
        message: 'The image preview did not arrive in a form the app can draw.',
      };
    }
    return {
      layout: 'image',
      src: body,
      note: preview.truncated ? 'Preview is truncated to fit.' : null,
    };
  }

  if (body.trim() === '') {
    return {
      layout: 'notice',
      message: 'The file opened, but there was no text in it to preview.',
    };
  }

  return {
    layout: 'body',
    body,
    mono: MONOSPACED.has(preview.kind),
    note: preview.truncated ? TRUNCATED_NOTE : null,
  };
}
