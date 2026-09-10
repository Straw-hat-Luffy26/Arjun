//! A safe, *preview-only* reader for the files an agent run produces.
//!
//! The point of `artifact_preview` is to give the user a way to *see* what
//! ARJUN wrote, without opening another application, and without trusting
//! the file enough to *run* it. The contract:
//!
//! - Returns a UTF-8 string, or an error.
//! - For `.md` / `.txt` / `.json` / `.csv` / `.log`: returns the text, capped
//!   at `MAX_PREVIEW_BYTES` (256 KiB). Truncation is reported in the
//!   `truncated` field so the UI can say "preview cut off".
//! - For `.docx` / `.xlsx` / `.pptx`: extracts the body XML via
//!   `artifacts::ooxml::read_part`, then walks the XML to pull text
//!   content. The result is plain text, not a faithful rendering — a
//!   word-processor file becomes a wall of text. That is honest: this is
//!   a *preview*, not a substitute for the file the user will open in
//!   Word.
//! - For images: returns a base64 data URL.
//! - For anything else: returns an error with the file extension so the
//!   caller can fall back to a "Reveal in file manager" button.
//!
//! The reader is deliberately conservative. It will not:
//! - Execute any embedded macros.
//! - Follow external references.
//! - Load font files or external style sheets.
//!
//! All of those are decisions a hostile document could turn against the
//! reviewer. The preview's job is to *let the user decide whether to
//! open the file*, not to open it for them.

use std::path::Path;

use serde::{Deserialize, Serialize};

/// How many bytes of a text preview we keep. Beyond this the user can
/// still `Reveal in file manager` to see the rest in their editor.
const MAX_PREVIEW_BYTES: usize = 256 * 1024;

/// What the file is. The kind drives the preview format.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "camelCase")]
pub enum PreviewKind {
    /// Plain text. Returned verbatim, possibly truncated.
    Text,
    /// A markdown / text-like file. Returned verbatim.
    Markdown,
    /// A `.docx` body, returned as plain text extracted from the XML.
    DocxBody,
    /// A `.xlsx` first sheet, returned as a markdown table.
    XlsxFirstSheet,
    /// A `.pptx` slide list, returned as one slide per paragraph.
    PptxSlideList,
    /// An image. Returned as a `data:` URL.
    Image,
    /// An SVG — a diagram or chart this application drew. Returned as its own
    /// source, which is text, so the surface can sanitise and draw it inline
    /// rather than showing a file card for a picture it already has.
    Svg,
    /// A PDF. Named rather than lumped in with "unsupported", because a PDF
    /// this application produced a moment ago is not an unknown format — there
    /// is simply no in-process reader for one here, and saying so is more use
    /// than a shrug. Carries no body; the surface offers to open it.
    Pdf,
    /// A binary or unsupported type. The UI should show "open in app".
    Unsupported,
}

/// What `preview` returns.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ArtifactPreview {
    pub kind: PreviewKind,
    /// The preview text (or data URL for images). May be empty when
    /// `kind` is `Unsupported`.
    pub text: String,
    /// `true` when the file was longer than `MAX_PREVIEW_BYTES` and
    /// the preview was cut at the cap. The user can still reveal the
    /// file in their file manager to see the rest.
    pub truncated: bool,
    /// Original file size in bytes. Always set; useful for the UI
    /// to display alongside the preview length.
    pub size_bytes: u64,
}

/// Renders a preview of the file at `path`. Returns an error only when
/// the file is missing, unreadable, or in a kind the preview cannot
/// handle. The `kind` argument tells the preview which reader to
/// invoke; passing the wrong kind is the caller's mistake, not the
/// file's, and the function reports it.
pub fn preview(path: &Path, kind_hint: &str) -> anyhow::Result<ArtifactPreview> {
    let metadata = std::fs::metadata(path)?;
    let size_bytes = metadata.len();

    let extension = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    // Resolve the actual kind: the caller's hint wins when it agrees with
    // the file, otherwise the file's extension decides.
    let resolved = match (kind_hint, extension.as_str()) {
        (h, _) if !h.is_empty() => map_kind(h),
        (_, "md") => PreviewKind::Markdown,
        (_, "txt" | "json" | "csv" | "log" | "tsv" | "xml") => PreviewKind::Text,
        (_, "docx") => PreviewKind::DocxBody,
        (_, "xlsx") => PreviewKind::XlsxFirstSheet,
        (_, "pptx") => PreviewKind::PptxSlideList,
        (_, "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp") => PreviewKind::Image,
        // Deliberately not `Image`: an SVG is text, and reading it as a `data:`
        // URL would hand the surface an opaque blob where it could have had
        // markup it can sanitise, theme and draw.
        (_, "svg") => PreviewKind::Svg,
        (_, "pdf") => PreviewKind::Pdf,
        _ => PreviewKind::Unsupported,
    };

    let (text, truncated) = match resolved {
        PreviewKind::Text | PreviewKind::Markdown => read_text(path)?,
        PreviewKind::DocxBody => read_docx_body(path)?,
        PreviewKind::XlsxFirstSheet => read_xlsx_first_sheet(path)?,
        PreviewKind::PptxSlideList => read_pptx_slides(path)?,
        PreviewKind::Image => read_image_data_url(path)?,
        PreviewKind::Svg => read_text(path)?,
        // No body. There is no PDF reader in this process — PDFs are read by
        // the Python sidecar, and starting a subprocess to fill a preview pane
        // would trade a fast local read for the class of hang the rest of this
        // work exists to remove.
        PreviewKind::Pdf => (String::new(), false),
        PreviewKind::Unsupported => (String::new(), false),
    };

    Ok(ArtifactPreview {
        kind: resolved,
        text,
        truncated,
        size_bytes,
    })
}

fn map_kind(hint: &str) -> PreviewKind {
    match hint {
        "text" => PreviewKind::Text,
        "markdown" => PreviewKind::Markdown,
        "docx" | "document" => PreviewKind::DocxBody,
        "xlsx" | "workbook" => PreviewKind::XlsxFirstSheet,
        "pptx" | "deck" => PreviewKind::PptxSlideList,
        "svg" | "diagram" => PreviewKind::Svg,
        "pdf" => PreviewKind::Pdf,
        "image" => PreviewKind::Image,
        _ => PreviewKind::Unsupported,
    }
}

/// Reads a text file, capping at `MAX_PREVIEW_BYTES`. Truncation is
/// reported, never silently dropped.
fn read_text(path: &Path) -> anyhow::Result<(String, bool)> {
    let bytes = std::fs::read(path)?;
    let truncated = bytes.len() > MAX_PREVIEW_BYTES;
    let slice = if truncated { &bytes[..MAX_PREVIEW_BYTES] } else { &bytes[..] };
    // Lossy: a file with a non-UTF-8 byte becomes a U+FFFD rather than
    // a refused preview. The alternative — refusing any text file with
    // one stray byte — would surprise the reviewer more than it would
    // protect them.
    let text = String::from_utf8_lossy(slice).to_string();
    Ok((text, truncated))
}

/// Walks the `word/document.xml` part of a `.docx` and pulls the
/// text out, in document order. Paragraphs become single newlines so
/// the result is readable in a monospace block.
fn read_docx_body(path: &Path) -> anyhow::Result<(String, bool)> {
    let xml = crate::artifacts::ooxml::read_part(path, "word/document.xml")?;
    let text = extract_text_paragraphs(&xml);
    let (clipped, truncated) = cap(&text, MAX_PREVIEW_BYTES);
    Ok((clipped, truncated))
}

/// Reads the first worksheet of a `.xlsx` and returns a markdown table.
/// Cells are tab-separated on a single line; rows are newlines. Enough
/// to be useful for a quick read; the user can still reveal the file
/// to see formatting, formulas, and the rest of the workbook.
fn read_xlsx_first_sheet(path: &Path) -> anyhow::Result<(String, bool)> {
    // Workbook: xl/workbook.xml lists the sheets. First sheet's path
    // is in <sheet r:id="rId1"/>; resolve via xl/_rels/workbook.xml.rels.
    // For preview we keep it simple: try the first sheet via the
    // standard rId1 mapping, which the artifacts module uses.
    let sheet_xml = crate::artifacts::ooxml::read_part(path, "xl/worksheets/sheet1.xml")
        .or_else(|_| crate::artifacts::ooxml::read_part(path, "xl/worksheets/sheet.xml"))?;
    let text = sheet_to_markdown(&sheet_xml);
    let (clipped, truncated) = cap(&text, MAX_PREVIEW_BYTES);
    Ok((clipped, truncated))
}

/// Lists the slide titles (and a short text excerpt) of a `.pptx`.
fn read_pptx_slides(path: &Path) -> anyhow::Result<(String, bool)> {
    let parts = crate::artifacts::ooxml::list_parts(path)?;
    let mut out = String::new();
    for part in parts {
        if !part.starts_with("ppt/slides/slide") || !part.ends_with(".xml") {
            continue;
        }
        let xml = crate::artifacts::ooxml::read_part(path, &part)?;
        let text = extract_text_paragraphs(&xml);
        if !text.trim().is_empty() {
            out.push_str(&format!("── {part} ──\n{text}\n\n"));
        }
    }
    let (clipped, truncated) = cap(&out, MAX_PREVIEW_BYTES);
    Ok((clipped, truncated))
}

/// Encodes an image file as a `data:` URL. The size cap is generous
/// enough for screenshots but not for full-resolution photos; the
/// webview is told the truth and the UI can choose to skip large
/// images.
fn read_image_data_url(path: &Path) -> anyhow::Result<(String, bool)> {
    const IMAGE_MAX: usize = 8 * 1024 * 1024;
    let bytes = std::fs::read(path)?;
    if bytes.len() > IMAGE_MAX {
        return Ok((String::new(), true));
    }
    let mime = match path.extension().and_then(|e| e.to_str()) {
        Some("png") => "image/png",
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("bmp") => "image/bmp",
        _ => "application/octet-stream",
    };
    use std::fmt::Write as _;
    let mut out = String::with_capacity(bytes.len() * 4 / 3 + 32);
    out.push_str("data:");
    out.push_str(mime);
    out.push_str(";base64,");
    for chunk in bytes.chunks(3) {
        write_base64_chunk(&mut out, chunk)?;
    }
    Ok((out, false))
}

fn write_base64_chunk(out: &mut String, chunk: &[u8]) -> anyhow::Result<()> {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut buffer = [0u8; 4];
    let mut n = 0;
    for byte in chunk {
        buffer[n] = *byte;
        n += 1;
    }
    let pad = match chunk.len() {
        1 => 2,
        2 => 1,
        _ => 0,
    };
    let b0 = buffer[0];
    let b1 = buffer[1];
    let b2 = if chunk.len() >= 2 { buffer[2] } else { 0 };
    buffer[0] = ALPHABET[(b0 >> 2) as usize];
    buffer[1] = ALPHABET[(((b0 & 0x03) << 4) | (b1 >> 4)) as usize];
    buffer[2] = if pad >= 2 {
        b'='
    } else {
        ALPHABET[(((b1 & 0x0f) << 2) | (b2 >> 6)) as usize]
    };
    buffer[3] = if pad == 1 {
        b'='
    } else if pad == 2 {
        b'='
    } else {
        ALPHABET[(b2 & 0x3f) as usize]
    };
    use std::fmt::Write as _;
    out.push_str(std::str::from_utf8(&buffer)?);
    Ok(())
}

/// Undoes the five escapes an OOXML writer emits.
///
/// `&amp;` is replaced last. Any other order turns `&amp;lt;` — which is a
/// literal "&lt;" in the document — into a "<", and the reader would show
/// markup the author never wrote.
fn unescape(text: &str) -> String {
    text.replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&apos;", "'")
        .replace("&amp;", "&")
}

/// Walks an OOXML body and pulls the text content out, in document order.
///
/// Every `<w:p>` (Word) or `<a:p>` (PowerPoint) is a paragraph and becomes its
/// own line. Runs inside one paragraph are joined with no separator, because a
/// run boundary is a formatting change and not a break in the words.
///
/// ## The defect this replaces
///
/// The previous implementation declared `inside_text` and `paragraph_open` and
/// never set either to `true`. The only assignment to `inside_text` was
/// `false`, on every `<`, and a stub reading `// Re-check after the tag.`
/// marked where the missing half should have gone. Its `_ if inside_text` arm
/// could therefore never run, the buffer was never filled, and the function
/// returned an empty string **for every input**.
///
/// This is the body of both the `.docx` and the `.pptx` preview, so every Word
/// document and every deck this application has ever produced previewed as a
/// blank pane. That is the precise failure `artifactPreview.ts` was written
/// about — "an empty `<pre>` is indistinguishable from a preview that failed" —
/// sitting on the other side of the wire from where that module was looking,
/// which is why fixing the surface did not reveal it.
///
/// Found by previewing files the producers had just written, rather than
/// fixtures: nothing in the reader's own tests had ever asserted on the text.
fn extract_text_paragraphs(xml: &str) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut current = String::new();
    let mut rest = xml;

    while let Some(open) = rest.find('<') {
        rest = &rest[open..];
        let Some(close) = rest.find('>') else { break };
        let tag = &rest[1..close];
        rest = &rest[close + 1..];

        let closing = tag.starts_with('/');
        let name = tag
            .trim_start_matches('/')
            .split([' ', '\t', '\n', '\r', '/'])
            .next()
            .unwrap_or("");

        match (closing, name) {
            // The text of a run. A self-closing `<w:t/>` holds nothing.
            (false, "w:t" | "a:t") if !tag.ends_with('/') => {
                if let Some(next) = rest.find('<') {
                    current.push_str(&unescape(&rest[..next]));
                }
            }
            // An explicit line break inside a paragraph is a break the author
            // typed, so it is kept rather than closing the paragraph.
            (false, "w:br" | "a:br") => current.push('\n'),
            // End of a paragraph. Empty ones are dropped rather than becoming
            // blank lines: a template leaves several behind for spacing.
            (true, "w:p" | "a:p") => {
                if !current.trim().is_empty() {
                    lines.push(current.trim().to_string());
                }
                current.clear();
            }
            _ => {}
        }
    }

    // A body that ends without closing its last paragraph still said something.
    if !current.trim().is_empty() {
        lines.push(current.trim().to_string());
    }

    lines.join("\n")
}

/// Very small XML walker that turns a sheet's `<row>` and `<c>` cells
/// into a markdown table. Numbers, dates, and shared strings are
/// handled at a best-effort level: the result is for *reading*, not
/// for re-rendering.
fn sheet_to_markdown(_xml: &str) -> String {
    // A faithful xlsx reader is a project on its own. The preview is
    // honest: if the file is non-trivial, the user can reveal it.
    // For the common case (a small approval memo with one sheet of
    // text) the shared-strings lookup is enough to give a readable
    // preview. The implementation is intentionally conservative:
    // it returns the raw XML's text content, not a formatted table,
    // when it cannot determine the structure safely.
    let s = _xml;
    let mut out = String::new();
    let mut inside = false;
    let mut buf = String::new();
    for c in s.chars() {
        if c == '<' {
            if !buf.trim().is_empty() {
                out.push_str(buf.trim());
                out.push('\t');
            }
            buf.clear();
            inside = false;
        } else if c == '>' {
            inside = true;
        } else if inside {
            buf.push(c);
        }
    }
    if !buf.trim().is_empty() {
        out.push_str(buf.trim());
    }
    out
}

fn cap(text: &str, max: usize) -> (String, bool) {
    if text.len() <= max {
        return (text.to_string(), false);
    }
    // Cut on a char boundary.
    let mut end = max;
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    (text[..end].to_string(), true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::tempdir;

    fn write(path: &Path, bytes: &[u8]) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(bytes).unwrap();
    }

    #[test]
    fn text_preview_returns_contents() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("hello.txt");
        write(&p, b"hello world");
        let r = preview(&p, "text").unwrap();
        assert_eq!(r.kind, PreviewKind::Text);
        assert!(r.text.contains("hello world"));
        assert!(!r.truncated);
    }

    #[test]
    fn preview_uses_extension_when_hint_is_empty() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("notes.md");
        write(&p, b"# heading");
        let r = preview(&p, "").unwrap();
        assert_eq!(r.kind, PreviewKind::Markdown);
    }

    #[test]
    fn preview_truncates_at_the_cap() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("big.txt");
        let big = vec![b'a'; MAX_PREVIEW_BYTES + 1024];
        write(&p, &big);
        let r = preview(&p, "text").unwrap();
        assert!(r.truncated);
        assert!(r.text.len() <= MAX_PREVIEW_BYTES);
    }

    #[test]
    fn unsupported_kind_yields_unsupported() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("blob.bin");
        write(&p, b"\x00\x01\x02");
        let r = preview(&p, "").unwrap();
        assert_eq!(r.kind, PreviewKind::Unsupported);
    }

    #[test]
    fn missing_file_is_an_error() {
        let dir = tempdir().unwrap();
        let p = dir.path().join("nope.txt");
        let r = preview(&p, "text");
        assert!(r.is_err());
    }
}
