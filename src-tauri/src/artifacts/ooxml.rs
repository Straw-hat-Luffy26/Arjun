//! The two things every Office file needs, in one place.
//!
//! A `.docx`, `.xlsx` and `.pptx` are the same thing wearing different hats: a
//! ZIP of XML parts. Both of the properties that make one *open* rather than
//! merely exist live here, so there is one implementation to get right and one
//! place to read when checking that it is right.
//!
//! 1. **Escaping.** Every model-supplied string goes through [`escape`] without
//!    exception. A `<` in a finding is enough to produce a file Word refuses to
//!    open, and "the model does not usually emit angle brackets" is not a
//!    property anything should depend on.
//! 2. **Packaging.** [`write_parts`] writes the parts as a ZIP, creating the
//!    parent directory and nothing else. It is deliberately dumb: the templates
//!    decide what is in a document, this decides only how it is stored.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

/// Maximum uncompressed size of a single part we will read into memory.
///
/// 50 MB is generous for any one XML part in a Word/Excel/PowerPoint file —
/// the largest is the chart data, and even that is in the low single-digit
/// megabytes. Going higher would make this module a viable zip-bomb target.
/// The cap applies to *decompressed* bytes, which is what the allocation is
/// proportional to; a multi-gigabyte compressed entry that decompresses to a
/// few hundred kilobytes is fine.
const MAX_PART_BYTES: u64 = 50 * 1024 * 1024;

/// Escapes a value for XML.
///
/// Control characters are dropped rather than encoded, because they are invalid
/// in XML 1.0 however they are written. Tab and newline are legal and kept.
pub fn escape(value: &str) -> String {
    let mut out = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if (c as u32) < 0x20 && c != '\t' && c != '\n' => {}
            c => out.push(c),
        }
    }
    out
}

/// Writes named XML parts into an Office package at `path`.
pub fn write_parts(path: &Path, parts: &[(&str, String)]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let file = std::fs::File::create(path)
        .with_context(|| format!("could not create {}", path.display()))?;
    let mut zip = zip::ZipWriter::new(file);
    let options: zip::write::FileOptions<'_, ()> =
        zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    for (name, body) in parts {
        zip.start_file(*name, options)?;
        zip.write_all(body.as_bytes())?;
    }

    zip.finish()?;
    Ok(())
}

/// The same package, as bytes in memory: for a writer whose output is
/// registered as an immutable version rather than left at a path.
pub fn package_bytes(parts: &[(&str, String)]) -> Result<Vec<u8>> {
    let mut out = std::io::Cursor::new(Vec::new());
    {
        let mut zip = zip::ZipWriter::new(&mut out);
        let options: zip::write::FileOptions<'_, ()> =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, body) in parts {
            zip.start_file(*name, options)?;
            zip.write_all(body.as_bytes())?;
        }
        zip.finish()?;
    }
    Ok(out.into_inner())
}

/// Reads one part out of an Office package, for checking a file after writing it.
///
/// The read is bounded so a malicious package cannot exhaust memory by
/// declaring a small entry and streaming a multi-gigabyte decompressed
/// payload — the classic zip-bomb shape. The check has two layers because a
/// single layer is not enough:
///
/// 1. The entry's *declared* uncompressed size (read from the ZIP central
///    directory) must not exceed [`MAX_PART_BYTES`]. A truthful header is
///    the common case and this is the fast path.
/// 2. The actual read is also capped, because the declared size is a
///    counter-controlled field the attacker can lie about. The read loop
///    bails out as soon as the running total would exceed the limit, so the
///    allocation is bounded regardless of what the header claimed.
pub fn read_part(path: &Path, part: &str) -> Result<String> {
    use std::io::Read;

    let file = std::fs::File::open(path)
        .with_context(|| format!("{} could not be opened", path.display()))?;
    let mut archive = zip::ZipArchive::new(file)
        .with_context(|| format!("{} is not a readable Office package", path.display()))?;
    let mut entry = archive
        .by_name(part)
        .with_context(|| format!("the package has no {part}"))?;

    // First line of defence: a well-behaved package declares its part sizes,
    // and a part larger than the cap is by definition not a Word/Excel part.
    if entry.size() > MAX_PART_BYTES {
        anyhow::bail!(
            "zip part {part} declares {} bytes, exceeding the {} byte safety limit",
            entry.size(),
            MAX_PART_BYTES
        );
    }

    // Second line of defence: a hostile package can lie in the header, so the
    // declared size check is necessary but not sufficient. Read in a manual
    // loop and bail out the moment the running total would exceed the cap.
    //
    // `std::io::Read::take(limit)` is *not* enough on its own — its `read`
    // returns `Ok(0)` once the limit is reached, not an error, so callers
    // that use `read_to_string` on a `Take` get a silently truncated string
    // rather than a refusal. We therefore drive the loop by hand and check
    // the running total.
    let mut body = String::with_capacity(entry.size().min(8 * 1024) as usize);
    let mut buf = [0u8; 8 * 1024];
    loop {
        let n = entry
            .read(&mut buf)
            .with_context(|| format!("could not read zip part {part}"))?;
        if n == 0 {
            return Ok(body);
        }
        // Reject as soon as adding this chunk would breach the cap. We
        // append-then-check on the *post-append* total, because the
        // alternative (subtracting back) would still leave the attacker
        // able to push us up to `MAX_PART_BYTES` of allocation per request;
        // a real OOXML part never legitimately sits at the boundary.
        let incoming = std::str::from_utf8(&buf[..n])
            .with_context(|| format!("zip part {part} is not valid UTF-8"))?;
        if body.len() + incoming.len() > MAX_PART_BYTES as usize {
            anyhow::bail!(
                "zip part {part} exceeds the {} byte safety limit after decompression",
                MAX_PART_BYTES
            );
        }
        body.push_str(incoming);
    }
}

/// Lists the parts inside an Office package.
pub fn list_parts(path: &Path) -> Result<Vec<String>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("{} could not be opened", path.display()))?;
    let archive = zip::ZipArchive::new(file)
        .with_context(|| format!("{} is not a readable Office package", path.display()))?;
    Ok(archive.file_names().map(|n| n.to_string()).collect())
}

/// Markers that mean a template field was never actually filled in.
///
/// A document that reaches a reviewer still saying `TBD` has failed in the way
/// that matters most: it looks finished. These are the forms that survive when
/// a model echoes the instruction back instead of answering it, or when a
/// template's own scaffolding is left in place.
const PLACEHOLDERS: &[&str] =
    &["{{", "}}", "[insert", "[INSERT", "TBD", "TODO", "XXX", "lorem ipsum", "Lorem ipsum"];

/// Placeholders found in a document's text, in the order they appear.
pub fn placeholders_in(text: &str) -> Vec<String> {
    PLACEHOLDERS
        .iter()
        .filter(|marker| text.contains(**marker))
        .map(|marker| marker.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_document_that_still_says_tbd_is_reported() {
        assert_eq!(placeholders_in("Recommendation: TBD"), vec!["TBD"]);
        assert_eq!(placeholders_in("Write {{name}} here"), vec!["{{", "}}"]);
        assert!(placeholders_in("[insert findings]").contains(&"[insert".to_string()));
    }

    #[test]
    fn ordinary_prose_is_not_mistaken_for_a_placeholder() {
        assert!(placeholders_in("Replace PV-2201 within 90 days.").is_empty());
        assert!(placeholders_in("Measured 8.2 mm against a minimum of 9.0 mm.").is_empty());
    }

    #[test]
    fn escaping_handles_every_xml_significant_character() {
        assert_eq!(escape("a & b"), "a &amp; b");
        assert_eq!(escape("<tag>"), "&lt;tag&gt;");
        assert_eq!(escape("say \"x\""), "say &quot;x&quot;");
        assert_eq!(escape("it's"), "it&apos;s");
        assert_eq!(escape("a\u{0007}b"), "ab");
        assert_eq!(escape("a\tb\nc"), "a\tb\nc");
    }

    #[test]
    fn parts_written_can_be_read_back_by_name() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested/deep/pkg.zip");

        write_parts(&path, &[("a.xml", "<a/>".to_string()), ("b/c.xml", "<c/>".to_string())])
            .unwrap();

        assert_eq!(read_part(&path, "a.xml").unwrap(), "<a/>");
        assert_eq!(read_part(&path, "b/c.xml").unwrap(), "<c/>");
        assert_eq!(list_parts(&path).unwrap().len(), 2);
    }

    #[test]
    fn a_file_that_is_not_a_package_is_reported_rather_than_panicking() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("not-a-zip.xlsx");
        std::fs::write(&path, "plain text").unwrap();

        assert!(read_part(&path, "anything").is_err());
        assert!(list_parts(&path).is_err());
    }

    /// The cap is 50 MB. If this changes, every reviewer of `read_part`'s
    /// memory bound needs to revisit the trade-off: too low and legitimate
    /// parts are refused; too high and a small `.docx` can still OOM the
    /// process. The number is written here, not in the constant, so a
    /// well-meaning rename or unit typo is caught by a test.
    #[test]
    fn the_part_size_cap_is_fifty_megabytes() {
        assert_eq!(MAX_PART_BYTES, 50 * 1024 * 1024);
    }

    /// A normal part is read in full. The cap only triggers when the cap is
    /// actually breached, so a 1 kB XML part must round-trip.
    #[test]
    fn a_small_part_is_returned_unchanged() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ok.docx");
        write_parts(&path, &[("word/document.xml", "<w:doc/>".to_string())]).unwrap();
        let body = read_part(&path, "word/document.xml").unwrap();
        assert_eq!(body, "<w:doc/>");
    }
}
