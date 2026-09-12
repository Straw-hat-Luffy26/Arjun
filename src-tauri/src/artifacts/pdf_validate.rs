//! Reading a produced PDF back, to find out whether it is really a PDF.
//!
//! ## Why this exists
//!
//! Until this module, a produced PDF was checked by
//! `Kind::Pdf => report(true, "Present, N byte(s).")` — the file is on disk and
//! is not empty. A PDF truncated halfway through its object table, or one whose
//! cross-reference offsets point into the middle of a content stream, passed
//! that check and was handed to somebody as a finished deliverable. They found
//! out when they opened it.
//!
//! The comment at that site argued that claiming to have verified a PDF "would
//! be inventing a standard this has no way to hold anything to". That was true
//! when there was nothing to read the file with. It is not true now:
//! [`super::pdf`] writes a real object graph, a real cross-reference table and
//! a real trailer, so those are exactly the things a reader can hold it to.
//!
//! ## What is checked, and what is not
//!
//! Checked, by parsing:
//!
//! - the header, and that the file ends in `%%EOF` — the truncation test
//! - `startxref`, and that it points at the `xref` keyword
//! - every cross-reference offset, and that each lands on its own `N 0 obj`
//! - the trailer's `/Size` against the number of objects, and `/Root`
//! - the catalogue, its `/Pages`, and the page tree's `/Count` against `/Kids`
//! - every page: that it exists, is `/Type /Page`, and names a `/Contents`
//! - every content stream: that `/Length` matches the bytes between `stream`
//!   and `endstream`
//! - the text drawn by those streams, recovered from the `Tj` operators
//! - `/Info`, and that it carries a title
//!
//! Not checked, and deliberately: whether the layout is *good*. Line spacing,
//! visual balance and whether a table looks like a table are not things this
//! can answer, and a check that cannot fail is worse than no check. Those are
//! quality heuristics and live in [`quality`].
//!
//! ## The parser
//!
//! Written against the subset [`super::pdf`] emits — classic cross-reference
//! tables, uncompressed streams, no object streams, no incremental updates. It
//! is a *verifier for our own writer*, not a general PDF reader, and it says so
//! rather than pretending to be one: a file it cannot parse is reported as one
//! it cannot parse, never as one that passed.

use std::collections::BTreeMap;
use std::path::Path;

/// What reading a PDF back found.
#[derive(Debug, Clone, Default)]
pub struct PdfCheck {
    /// Whether the object graph parsed at all.
    pub opens: bool,
    /// Pages reachable from the catalogue.
    pub pages: usize,
    /// Characters recovered from the content streams' `Tj` operators.
    pub characters: usize,
    /// Whether `/Info` is present and carries a title.
    pub has_metadata: bool,
    /// Objects the cross-reference table declares.
    pub objects: usize,
    /// Everything recovered from the page streams, for the quality pass.
    pub text: String,
    pub problems: Vec<String>,
}

impl PdfCheck {
    pub fn is_sound(&self) -> bool {
        self.problems.is_empty() && self.opens
    }
}

/// Reads a PDF from disk and checks it.
pub fn check_pdf(path: &Path) -> PdfCheck {
    match std::fs::read(path) {
        Ok(bytes) => check_pdf_bytes(&bytes),
        Err(error) => PdfCheck {
            problems: vec![format!("the file could not be read: {error}")],
            ..Default::default()
        },
    }
}

/// Finds the last occurrence of `needle`, which is what `startxref` requires:
/// an incrementally-updated file has several, and the last one is current.
fn rfind(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() {
        return None;
    }
    (0..=haystack.len() - needle.len()).rev().find(|&i| &haystack[i..i + needle.len()] == needle)
}

fn find_from(haystack: &[u8], needle: &[u8], from: usize) -> Option<usize> {
    if needle.is_empty() || needle.len() > haystack.len() || from > haystack.len() - needle.len() {
        return None;
    }
    (from..=haystack.len() - needle.len()).find(|&i| &haystack[i..i + needle.len()] == needle)
}

/// Reads the decimal integer starting at `at`, and where it ended.
fn integer_at(bytes: &[u8], at: usize) -> Option<(usize, usize)> {
    let mut i = at;
    while i < bytes.len() && bytes[i].is_ascii_whitespace() {
        i += 1;
    }
    let start = i;
    while i < bytes.len() && bytes[i].is_ascii_digit() {
        i += 1;
    }
    if i == start {
        return None;
    }
    std::str::from_utf8(&bytes[start..i])
        .ok()
        .and_then(|s| s.parse().ok())
        .map(|value| (value, i))
}

/// The text following a `/Name` key in a dictionary body, up to the next
/// delimiter. Enough for `/Count 3`, `/Root 1 0 R` and `/Type /Page`.
fn dict_value<'a>(body: &'a str, key: &str) -> Option<&'a str> {
    let at = body.find(key)? + key.len();
    let rest = body[at..].trim_start();
    if rest.starts_with('/') {
        // The value is itself a name: `/Type /Page`.
        let inner = &rest[1..];
        let end = inner
            .find(|c: char| c.is_whitespace() || c == '/' || c == '>')
            .unwrap_or(inner.len());
        return Some(&inner[..end]);
    }
    let end = rest
        .find(|c: char| c == '/' || c == '>' || c == '[' || c == '\n')
        .unwrap_or(rest.len());
    Some(rest[..end].trim())
}

/// `1 0 R` -> `1`.
fn reference(body: &str, key: &str) -> Option<usize> {
    dict_value(body, key)?.split_whitespace().next()?.parse().ok()
}

/// Every PDF string literal in a content stream, unescaped.
///
/// Only meaningful for a stream that paints — a stream with parentheses and no
/// `Tj` draws nothing, and counting it would let a blank page pass the "has
/// text" check.
fn drawn_text(stream: &str) -> String {
    if !stream.contains("Tj") {
        return String::new();
    }
    let chars: Vec<char> = stream.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] != '(' {
            i += 1;
            continue;
        }
        let mut depth = 1;
        i += 1;
        let mut literal = String::new();
        while i < chars.len() && depth > 0 {
            match chars[i] {
                '\\' if i + 1 < chars.len() => {
                    let next = chars[i + 1];
                    if next.is_digit(8) {
                        // Octal triple: how `pdf::esc` encodes anything above
                        // ASCII. Missing these would under-count every
                        // document with a dash or a curly quote in it.
                        let mut code = 0u32;
                        let mut digits = 0;
                        let mut j = i + 1;
                        while j < chars.len() && digits < 3 && chars[j].is_digit(8) {
                            code = code * 8 + chars[j].to_digit(8).unwrap_or(0);
                            j += 1;
                            digits += 1;
                        }
                        if let Some(c) = char::from_u32(code) {
                            literal.push(c);
                        }
                        i = j;
                    } else {
                        literal.push(next);
                        i += 2;
                    }
                }
                '(' => {
                    depth += 1;
                    literal.push('(');
                    i += 1;
                }
                ')' => {
                    depth -= 1;
                    if depth > 0 {
                        literal.push(')');
                    }
                    i += 1;
                }
                c => {
                    literal.push(c);
                    i += 1;
                }
            }
        }
        out.push_str(&literal);
        out.push(' ');
    }
    out
}

/// Parses and checks a PDF's bytes.
pub fn check_pdf_bytes(bytes: &[u8]) -> PdfCheck {
    let mut check = PdfCheck::default();

    if bytes.len() < 32 {
        check.problems.push(format!("the file is {} bytes; a PDF cannot be", bytes.len()));
        return check;
    }
    if !bytes.starts_with(b"%PDF-") {
        check.problems.push("the file does not begin with a PDF header".to_string());
        return check;
    }

    // Truncation. Everything below reads offsets, and an offset into a file
    // that stops early is the failure this catches first.
    let tail = &bytes[bytes.len().saturating_sub(32)..];
    if rfind(tail, b"%%EOF").is_none() {
        check.problems.push(
            "the file does not end with %%EOF, so it was truncated before it was finished"
                .to_string(),
        );
        return check;
    }

    let Some(startxref_at) = rfind(bytes, b"startxref") else {
        check
            .problems
            .push("there is no startxref, so the object table cannot be found".to_string());
        return check;
    };
    let Some((xref_at, _)) = integer_at(bytes, startxref_at + b"startxref".len()) else {
        check.problems.push("startxref is not followed by an offset".to_string());
        return check;
    };
    if xref_at + 4 > bytes.len() || &bytes[xref_at..xref_at + 4] != b"xref" {
        check
            .problems
            .push(format!("startxref points at byte {xref_at}, which is not the xref table"));
        return check;
    }

    // The table: `xref`, then `first count`, then `count` twenty-byte entries.
    let Some((first, after_first)) = integer_at(bytes, xref_at + 4) else {
        check.problems.push("the xref table has no first-object number".to_string());
        return check;
    };
    let Some((count, after_count)) = integer_at(bytes, after_first) else {
        check.problems.push("the xref table has no object count".to_string());
        return check;
    };
    check.objects = count.saturating_sub(1);

    let mut cursor = after_count;
    while cursor < bytes.len() && (bytes[cursor] == b'\r' || bytes[cursor] == b'\n') {
        cursor += 1;
    }

    let mut offsets: BTreeMap<usize, usize> = BTreeMap::new();
    for index in 0..count {
        if cursor + 20 > bytes.len() {
            check.problems.push(format!(
                "the xref table claims {count} entries and the file ends after {index}"
            ));
            return check;
        }
        let entry = &bytes[cursor..cursor + 20];
        if entry[17] == b'n' {
            let Some((offset, _)) = integer_at(entry, 0) else {
                check.problems.push(format!("xref entry {index} has no offset"));
                return check;
            };
            let object = first + index;
            if offset >= bytes.len() {
                check.problems.push(format!(
                    "object {object} is recorded at byte {offset}, past the end of a {}-byte file",
                    bytes.len()
                ));
                return check;
            }
            let expected = format!("{object} 0 obj");
            let there = &bytes[offset..(offset + expected.len()).min(bytes.len())];
            if there != expected.as_bytes() {
                check.problems.push(format!(
                    "object {object} is recorded at byte {offset}, which holds {:?} rather than \
                     {expected:?}",
                    String::from_utf8_lossy(&there[..there.len().min(20)])
                ));
                return check;
            }
            offsets.insert(object, offset);
        }
        cursor += 20;
    }

    // The trailer.
    let Some(trailer_at) = rfind(bytes, b"trailer") else {
        check.problems.push("there is no trailer".to_string());
        return check;
    };
    let trailer = String::from_utf8_lossy(&bytes[trailer_at..]).to_string();
    let Some(size) = dict_value(&trailer, "/Size").and_then(|v| v.trim().parse::<usize>().ok())
    else {
        check.problems.push("the trailer has no /Size".to_string());
        return check;
    };
    if size != count {
        check.problems.push(format!(
            "the trailer says /Size {size} and the xref table holds {count} entries"
        ));
    }
    let Some(root) = reference(&trailer, "/Root") else {
        check.problems.push("the trailer has no /Root".to_string());
        return check;
    };

    let body_of = |object: usize| -> Option<String> {
        let offset = *offsets.get(&object)?;
        let start = find_from(bytes, b"obj", offset)? + 3;
        let end = find_from(bytes, b"endobj", start)?;
        Some(String::from_utf8_lossy(&bytes[start..end]).to_string())
    };

    let Some(catalog) = body_of(root) else {
        check
            .problems
            .push(format!("/Root points at object {root}, which is not in the file"));
        return check;
    };
    if !catalog.contains("/Catalog") {
        check
            .problems
            .push(format!("object {root} is named as /Root but is not a /Catalog"));
    }
    let Some(pages_ref) = reference(&catalog, "/Pages") else {
        check.problems.push("the catalogue has no /Pages".to_string());
        return check;
    };
    let Some(pages) = body_of(pages_ref) else {
        check
            .problems
            .push(format!("/Pages points at object {pages_ref}, which is not in the file"));
        return check;
    };

    let declared: usize = dict_value(&pages, "/Count")
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(0);

    let kids: Vec<usize> = pages
        .find("/Kids")
        .and_then(|at| {
            let rest = &pages[at..];
            let open = rest.find('[')?;
            let close = rest.find(']')?;
            Some(rest[open + 1..close].to_string())
        })
        .map(|list| {
            list.split(" R")
                .filter_map(|item| item.split_whitespace().next()?.parse().ok())
                .collect()
        })
        .unwrap_or_default();

    if kids.len() != declared {
        check.problems.push(format!(
            "the page tree says /Count {declared} and lists {} kid(s)",
            kids.len()
        ));
    }
    if kids.is_empty() {
        check.problems.push("the document has no pages".to_string());
        return check;
    }
    check.pages = kids.len();

    let mut recovered = String::new();
    let mut pages_without_text = Vec::new();
    for (index, kid) in kids.iter().enumerate() {
        let Some(page) = body_of(*kid) else {
            check.problems.push(format!(
                "page {} is object {kid}, which is not in the file",
                index + 1
            ));
            continue;
        };
        if !page.contains("/Page") {
            check
                .problems
                .push(format!("object {kid} is in the page tree but is not a /Page"));
        }
        let Some(contents) = reference(&page, "/Contents") else {
            check.problems.push(format!("page {} names no /Contents", index + 1));
            continue;
        };
        let Some(offset) = offsets.get(&contents) else {
            check.problems.push(format!(
                "page {}'s content stream is object {contents}, which is not in the file",
                index + 1
            ));
            continue;
        };
        let Some(stream_at) = find_from(bytes, b"stream", *offset) else {
            check
                .problems
                .push(format!("page {}'s content object has no stream", index + 1));
            continue;
        };
        let Some(end_at) = find_from(bytes, b"endstream", stream_at) else {
            check.problems.push(format!(
                "page {}'s content stream is never closed, so the file is damaged",
                index + 1
            ));
            continue;
        };
        let header = String::from_utf8_lossy(&bytes[*offset..stream_at]).to_string();
        let declared_length: usize = dict_value(&header, "/Length")
            .and_then(|v| v.trim().parse().ok())
            .unwrap_or(0);
        let body_start = stream_at + b"stream".len() + 1; // the newline after `stream`
        let actual = end_at.saturating_sub(body_start);
        if declared_length != actual {
            check.problems.push(format!(
                "page {}'s content stream declares /Length {declared_length} and holds {actual} \
                 bytes",
                index + 1
            ));
        }
        let stream = String::from_utf8_lossy(&bytes[body_start..end_at]).to_string();
        let text = drawn_text(&stream);
        if text.trim().is_empty() {
            pages_without_text.push(index + 1);
        }
        recovered.push_str(&text);
    }
    check.characters = recovered.chars().filter(|c| !c.is_whitespace()).count();
    check.text = recovered;

    if !pages_without_text.is_empty() {
        check.problems.push(format!(
            "page(s) {} draw no text at all",
            pages_without_text.iter().map(usize::to_string).collect::<Vec<_>>().join(", ")
        ));
    }
    if check.characters == 0 {
        check.problems.push("the document contains no text".to_string());
    }

    // `/Info`, which is what a reader's Properties panel shows.
    match reference(&trailer, "/Info") {
        Some(info_ref) => match body_of(info_ref) {
            Some(info) => {
                check.has_metadata = info.contains("/Title");
                if !check.has_metadata {
                    check.problems.push("the document information has no /Title".to_string());
                }
            }
            None => check.problems.push(format!(
                "/Info points at object {info_ref}, which is not in the file"
            )),
        },
        None => check
            .problems
            .push("the file carries no document information (/Info)".to_string()),
    }

    check.opens = true;
    check
}

/// Quality checks: things a professional document has that a merely valid one
/// may not.
///
/// Kept apart from [`check_pdf_bytes`] because they are a different kind of
/// claim. A failure here means the file opens and is structurally sound but is
/// not something to hand over; a failure there means it is broken.
pub mod quality {
    use super::PdfCheck;

    /// The least text a page of a real deliverable carries.
    ///
    /// A page holding only a heading is a page somebody will ask about. Low
    /// enough that a genuine title page passes.
    const MIN_CHARS_PER_PAGE: usize = 40;

    /// Text that must never reach a reader.
    const PLACEHOLDERS: &[&str] = &["lorem ipsum", "tbd", "todo", "placeholder", "[insert"];

    pub fn inspect(check: &PdfCheck) -> Vec<String> {
        let mut notes = Vec::new();
        if check.pages > 0 {
            let per_page = check.characters / check.pages;
            if per_page < MIN_CHARS_PER_PAGE {
                notes.push(format!(
                    "averages {per_page} characters a page, which is too sparse to be a finished \
                     document"
                ));
            }
        }
        let lowered = check.text.to_lowercase();
        for marker in PLACEHOLDERS {
            if lowered.contains(marker) {
                notes.push(format!("contains the placeholder {marker:?}"));
            }
        }
        notes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::pdf::{render, Block, PdfSpec};

    fn spec() -> PdfSpec {
        PdfSpec {
            title: "Pump PV-2201 Inspection".to_string(),
            classification: "OFFICIAL".to_string(),
            blocks: vec![
                Block::Heading("Findings".to_string()),
                Block::Paragraph(
                    "The casing was examined during the March outage and the wear ring \
                     clearance was measured at 0.45 mm against a stated limit of 0.60 mm."
                        .to_string(),
                ),
                Block::Bullet("Clearance is within the tolerance stated in the manual.".to_string()),
                Block::Fixed("Tag      Reading   Limit".to_string()),
            ],
        }
    }

    #[test]
    fn a_document_this_product_wrote_reads_back_sound() {
        let bytes = render(&spec()).expect("renders");
        let check = check_pdf_bytes(&bytes);
        assert!(check.is_sound(), "{:?}", check.problems);
        assert!(check.opens);
        assert!(check.pages >= 1);
        assert!(check.characters > 0, "no text was recovered from the page");
    }

    /// Each case below passed the old "Present, N bytes" check.
    #[test]
    fn a_truncated_file_is_rejected() {
        let bytes = render(&spec()).expect("renders");
        let check = check_pdf_bytes(&bytes[..bytes.len() / 2]);
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("truncated")), "{:?}", check.problems);
    }

    #[test]
    fn a_cross_reference_pointing_nowhere_is_rejected() {
        let mut bytes = render(&spec()).expect("renders");
        // Move the first object's recorded offset to a byte that is not an
        // object header. The file still ends in %%EOF and is still the right
        // length, so only reading the table finds this.
        let at = rfind(&bytes, b"0000000000 65535 f \n").expect("free entry") + 20;
        bytes[at..at + 10].copy_from_slice(b"0000000999");
        let check = check_pdf_bytes(&bytes);
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("rather than")), "{:?}", check.problems);
    }

    #[test]
    fn a_startxref_that_misses_the_table_is_rejected() {
        let bytes = render(&spec()).expect("renders");
        let text = String::from_utf8_lossy(&bytes).to_string();
        let at = text.rfind("startxref\n").expect("startxref");
        let broken = format!("{}startxref\n17\n%%EOF\n", &text[..at]);
        let check = check_pdf_bytes(broken.as_bytes());
        assert!(!check.is_sound());
        assert!(
            check.problems.iter().any(|p| p.contains("not the xref table")),
            "{:?}",
            check.problems
        );
    }

    #[test]
    fn a_file_that_is_not_a_pdf_is_rejected() {
        let check = check_pdf_bytes(b"This is a text file that somebody renamed to .pdf.\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("PDF header")));
    }

    #[test]
    fn an_empty_file_is_rejected() {
        assert!(!check_pdf_bytes(b"").is_sound());
    }

    /// A byte-level find, because a PDF is not UTF-8.
    ///
    /// `String::from_utf8_lossy` on these bytes replaces the writer's binary
    /// comment with U+FFFD, which is three bytes where there were four. Every
    /// mutation built that way silently moved every cross-reference offset, so
    /// the parser rejected the file for a broken xref and the test passed
    /// while proving nothing about what it named.
    fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
        (0..=haystack.len().saturating_sub(needle.len()))
            .find(|&i| &haystack[i..i + needle.len()] == needle)
    }

    /// A document with no content never becomes a file.
    ///
    /// The writer refuses it outright, which is the better place to stop it:
    /// nothing reaches the disk and nothing has to be checked. Asserted here
    /// so that if the writer ever becomes permissive, this says so rather than
    /// a blank deliverable quietly reaching somebody.
    #[test]
    fn a_document_with_no_content_is_refused_before_a_file_exists() {
        let bare = PdfSpec {
            title: "Bare".to_string(),
            classification: "OFFICIAL".to_string(),
            blocks: vec![],
        };
        let error = render(&bare).expect_err("a blank document must not be written");
        assert!(error.to_lowercase().contains("blank"), "{error}");
    }

    /// A stream that paints nothing is a structural failure.
    #[test]
    fn a_content_stream_that_paints_nothing_is_rejected() {
        let mut bytes = render(&spec()).expect("renders");
        // Same width, so every recorded offset stays correct: the point is to
        // test the text check, not the xref check.
        let mut painted = 0;
        while let Some(at) = find_bytes(&bytes, b" Tj ") {
            bytes[at + 1] = b'T';
            bytes[at + 2] = b'q';
            painted += 1;
        }
        assert!(painted > 0, "the fixture must contain a paint operator");
        let check = check_pdf_bytes(&bytes);
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("no text")), "{:?}", check.problems);
    }

    #[test]
    fn a_content_stream_whose_length_is_wrong_is_rejected() {
        let mut bytes = render(&spec()).expect("renders");
        let at = find_bytes(&bytes, b"/Length ").expect("a stream length") + b"/Length ".len();
        let mut end = at;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        assert!(end > at, "the length must be a number");
        let before = bytes.len();
        for byte in &mut bytes[at..end] {
            *byte = b'9';
        }
        assert_eq!(bytes.len(), before, "the mutation must not move any offset");
        let check = check_pdf_bytes(&bytes);
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("/Length")), "{:?}", check.problems);
    }

    /// The page tree has to agree with itself.
    #[test]
    fn a_page_count_that_disagrees_with_the_kids_is_rejected() {
        let mut bytes = render(&spec()).expect("renders");
        let at = find_bytes(&bytes, b"/Count ").expect("a page count") + b"/Count ".len();
        // One digit, same width: 1 kid declared as 9.
        assert!(bytes[at].is_ascii_digit());
        bytes[at] = b'9';
        let check = check_pdf_bytes(&bytes);
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("/Count")), "{:?}", check.problems);
    }

    /// Without `/Info` a reader shows the file as untitled, and a document
    /// system has nothing to index it on.
    #[test]
    fn a_document_carries_its_title_in_the_information_dictionary() {
        let bytes = render(&spec()).expect("renders");
        let check = check_pdf_bytes(&bytes);
        assert!(check.has_metadata, "{:?}", check.problems);
        assert!(
            find_bytes(&bytes, b"/Title (Pump PV-2201 Inspection)").is_some(),
            "the title the caller gave must be the title in the file"
        );
    }

    #[test]
    fn text_is_recovered_through_the_escapes_the_writer_uses() {
        let spec = PdfSpec {
            title: "Escapes".to_string(),
            classification: "OFFICIAL".to_string(),
            blocks: vec![Block::Paragraph(
                "A clearance of 0.45 mm (measured) — within tolerance, per the manual.".to_string(),
            )],
        };
        let bytes = render(&spec).expect("renders");
        let check = check_pdf_bytes(&bytes);
        assert!(check.is_sound(), "{:?}", check.problems);
        // The parenthesis and the em dash both go through `esc`, and both have
        // to come back or the "has text" check measures the wrong string.
        assert!(check.characters > 30, "recovered only {} characters", check.characters);
        assert!(check.text.contains("(measured)"), "{:?}", check.text);
    }

    #[test]
    fn quality_notices_a_placeholder() {
        let check = PdfCheck {
            pages: 1,
            characters: 500,
            text: "Findings: TBD".to_string(),
            ..Default::default()
        };
        assert!(quality::inspect(&check).iter().any(|n| n.contains("tbd")));
    }

    #[test]
    fn quality_notices_a_page_with_almost_nothing_on_it() {
        let check = PdfCheck {
            pages: 4,
            characters: 40,
            text: "a heading".to_string(),
            ..Default::default()
        };
        assert!(quality::inspect(&check).iter().any(|n| n.contains("sparse")));
    }
}
