//! PDF, written directly.
//!
//! ## Why not a crate
//!
//! `printpdf` was evaluated and measured: it brings roughly two hundred
//! transitive dependencies, among them a complete GUI layout engine, a font
//! configuration stack and several crypto crates. This product installs air
//! gapped, ships an SBOM a reviewer reads, and already writes OOXML by hand for
//! exactly this reason. Two hundred crates to lay out paragraphs of Helvetica
//! is not a trade this codebase makes.
//!
//! What is written here is deliberately narrow: a text document in one of the
//! base fourteen fonts, which every conforming reader already has. No embedded
//! fonts, no images, no transparency. That covers an approval note or a report
//! and stops well short of pretending to be a general PDF library.
//!
//! ## The format, briefly
//!
//! A PDF is a header, a set of numbered objects, a cross-reference table giving
//! each object's byte offset, and a trailer pointing at the table. The offsets
//! are the whole difficulty: they must be counted in bytes, not characters, and
//! a reader will reject the file over a single wrong number. So the writer
//! builds each object first, then measures, rather than writing and hoping.

/// A page of a document, already broken into lines.
#[derive(Debug, Clone)]
struct Page {
    lines: Vec<Line>,
}

#[derive(Debug, Clone)]
struct Line {
    text: String,
    size: f64,
    bold: bool,
    /// Space above this line, in points.
    lead: f64,
}

/// One block of a document.
#[derive(Debug, Clone)]
pub enum Block {
    /// A document or section title.
    Heading(String),
    /// Ordinary prose.
    Paragraph(String),
    /// A bulleted point.
    Bullet(String),
    /// A row of a table, already joined. Drawn in a monospaced face so columns
    /// line up without a layout engine.
    Fixed(String),
}

#[derive(Debug, Clone)]
pub struct PdfSpec {
    pub title: String,
    /// Drawn small at the top of every page, as a classification banner is on
    /// the Word and PowerPoint deliverables.
    pub classification: String,
    pub blocks: Vec<Block>,
}

// A4 at 72 points per inch.
const PAGE_W: f64 = 595.0;
const PAGE_H: f64 = 842.0;
const MARGIN: f64 = 56.0;
const BODY: f64 = 11.0;
const HEADING: f64 = 15.0;
const LINE_H: f64 = 15.0;

/// Escapes a string for a PDF literal.
///
/// A PDF string is delimited by parentheses, so unbalanced ones and backslashes
/// have to be escaped or the object ends early and the file is unreadable.
fn esc(text: &str) -> String {
    let mut out = String::with_capacity(text.len() + 8);
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '(' => out.push_str("\\("),
            ')' => out.push_str("\\)"),
            // WinAnsi covers the base-14 fonts. Anything outside it would be
            // drawn as the wrong glyph, so it is replaced visibly rather than
            // silently mangled.
            c if (c as u32) < 32 || (c as u32) > 255 => out.push('?'),
            c => out.push(c),
        }
    }
    out
}

/// How many characters of a given size fit across the text column.
///
/// Helvetica averages about 0.5 em and Courier is exactly 0.6 em. Approximate
/// for the proportional face, which is why the wrap is conservative: a line
/// that runs slightly short is invisible, one that runs off the page is not.
fn per_line(size: f64, fixed: bool) -> usize {
    let width = PAGE_W - MARGIN * 2.0;
    let em = if fixed { 0.6 } else { 0.5 };
    ((width / (size * em)).floor() as usize).max(8)
}

fn wrap(text: &str, size: f64, fixed: bool) -> Vec<String> {
    let limit = per_line(size, fixed);
    let mut lines = Vec::new();
    let mut current = String::new();
    for word in text.split_whitespace() {
        if current.is_empty() {
            current = word.to_string();
        } else if current.chars().count() + 1 + word.chars().count() <= limit {
            current.push(' ');
            current.push_str(word);
        } else {
            lines.push(std::mem::take(&mut current));
            current = word.to_string();
        }
    }
    if !current.is_empty() {
        lines.push(current);
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

/// Breaks the blocks into pages that fit.
fn paginate(spec: &PdfSpec) -> Vec<Page> {
    let usable = PAGE_H - MARGIN * 2.0 - 24.0;
    let mut pages: Vec<Page> = Vec::new();
    let mut page = Page { lines: Vec::new() };
    let mut used = 0.0;

    let mut push = |line: Line, page: &mut Page, used: &mut f64, pages: &mut Vec<Page>| {
        let needed = LINE_H + line.lead;
        if *used + needed > usable && !page.lines.is_empty() {
            pages.push(std::mem::replace(page, Page { lines: Vec::new() }));
            *used = 0.0;
        }
        *used += needed;
        page.lines.push(line);
    };

    push(
        Line {
            text: spec.title.clone(),
            size: HEADING + 3.0,
            bold: true,
            lead: 0.0,
        },
        &mut page,
        &mut used,
        &mut pages,
    );

    for block in &spec.blocks {
        match block {
            Block::Heading(text) => {
                for (index, line) in wrap(text, HEADING, false).into_iter().enumerate() {
                    push(
                        Line {
                            text: line,
                            size: HEADING,
                            bold: true,
                            lead: if index == 0 { 14.0 } else { 0.0 },
                        },
                        &mut page,
                        &mut used,
                        &mut pages,
                    );
                }
            }
            Block::Paragraph(text) => {
                for (index, line) in wrap(text, BODY, false).into_iter().enumerate() {
                    push(
                        Line {
                            text: line,
                            size: BODY,
                            bold: false,
                            lead: if index == 0 { 8.0 } else { 0.0 },
                        },
                        &mut page,
                        &mut used,
                        &mut pages,
                    );
                }
            }
            Block::Bullet(text) => {
                for (index, line) in wrap(text, BODY, false).into_iter().enumerate() {
                    push(
                        Line {
                            text: if index == 0 {
                                format!("- {line}")
                            } else {
                                format!("  {line}")
                            },
                            size: BODY,
                            bold: false,
                            lead: if index == 0 { 4.0 } else { 0.0 },
                        },
                        &mut page,
                        &mut used,
                        &mut pages,
                    );
                }
            }
            Block::Fixed(text) => {
                for line in wrap(text, BODY, true) {
                    push(
                        Line {
                            text: line,
                            size: BODY,
                            bold: false,
                            // Marked by lead alone; the content stream picks the
                            // monospaced font for these.
                            lead: 2.0,
                        },
                        &mut page,
                        &mut used,
                        &mut pages,
                    );
                }
            }
        }
    }

    if !page.lines.is_empty() {
        pages.push(page);
    }
    pages
}

/// Which font a line is drawn in, by its name in the resource dictionary.
fn font_of(line: &Line, fixed: bool) -> &'static str {
    if fixed {
        "F3"
    } else if line.bold {
        "F2"
    } else {
        "F1"
    }
}

/// Writes the PDF bytes, or says why it cannot.
pub fn render(spec: &PdfSpec) -> Result<Vec<u8>, String> {
    if spec.title.trim().is_empty() {
        return Err("A PDF needs a title. Nothing was written.".to_string());
    }
    if spec.blocks.is_empty() {
        return Err(
            "A PDF with no content would be a blank page. Nothing was written.".to_string(),
        );
    }

    let any_fixed = spec
        .blocks
        .iter()
        .any(|b| matches!(b, Block::Fixed(_)));
    let pages = paginate(spec);

    // Object numbering: 1 catalog, 2 pages, 3..3+n page objects, then contents,
    // then the three fonts.
    let count = pages.len();
    let first_page = 3;
    let first_content = first_page + count;
    let first_font = first_content + count;

    let mut objects: Vec<String> = Vec::new();

    let kids: Vec<String> = (0..count)
        .map(|i| format!("{} 0 R", first_page + i))
        .collect();
    objects.push("<< /Type /Catalog /Pages 2 0 R >>".to_string());
    objects.push(format!(
        "<< /Type /Pages /Count {count} /Kids [{}] >>",
        kids.join(" ")
    ));

    for index in 0..count {
        objects.push(format!(
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] \
             /Resources << /Font << /F1 {} 0 R /F2 {} 0 R /F3 {} 0 R >> >> \
             /Contents {} 0 R >>",
            first_font,
            first_font + 1,
            first_font + 2,
            first_content + index
        ));
    }

    for (index, page) in pages.iter().enumerate() {
        let mut stream = String::new();

        // The classification banner, drawn on every page in small type.
        if !spec.classification.trim().is_empty() {
            stream.push_str(&format!(
                "BT /F1 8 Tf 1 0 0 1 {MARGIN} {} Tm ({}) Tj ET\n",
                PAGE_H - MARGIN + 14.0,
                esc(spec.classification.trim())
            ));
        }

        let mut y = PAGE_H - MARGIN;
        for line in &page.lines {
            y -= LINE_H + line.lead;
            if line.text.is_empty() {
                continue;
            }
            // `any_fixed` decides only whether the monospaced font is used at
            // all; a document without a table never selects it.
            let font = font_of(line, any_fixed && line.lead == 2.0);
            stream.push_str(&format!(
                "BT /{font} {} Tf 1 0 0 1 {MARGIN} {y:.1} Tm ({}) Tj ET\n",
                line.size,
                esc(&line.text)
            ));
        }

        // Page number, so a printed deliverable can be put back in order.
        stream.push_str(&format!(
            "BT /F1 8 Tf 1 0 0 1 {} {} Tm ({} of {count}) Tj ET\n",
            PAGE_W - MARGIN - 40.0,
            MARGIN - 18.0,
            index + 1
        ));

        objects.push(format!(
            "<< /Length {} >>\nstream\n{stream}endstream",
            stream.len()
        ));
    }

    for face in ["Helvetica", "Helvetica-Bold", "Courier"] {
        objects.push(format!(
            "<< /Type /Font /Subtype /Type1 /BaseFont /{face} /Encoding /WinAnsiEncoding >>"
        ));
    }

    // Assemble, measuring as we go: the cross-reference table is byte offsets
    // and nothing else may be guessed.
    let mut out: Vec<u8> = Vec::with_capacity(4096);
    out.extend_from_slice(b"%PDF-1.7\n");
    // A binary comment, so tools treat the file as binary rather than text and
    // do not helpfully rewrite the line endings.
    out.extend_from_slice(b"%\xE2\xE3\xCF\xD3\n");

    let mut offsets = Vec::with_capacity(objects.len());
    for (index, body) in objects.iter().enumerate() {
        offsets.push(out.len());
        out.extend_from_slice(format!("{} 0 obj\n{body}\nendobj\n", index + 1).as_bytes());
    }

    let xref_at = out.len();
    out.extend_from_slice(format!("xref\n0 {}\n", objects.len() + 1).as_bytes());
    out.extend_from_slice(b"0000000000 65535 f \n");
    for offset in &offsets {
        out.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    out.extend_from_slice(
        format!(
            "trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n",
            objects.len() + 1
        )
        .as_bytes(),
    );

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spec() -> PdfSpec {
        PdfSpec {
            title: "Approval Note - PV-2201".to_string(),
            classification: "OFFICIAL".to_string(),
            blocks: vec![
                Block::Heading("Findings".to_string()),
                Block::Paragraph(
                    "Northern Valve Company attended site during the March outage and \
                     replaced the control valve."
                        .to_string(),
                ),
                Block::Bullet("Valve replaced within the shift.".to_string()),
                Block::Fixed("Item      Qty  Cost".to_string()),
            ],
        }
    }

    #[test]
    fn it_writes_a_file_a_reader_will_recognise() {
        let bytes = render(&spec()).expect("pdf");
        assert!(bytes.starts_with(b"%PDF-1.7"), "no header");
        assert!(bytes.ends_with(b"%%EOF\n"), "no trailer");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("/Type /Catalog"));
        assert!(text.contains("/Type /Pages"));
        assert!(text.contains("/Type /Page "));
        assert!(text.contains("startxref"));
    }

    /// The cross-reference table is the part a reader rejects a file over, and
    /// the part no eye can check. Every offset must land on its own object.
    ///
    /// Checked over the raw bytes, never over a lossy string: the header
    /// carries a deliberate binary comment, and decoding it replaces four bytes
    /// with three-byte replacement characters. Every offset after that point
    /// would then be measured against a document that is not the one written -
    /// which is exactly the class of error this test exists to catch.
    #[test]
    fn every_cross_reference_offset_points_at_its_object() {
        let bytes = render(&spec()).expect("pdf");

        const NEWLINE: u8 = 10;
        let marker = b"startxref\n";
        let at = bytes
            .windows(marker.len())
            .rposition(|w| w == marker)
            .expect("startxref");
        let tail = &bytes[at + marker.len()..];
        let digits: Vec<u8> = tail.iter().copied().take_while(u8::is_ascii_digit).collect();
        let xref_at: usize = String::from_utf8(digits).unwrap().parse().expect("offset");

        assert_eq!(&bytes[xref_at..xref_at + 4], b"xref", "startxref misses the table");

        // Entries are fixed width: ten digits, a space, five, a space, a flag,
        // a space, a newline.
        let mut cursor = xref_at;
        while bytes[cursor] != NEWLINE {
            cursor += 1;
        }
        cursor += 1;
        while bytes[cursor] != NEWLINE {
            cursor += 1;
        }
        cursor += 1;
        cursor += 20; // the free entry for object zero

        let mut object = 1usize;
        while cursor + 20 <= bytes.len() && bytes[cursor..cursor + 10].iter().all(u8::is_ascii_digit)
        {
            let offset: usize = std::str::from_utf8(&bytes[cursor..cursor + 10])
                .unwrap()
                .parse()
                .expect("entry");
            let expected = format!("{object} 0 obj");
            assert!(
                bytes[offset..].starts_with(expected.as_bytes()),
                "object {object} offset {offset} points at {:?}, not {expected:?}",
                String::from_utf8_lossy(&bytes[offset..(offset + 20).min(bytes.len())])
            );
            cursor += 20;
            object += 1;
        }
        assert!(object > 5, "only {} entries were checked", object - 1);
    }

    #[test]
    fn the_content_is_actually_in_the_file() {
        let bytes = render(&spec()).expect("pdf");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("Approval Note - PV-2201"));
        assert!(text.contains("Findings"));
        assert!(text.contains("Northern Valve Company"));
        assert!(text.contains("OFFICIAL"));
    }

    /// A stray parenthesis ends a PDF string early and corrupts the file.
    #[test]
    fn a_parenthesis_cannot_end_the_string_it_sits_in() {
        let mut s = spec();
        s.blocks = vec![Block::Paragraph(
            "Flow (measured) was 40 m3/h \\ nominal".to_string(),
        )];
        let bytes = render(&s).expect("pdf");
        let text = String::from_utf8_lossy(&bytes);
        assert!(text.contains("\\(measured\\)"), "unescaped parenthesis");
        assert!(text.contains("\\\\"), "unescaped backslash");
    }

    #[test]
    fn a_long_document_runs_onto_more_than_one_page() {
        let mut s = spec();
        s.blocks = (0..120)
            .map(|i| Block::Paragraph(format!("Paragraph {i} of the inspection record.")))
            .collect();
        let bytes = render(&s).expect("pdf");
        let text = String::from_utf8_lossy(&bytes);
        let pages = text.matches("/Type /Page ").count();
        assert!(pages > 1, "everything was crammed onto one page");
        assert!(text.contains(&format!("of {pages}")), "no page numbering");
    }

    #[test]
    fn every_declared_stream_length_matches_the_stream() {
        let bytes = render(&spec()).expect("pdf");
        let text = String::from_utf8_lossy(&bytes);
        for chunk in text.split("<< /Length ").skip(1) {
            let declared: usize = chunk.split(' ').next().unwrap().parse().expect("length");
            let body = chunk
                .split_once("stream\n")
                .and_then(|(_, rest)| rest.split_once("endstream"))
                .expect("stream body")
                .0;
            assert_eq!(declared, body.len(), "declared length does not match");
        }
    }

    #[test]
    fn it_refuses_rather_than_writing_a_blank_page() {
        let mut s = spec();
        s.blocks.clear();
        assert!(render(&s).unwrap_err().contains("blank page"));

        let mut untitled = spec();
        untitled.title = "  ".to_string();
        assert!(render(&untitled).unwrap_err().contains("needs a title"));
    }

    #[test]
    fn the_same_document_is_written_the_same_way_every_time() {
        assert_eq!(render(&spec()).unwrap(), render(&spec()).unwrap());
    }
}
