//! Cutting a document into pieces that still know where they came from.
//!
//! This is the highest-leverage thing in the retrieval path, and it is not
//! close. Published work comparing document pipelines found that switching from
//! fixed-size chunking to hierarchical, structure-aware chunking moved
//! table-dependent question accuracy from roughly two-thirds to near-perfect —
//! a far larger effect than swapping the parser underneath it. The reason is
//! simple once stated: a chunk cut at 500 characters lands mid-sentence, mid-row
//! and mid-argument, and arrives at the model with no idea which procedure it
//! belonged to.
//!
//! So chunks are cut at the boundaries the *document* has, not at a character
//! count, and every chunk carries its heading trail. A passage that reads
//! "Replace within 90 days" is nearly useless on its own; the same passage
//! carrying `["4 Inspection", "4.2 Wall Thickness"]` is evidence.
//!
//! ## What counts as a boundary
//!
//! Refinery paperwork is heavily numbered — `4.2.1 Minimum Thickness` — and that
//! numbering is the most reliable structural signal available, more so than
//! typography, which a text-layer read discards anyway. Markdown headings are
//! honoured too, since that is what a layout-aware engine emits.
//!
//! ## What is never split
//!
//! A table. A table cut in half produces two chunks, neither of which is a
//! table: the half without the header row has columns nobody can name. Tables
//! are kept whole even when that makes an oversized chunk, because an oversized
//! table beats two useless ones.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::documents::ExtractedDocument;

/// Target size for a prose chunk, in characters.
///
/// Large enough to hold a complete clause of a procedure, small enough that a
/// handful fit in a prompt alongside everything else. Sections shorter than this
/// are never padded — a two-line clause is a complete thought and splitting or
/// merging it would only blur what it says.
const TARGET_CHARS: usize = 1200;

/// A prose section longer than this is split, at paragraph boundaries.
const MAX_CHARS: usize = 2000;

/// Sentences of overlap carried between consecutive pieces of a split section.
///
/// Only within a section that had to be split, never across a heading: the
/// point of overlap is to avoid severing an argument mid-flow, and a heading is
/// exactly where the flow is *supposed* to break.
const OVERLAP_CHARS: usize = 150;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ChunkKind {
    Prose,
    /// Kept whole regardless of size.
    Table,
}

/// One retrievable piece of a document.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Chunk {
    /// Stable across re-chunking of the same document, so a citation made today
    /// still resolves after the document is re-read by a better engine.
    pub id: String,
    pub document_sha256: String,
    pub ordinal: u32,
    pub text: String,
    /// The page this chunk starts on.
    pub page: u32,
    /// Headings above this chunk, outermost first. This is what turns a passage
    /// into evidence.
    pub section_path: Vec<String>,
    pub kind: ChunkKind,
    pub char_count: u32,
}

impl Chunk {
    /// How a citation to this chunk reads.
    pub fn citation(&self) -> String {
        if self.section_path.is_empty() {
            format!("page {}", self.page)
        } else {
            format!("{}, page {}", self.section_path.join(" › "), self.page)
        }
    }
}

/// One page on its way through the cut.
///
/// A borrowed view rather than an owned type: chunking a forty-page scan
/// should not begin by copying it.
struct Page<'a> {
    page: u32,
    text: &'a str,
}

/// A heading, and how deep it sits.
struct Heading {
    level: usize,
    title: String,
}

/// Recognises a heading, and how deep it is.
///
/// Depth matters: `4.2.1` sits under `4.2`, which sits under `4`. Getting that
/// wrong flattens the trail and loses exactly the context the trail exists for.
fn heading_of(line: &str) -> Option<Heading> {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.len() > 120 {
        return None;
    }

    // Markdown, as emitted by a layout-aware engine.
    if let Some(hashes) = trimmed.strip_prefix('#') {
        let level = 1 + hashes.chars().take_while(|c| *c == '#').count();
        let title = trimmed.trim_start_matches('#').trim();
        if !title.is_empty() {
            return Some(Heading {
                level,
                title: title.to_string(),
            });
        }
    }

    // Numbered sections: `4`, `4.2`, `4.2.1`, optionally followed by a title.
    // Depth is the number of components, so the tree matches the numbering.
    let mut parts = trimmed.splitn(2, char::is_whitespace);
    let number = parts.next().unwrap_or("");
    let rest = parts.next().unwrap_or("").trim();

    let stripped = number.trim_end_matches('.');
    let looks_numbered = !stripped.is_empty()
        && stripped.split('.').all(|c| !c.is_empty() && c.chars().all(|d| d.is_ascii_digit()));

    if looks_numbered && !rest.is_empty() {
        // A line that merely starts with a figure — "8.2 mm measured" — is not a
        // heading. A real one is short and does not read as a sentence.
        let reads_as_prose = rest.ends_with('.') || rest.split_whitespace().count() > 12;
        if !reads_as_prose {
            return Some(Heading {
                level: stripped.split('.').count(),
                title: trimmed.to_string(),
            });
        }
    }

    None
}

/// Whether a line is part of a table.
///
/// Markdown pipe tables are what every document engine here emits for tabular
/// data, so that is what is detected.
fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.matches('|').count() >= 2
}

/// Splits a long prose block at paragraph boundaries, with a little overlap.
fn split_prose(text: &str) -> Vec<String> {
    if text.len() <= MAX_CHARS {
        return vec![text.to_string()];
    }

    let mut pieces = Vec::new();
    let mut current = String::new();

    for paragraph in text.split("\n\n") {
        let paragraph = paragraph.trim();
        if paragraph.is_empty() {
            continue;
        }

        if !current.is_empty() && current.len() + paragraph.len() > TARGET_CHARS {
            // Carry the tail of the previous piece so a thought split across the
            // boundary is still readable from either side.
            let tail = if current.len() > OVERLAP_CHARS {
                let start = current
                    .char_indices()
                    .rev()
                    .nth(OVERLAP_CHARS)
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                current[start..].to_string()
            } else {
                String::new()
            };

            pieces.push(std::mem::take(&mut current));
            if !tail.is_empty() {
                current.push_str(tail.trim_start());
                current.push_str("\n\n");
            }
        }

        current.push_str(paragraph);
        current.push_str("\n\n");
    }

    let remainder = current.trim();
    if !remainder.is_empty() {
        pieces.push(remainder.to_string());
    }

    pieces
}

/// Stable id for a chunk. Derived from the document and the ordinal, so a
/// citation survives the document being re-chunked identically.
fn chunk_id(document_sha256: &str, ordinal: u32) -> String {
    let mut hasher = Sha256::new();
    hasher.update(document_sha256.as_bytes());
    hasher.update(b":");
    hasher.update(ordinal.to_string().as_bytes());
    format!("{:x}", hasher.finalize())[..32].to_string()
}

/// Cuts a document into chunks that carry their structure.
pub fn chunk_document(document_sha256: &str, extracted: &ExtractedDocument) -> Vec<Chunk> {
    let pages: Vec<(u32, &str)> = extracted
        .pages
        .iter()
        .map(|page| (page.page, page.text.as_str()))
        .collect();
    chunk_pages(document_sha256, &pages)
}

/// The same cut, over pages that did not come from the document sidecar.
///
/// Split out from [`chunk_document`] so a chat attachment gets **this** logic
/// rather than a second, worse copy of it. The two callers hold their pages in
/// different structures — the collection pipeline has a
/// [`crate::documents::ExtractedDocument`] with confidences and regions, and
/// [`crate::agent_runtime::documents`] has page text and provenance — but the
/// cut is the same cut, and the reason to share it is at the top of this file:
/// structure-aware chunking is what makes a table-dependent question
/// answerable, and arriving by paperclip is not a reason to get the worse
/// pipeline.
///
/// Pages are `(page number, text)` in reading order. The number is the
/// document's own, so a chunk cut from page 31 says 31 whether or not pages
/// 1-30 produced any text at all.
pub fn chunk_pages(document_sha256: &str, pages: &[(u32, &str)]) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let mut ordinal = 0u32;
    // The heading stack, deepest last. Popped when a heading of equal or
    // shallower depth arrives.
    let mut section: Vec<Heading> = Vec::new();

    for page in pages.iter().map(|(number, text)| Page {
        page: *number,
        text: *text,
    }) {
        if page.text.trim().is_empty() {
            continue;
        }

        let mut buffer = String::new();
        let mut table_buffer = String::new();

        // Flushes whatever prose has accumulated under the current heading.
        macro_rules! flush_prose {
            () => {
                let text = buffer.trim().to_string();
                if !text.is_empty() {
                    for piece in split_prose(&text) {
                        chunks.push(Chunk {
                            id: chunk_id(document_sha256, ordinal),
                            document_sha256: document_sha256.to_string(),
                            ordinal,
                            char_count: piece.len() as u32,
                            text: piece,
                            page: page.page,
                            section_path: section.iter().map(|h| h.title.clone()).collect(),
                            kind: ChunkKind::Prose,
                        });
                        ordinal += 1;
                    }
                }
                buffer.clear();
            };
        }

        macro_rules! flush_table {
            () => {
                let text = table_buffer.trim().to_string();
                if !text.is_empty() {
                    // Never split, whatever the size.
                    chunks.push(Chunk {
                        id: chunk_id(document_sha256, ordinal),
                        document_sha256: document_sha256.to_string(),
                        ordinal,
                        char_count: text.len() as u32,
                        text,
                        page: page.page,
                        section_path: section.iter().map(|h| h.title.clone()).collect(),
                        kind: ChunkKind::Table,
                    });
                    ordinal += 1;
                }
                table_buffer.clear();
            };
        }

        for line in page.text.lines() {
            if is_table_line(line) {
                flush_prose!();
                table_buffer.push_str(line);
                table_buffer.push('\n');
                continue;
            }

            if !table_buffer.is_empty() {
                flush_table!();
            }

            if let Some(heading) = heading_of(line) {
                flush_prose!();
                // Everything at this depth or deeper is now closed.
                while section.last().is_some_and(|h| h.level >= heading.level) {
                    section.pop();
                }
                section.push(heading);
                continue;
            }

            buffer.push_str(line);
            buffer.push('\n');
        }

        flush_table!();
        flush_prose!();
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::documents::{EngineCapabilities, EscalationPlan, ExtractedPage, InjectionScan};

    fn document(pages: Vec<&str>) -> ExtractedDocument {
        ExtractedDocument {
            engine: "test".into(),
            engine_version: "1".into(),
            pages: pages
                .into_iter()
                .enumerate()
                .map(|(i, text)| ExtractedPage {
                    page: i as u32 + 1,
                    text: text.to_string(),
                    confidence: 1.0,
                    needs_review: false,
                    review_reason: None,
                    char_count: text.len() as u32,
                    regions: Vec::new(),
                    read_by: None,
                })
                .collect(),
            capabilities: EngineCapabilities::default(),
            warnings: vec![],
            pages_needing_review: 0,
            source_path: "sop.pdf".into(),
            source_bytes: 1,
            injection_scan: InjectionScan::default(),
            escalation: EscalationPlan::default(),
        }
    }

    const SOP: &str = "\
4 Inspection
General requirements for pressure vessel inspection.

4.2 Wall Thickness
Minimum acceptable wall thickness is 9.0 mm.

4.2.1 Measurement
Ultrasonic measurement at four points around the circumference.
";

    #[test]
    fn headings_build_a_nested_trail() {
        let chunks = chunk_document("abc", &document(vec![SOP]));
        let deepest = chunks.last().unwrap();

        assert_eq!(
            deepest.section_path,
            vec!["4 Inspection", "4.2 Wall Thickness", "4.2.1 Measurement"]
        );
    }

    /// The whole reason for the trail: a passage that is meaningless alone
    /// becomes evidence when it says which procedure it came from.
    #[test]
    fn a_citation_names_the_section_and_the_page() {
        let chunks = chunk_document("abc", &document(vec![SOP]));
        let deepest = chunks.last().unwrap();
        assert_eq!(
            deepest.citation(),
            "4 Inspection › 4.2 Wall Thickness › 4.2.1 Measurement, page 1"
        );
    }

    #[test]
    fn a_shallower_heading_closes_the_deeper_ones() {
        let text = "\
4 Inspection
Intro.

4.2.1 Deep
Detail.

5 Maintenance
Different topic.
";
        let chunks = chunk_document("abc", &document(vec![text]));
        let last = chunks.last().unwrap();
        assert_eq!(last.section_path, vec!["5 Maintenance"]);
    }

    /// A measurement is not a heading, however much it looks like a numbered one.
    #[test]
    fn a_line_starting_with_a_figure_is_not_a_heading() {
        assert!(heading_of("8.2 mm measured at the north face.").is_none());
        assert!(heading_of("9.0 mm is the minimum acceptable value for this vessel.").is_none());
        assert!(heading_of("4.2 Wall Thickness").is_some());
    }

    #[test]
    fn markdown_headings_are_honoured_with_their_depth() {
        let text = "# Procedure\nIntro.\n\n## Thickness\nDetail.\n";
        let chunks = chunk_document("abc", &document(vec![text]));
        assert_eq!(chunks.last().unwrap().section_path, vec!["Procedure", "Thickness"]);
    }

    /// A table cut in half produces two things, neither of which is a table.
    #[test]
    fn a_table_is_never_split_and_keeps_its_header() {
        let mut table = String::from("4.3 Results\n\n| Point | Thickness | Limit |\n|---|---|---|\n");
        for i in 0..200 {
            table.push_str(&format!("| P{i} | 8.{i} mm | 9.0 mm |\n"));
        }

        let chunks = chunk_document("abc", &document(vec![&table]));
        let tables: Vec<_> = chunks.iter().filter(|c| c.kind == ChunkKind::Table).collect();

        assert_eq!(tables.len(), 1, "the table should be one chunk however long");
        assert!(tables[0].text.contains("| Point | Thickness | Limit |"));
        assert!(tables[0].text.contains("| P199 |"));
        assert!(tables[0].char_count as usize > MAX_CHARS, "oversized on purpose");
    }

    #[test]
    fn a_table_carries_the_section_it_sits_in() {
        let text = "4.3 Results\n\n| Point | Thickness |\n|---|---|\n| P1 | 8.2 mm |\n";
        let chunks = chunk_document("abc", &document(vec![text]));
        let table = chunks.iter().find(|c| c.kind == ChunkKind::Table).unwrap();
        assert_eq!(table.section_path, vec!["4.3 Results"]);
    }

    #[test]
    fn a_long_section_is_split_at_paragraph_boundaries() {
        let paragraph = "Ultrasonic measurement is taken at four points. ".repeat(20);
        let text = format!("4.2 Method\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}\n");

        let chunks = chunk_document("abc", &document(vec![&text]));
        assert!(chunks.len() > 1, "a long section should be split");
        // Every piece keeps the trail, which is the point.
        assert!(chunks.iter().all(|c| c.section_path == vec!["4.2 Method"]));
    }

    #[test]
    fn a_short_section_is_left_whole() {
        let text = "4.2 Limit\n\nMinimum 9.0 mm.\n";
        let chunks = chunk_document("abc", &document(vec![text]));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.contains("Minimum 9.0 mm."));
    }

    #[test]
    fn page_numbers_follow_the_content() {
        let chunks = chunk_document("abc", &document(vec!["4 First\n\nOn page one.", "5 Second\n\nOn page two."]));
        assert_eq!(chunks[0].page, 1);
        assert_eq!(chunks.last().unwrap().page, 2);
    }

    /// A page that could not be read contributes nothing rather than an empty
    /// chunk that would dilute retrieval.
    #[test]
    fn an_unread_page_produces_no_chunks() {
        let chunks = chunk_document("abc", &document(vec!["4 Real\n\nContent.", "   "]));
        assert!(chunks.iter().all(|c| c.page == 1));
    }

    #[test]
    fn chunk_ids_are_stable_across_identical_runs() {
        let doc = document(vec![SOP]);
        let first = chunk_document("abc", &doc);
        let second = chunk_document("abc", &doc);
        assert_eq!(
            first.iter().map(|c| c.id.clone()).collect::<Vec<_>>(),
            second.iter().map(|c| c.id.clone()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn chunks_from_different_documents_never_share_an_id() {
        let doc = document(vec![SOP]);
        let a = chunk_document("aaa", &doc);
        let b = chunk_document("bbb", &doc);
        assert_ne!(a[0].id, b[0].id);
    }

    #[test]
    fn a_document_with_no_headings_still_chunks_and_cites_the_page() {
        let chunks = chunk_document("abc", &document(vec!["Just some loose prose with no structure."]));
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].section_path.is_empty());
        assert_eq!(chunks[0].citation(), "page 1");
    }
}
