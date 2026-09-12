//! The typed shape of a deliverable, before any format has been chosen.
//!
//! ## Why a model rather than a string
//!
//! Every generator here used to take either a map of template fields or a body
//! string, and write it. There was no intermediate representation, which had
//! two consequences that ran through the whole subsystem:
//!
//! - Nothing could be checked **before** rendering. A document with a table
//!   whose rows had different widths, or a heading hierarchy that jumped from
//!   1 to 3, was discovered — if at all — by re-opening the file afterwards.
//! - Nothing could be **repaired** without re-rendering. "Repair" and
//!   "regenerate" were the same operation, so the distinction the correction
//!   loop rests on could not be expressed.
//!
//! A model fixes both. It is validated in memory, repaired in memory, and only
//! then handed to a format-specific writer.
//!
//! ## Why one model and not four
//!
//! A document, a report and a printed PDF are the same thing with different
//! output. Keeping one [`Document`] that several writers render means a heading
//! hierarchy is checked once rather than three times, and means asking for the
//! same content as `.docx` and `.pdf` cannot produce two different documents.
//!
//! A workbook and a deck are genuinely different shapes — cells with types and
//! formulas, slides with a density limit — so they are their own models
//! ([`Workbook`], [`Deck`]) rather than being forced through `Document`.
//!
//! ## What validation is, and is not
//!
//! `problems()` answers *"is this well formed and complete enough to render"*.
//! It is structural. Whether the prose is any good is not a question this can
//! answer; the parts that geometry or arithmetic **can** answer — a slide that
//! will overflow, a formula in a column of numbers — are here, because this is
//! where the data to answer them lives.

use serde::{Deserialize, Serialize};

/// Text that must never reach a reader inside a finished deliverable.
///
/// `tbd` is here and `n/a` is not: "n/a" is a legitimate answer to a field, and
/// "to be decided" is an admission that the work is not done.
pub const PLACEHOLDERS: &[&str] = &[
    "lorem ipsum",
    "tbd",
    "to be decided",
    "to be determined",
    "todo",
    "placeholder",
    "[insert",
    "<insert",
];

/// Whether `text` contains something that should never be handed over.
pub fn placeholder_in(text: &str) -> Option<&'static str> {
    let lowered = text.to_lowercase();
    PLACEHOLDERS.iter().find(|marker| lowered.contains(*marker)).copied()
}

// ── Documents ────────────────────────────────────────────────────────────

/// One piece of a document's body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "kind")]
pub enum Block {
    /// A run of prose.
    Paragraph { text: String },
    /// An unordered list. Kept as a block rather than as paragraphs so a writer
    /// can render real list formatting and a checker can see it is a list.
    Bullets { items: Vec<String> },
    /// A numbered list, for a procedure where the order is the meaning.
    Numbered { items: Vec<String> },
    /// A table. The header is separate from the rows because every format
    /// renders it differently and every checker has to know which is which.
    Table {
        header: Vec<String>,
        rows: Vec<Vec<String>>,
        /// What the table is. Rendered as a caption where the format supports
        /// one, and used by the checker to say which table is wrong.
        caption: Option<String>,
    },
    /// An explicit break. Meaningful in paged formats, ignored elsewhere.
    PageBreak,
}

impl Block {
    /// Characters of real content, for the emptiness checks.
    pub fn characters(&self) -> usize {
        match self {
            Block::Paragraph { text } => text.trim().chars().count(),
            Block::Bullets { items } | Block::Numbered { items } => {
                items.iter().map(|i| i.trim().chars().count()).sum()
            }
            Block::Table { header, rows, caption } => {
                header.iter().map(|h| h.trim().chars().count()).sum::<usize>()
                    + rows
                        .iter()
                        .flat_map(|row| row.iter())
                        .map(|c| c.trim().chars().count())
                        .sum::<usize>()
                    + caption.as_deref().map(|c| c.trim().chars().count()).unwrap_or(0)
            }
            Block::PageBreak => 0,
        }
    }

    /// All the text in the block, for the placeholder scan.
    pub fn text(&self) -> String {
        match self {
            Block::Paragraph { text } => text.clone(),
            Block::Bullets { items } | Block::Numbered { items } => items.join(" "),
            Block::Table { header, rows, caption } => {
                let mut out = caption.clone().unwrap_or_default();
                out.push(' ');
                out.push_str(&header.join(" "));
                for row in rows {
                    out.push(' ');
                    out.push_str(&row.join(" "));
                }
                out
            }
            Block::PageBreak => String::new(),
        }
    }

    fn problems(&self, where_: &str) -> Vec<String> {
        let mut problems = Vec::new();
        match self {
            Block::Paragraph { text } => {
                if text.trim().is_empty() {
                    problems.push(format!("{where_} contains an empty paragraph"));
                }
            }
            Block::Bullets { items } | Block::Numbered { items } => {
                if items.is_empty() {
                    problems.push(format!("{where_} contains a list with no items"));
                }
                if items.iter().any(|item| item.trim().is_empty()) {
                    problems.push(format!("{where_} contains a list item with no text"));
                }
            }
            Block::Table { header, rows, .. } => {
                if header.is_empty() {
                    problems.push(format!("{where_} contains a table with no header row"));
                }
                if rows.is_empty() {
                    problems.push(format!("{where_} contains a table with no rows"));
                }
                // A ragged table renders as a document somebody has to fix by
                // hand. Named per row so they can.
                for (index, row) in rows.iter().enumerate() {
                    if row.len() != header.len() {
                        problems.push(format!(
                            "{where_}: table row {} has {} cell(s) against {} column(s)",
                            index + 1,
                            row.len(),
                            header.len()
                        ));
                    }
                }
            }
            Block::PageBreak => {}
        }
        problems
    }
}

/// A titled part of a document.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Section {
    pub heading: String,
    /// 1 for a top-level section. Checked for skips: a document that goes from
    /// 1 to 3 has a broken outline, and every reader that builds a table of
    /// contents from it produces a broken one.
    pub level: u8,
    pub blocks: Vec<Block>,
}

/// A document, independent of whether it becomes `.docx` or `.pdf`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Document {
    pub title: String,
    /// Stamped on every page. Never empty in this product: an unclassified
    /// deliverable is one nobody can file.
    pub classification: String,
    pub sections: Vec<Section>,
    /// What a reader's Properties panel shows and a document system indexes on.
    #[serde(default)]
    pub properties: Properties,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Properties {
    pub author: Option<String>,
    pub subject: Option<String>,
    #[serde(default)]
    pub keywords: Vec<String>,
}

/// The least text a section must carry to count as written.
///
/// A heading with nothing under it is the shape of a document somebody
/// abandoned. Low enough that a genuine one-line section passes.
const MIN_SECTION_CHARACTERS: usize = 20;

impl Document {
    /// Everything wrong with it, in the words the model needs to fix it.
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.title.trim().is_empty() {
            problems.push("the document has no title".to_string());
        }
        if self.classification.trim().is_empty() {
            problems.push("the document has no classification".to_string());
        }
        if self.sections.is_empty() {
            problems.push("the document has no sections".to_string());
            return problems;
        }

        let mut previous: u8 = 0;
        for section in &self.sections {
            let name = if section.heading.trim().is_empty() {
                problems.push("a section has no heading".to_string());
                "an unnamed section".to_string()
            } else {
                format!("the {:?} section", section.heading.trim())
            };

            if section.level == 0 {
                problems.push(format!("{name} has a heading level of 0; levels start at 1"));
            } else if previous > 0 && section.level > previous + 1 {
                // A jump means every generated table of contents, navigation
                // pane and screen reader reports a structure the document does
                // not have.
                problems.push(format!(
                    "{name} is at heading level {} under a level {previous} section, which skips \
                     level {}",
                    section.level,
                    previous + 1
                ));
            }
            if section.level > 0 {
                previous = section.level;
            }

            if section.blocks.is_empty() {
                problems.push(format!("{name} has a heading and nothing under it"));
            }
            for block in &section.blocks {
                problems.extend(block.problems(&name));
            }

            let characters: usize = section.blocks.iter().map(Block::characters).sum();
            if !section.blocks.is_empty() && characters < MIN_SECTION_CHARACTERS {
                problems.push(format!(
                    "{name} holds {characters} characters, too little to be a written section"
                ));
            }

            for block in &section.blocks {
                if let Some(marker) = placeholder_in(&block.text()) {
                    problems.push(format!("{name} still contains the placeholder {marker:?}"));
                }
            }
        }

        problems
    }

    pub fn is_sound(&self) -> bool {
        self.problems().is_empty()
    }

    /// Repairs that need no new information.
    ///
    /// Returns what it did, so the caller can say so rather than silently
    /// changing the deliverable. What it will not do is invent content: a
    /// missing section stays missing and a row longer than its header is
    /// reported rather than truncated, because dropping a cell loses data.
    pub fn repair(&mut self) -> Vec<String> {
        let mut done = Vec::new();

        for section in &mut self.sections {
            // An empty paragraph is a rendering artefact of a model writing
            // "\n\n", not a decision.
            let before = section.blocks.len();
            section.blocks.retain(|block| match block {
                Block::Paragraph { text } => !text.trim().is_empty(),
                Block::Bullets { items } | Block::Numbered { items } => {
                    items.iter().any(|i| !i.trim().is_empty())
                }
                _ => true,
            });
            if section.blocks.len() != before {
                done.push(format!(
                    "removed {} empty block(s) from {:?}",
                    before - section.blocks.len(),
                    section.heading
                ));
            }

            for block in &mut section.blocks {
                match block {
                    Block::Bullets { items } | Block::Numbered { items } => {
                        let before = items.len();
                        items.retain(|item| !item.trim().is_empty());
                        if items.len() != before {
                            done.push(format!(
                                "removed {} empty list item(s) from {:?}",
                                before - items.len(),
                                section.heading
                            ));
                        }
                    }
                    Block::Table { header, rows, .. } => {
                        for row in rows.iter_mut() {
                            if row.len() < header.len() {
                                let short = header.len() - row.len();
                                row.resize(header.len(), String::new());
                                done.push(format!(
                                    "padded a short table row with {short} empty cell(s) in {:?}",
                                    section.heading
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        // A level that skips is pulled up to one below its parent. The heading
        // text is unchanged.
        let mut previous: u8 = 0;
        for section in &mut self.sections {
            if section.level == 0 {
                section.level = 1;
                done.push(format!("set {:?} to heading level 1", section.heading));
            } else if previous > 0 && section.level > previous + 1 {
                let was = section.level;
                section.level = previous + 1;
                done.push(format!(
                    "moved {:?} from heading level {was} to {} so the outline does not skip",
                    section.heading, section.level
                ));
            }
            previous = section.level;
        }

        done
    }

    /// Every word in the document, for the text-presence checks a format's own
    /// validator makes after rendering.
    pub fn text(&self) -> String {
        let mut out = self.title.clone();
        for section in &self.sections {
            out.push(' ');
            out.push_str(&section.heading);
            for block in &section.blocks {
                out.push(' ');
                out.push_str(&block.text());
            }
        }
        out
    }
}

// ── Workbooks ────────────────────────────────────────────────────────────

/// What a column holds. Decides the cell type written and the number format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum ColumnType {
    Text,
    Number,
    /// A number rendered with a currency format.
    Currency,
    /// A fraction rendered as a percentage.
    Percent,
    /// ISO-8601 `YYYY-MM-DD`, which is the only date format accepted: anything
    /// else is ambiguous across locales, and a spreadsheet that reads `03/04`
    /// differently from its author is worse than one that refuses.
    Date,
    /// A live formula, e.g. `=SUM(B2:B9)`.
    Formula,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Column {
    pub header: String,
    #[serde(rename = "type")]
    pub kind: ColumnType,
    /// Width in characters. `None` lets the writer size it from the content.
    pub width: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Sheet {
    pub name: String,
    pub columns: Vec<Column>,
    /// Row-major, each inner vector one row of cells as written text. A
    /// `Formula` column's cell is the formula itself.
    pub rows: Vec<Vec<String>>,
    /// Whether to freeze the header row so it stays visible while scrolling.
    #[serde(default)]
    pub freeze_header: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Workbook {
    pub title: String,
    pub classification: String,
    pub sheets: Vec<Sheet>,
}

/// Characters Excel refuses in a sheet name, plus the length limit.
const ILLEGAL_SHEET_CHARS: &[char] = &['[', ']', ':', '*', '?', '/', '\\'];
const MAX_SHEET_NAME: usize = 31;

impl Workbook {
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.title.trim().is_empty() {
            problems.push("the workbook has no title".to_string());
        }
        if self.sheets.is_empty() {
            problems.push("the workbook has no sheets".to_string());
            return problems;
        }

        let mut seen: Vec<String> = Vec::new();
        for sheet in &self.sheets {
            let name = sheet.name.trim();
            if name.is_empty() {
                problems.push("a sheet has no name".to_string());
                continue;
            }
            // Excel's own rules. A workbook that breaks them does not open, and
            // the message Excel gives is not one a person can act on.
            if name.chars().count() > MAX_SHEET_NAME {
                problems.push(format!(
                    "the sheet name {name:?} is {} characters; Excel allows {MAX_SHEET_NAME}",
                    name.chars().count()
                ));
            }
            if let Some(bad) = name.chars().find(|c| ILLEGAL_SHEET_CHARS.contains(c)) {
                problems.push(format!(
                    "the sheet name {name:?} contains {bad:?}, which Excel forbids"
                ));
            }
            let lowered = name.to_lowercase();
            if seen.contains(&lowered) {
                problems.push(format!("there is more than one sheet named {name:?}"));
            }
            seen.push(lowered);

            if sheet.columns.is_empty() {
                problems.push(format!("the {name:?} sheet has no columns"));
                continue;
            }
            if sheet.columns.iter().any(|c| c.header.trim().is_empty()) {
                problems.push(format!("the {name:?} sheet has a column with no header"));
            }
            if sheet.rows.is_empty() {
                problems.push(format!("the {name:?} sheet has no rows"));
            }

            for (index, row) in sheet.rows.iter().enumerate() {
                if row.len() != sheet.columns.len() {
                    problems.push(format!(
                        "the {name:?} sheet: row {} has {} cell(s) against {} column(s)",
                        index + 1,
                        row.len(),
                        sheet.columns.len()
                    ));
                    continue;
                }
                for (column, cell) in sheet.columns.iter().zip(row) {
                    if let Some(problem) = cell_problem(column, cell, name, index + 1) {
                        problems.push(problem);
                    }
                }
            }
        }

        problems
    }

    pub fn is_sound(&self) -> bool {
        self.problems().is_empty()
    }

    /// Deterministic repairs: sheet names and short rows.
    pub fn repair(&mut self) -> Vec<String> {
        let mut done = Vec::new();
        let mut seen: Vec<String> = Vec::new();

        for sheet in &mut self.sheets {
            let original = sheet.name.clone();
            let mut name: String = sheet
                .name
                .trim()
                .chars()
                .filter(|c| !ILLEGAL_SHEET_CHARS.contains(c))
                .collect();
            if name.chars().count() > MAX_SHEET_NAME {
                name = name.chars().take(MAX_SHEET_NAME).collect();
            }
            if name.trim().is_empty() {
                name = format!("Sheet{}", seen.len() + 1);
            }
            // Uniqueness by suffixing rather than by dropping a sheet.
            let mut candidate = name.clone();
            let mut n = 2;
            while seen.contains(&candidate.to_lowercase()) {
                let suffix = format!(" ({n})");
                let room = MAX_SHEET_NAME.saturating_sub(suffix.len());
                let stem: String = name.chars().take(room).collect();
                candidate = format!("{stem}{suffix}");
                n += 1;
            }
            seen.push(candidate.to_lowercase());
            if candidate != original {
                done.push(format!("renamed the sheet {original:?} to {candidate:?}"));
                sheet.name = candidate;
            }

            for row in sheet.rows.iter_mut() {
                if row.len() < sheet.columns.len() {
                    row.resize(sheet.columns.len(), String::new());
                    done.push(format!("padded a short row in {:?}", sheet.name));
                }
            }
        }
        done
    }
}

/// Whether one cell matches the type its column declares.
fn cell_problem(column: &Column, cell: &str, sheet: &str, row: usize) -> Option<String> {
    let value = cell.trim();
    if value.is_empty() {
        // An empty cell is legitimate — not every reading was taken.
        return None;
    }
    match column.kind {
        ColumnType::Text => None,
        ColumnType::Number | ColumnType::Currency => {
            value.replace(',', "").parse::<f64>().err().map(|_| {
                format!(
                    "the {sheet:?} sheet, row {row}: {:?} is declared a number and holds {value:?}",
                    column.header
                )
            })
        }
        ColumnType::Percent => value.trim_end_matches('%').parse::<f64>().err().map(|_| {
            format!(
                "the {sheet:?} sheet, row {row}: {:?} is declared a percentage and holds {value:?}",
                column.header
            )
        }),
        ColumnType::Date => {
            let parts: Vec<&str> = value.split('-').collect();
            let ok = parts.len() == 3
                && parts[0].len() == 4
                && parts[1].len() == 2
                && parts[2].len() == 2
                && parts.iter().all(|p| p.chars().all(|c| c.is_ascii_digit()));
            (!ok).then(|| {
                format!(
                    "the {sheet:?} sheet, row {row}: {:?} is declared a date and holds {value:?}, \
                     which is not YYYY-MM-DD",
                    column.header
                )
            })
        }
        ColumnType::Formula => (!value.starts_with('=')).then(|| {
            format!(
                "the {sheet:?} sheet, row {row}: {:?} is declared a formula and holds {value:?}, \
                 which does not begin with =",
                column.header
            )
        }),
    }
}

// ── Decks ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SlideModel {
    pub heading: String,
    #[serde(default)]
    pub bullets: Vec<String>,
    /// Optional table. A slide carrying one holds fewer bullets.
    #[serde(default)]
    pub table: Option<Block>,
    /// Speaker notes, which is where the detail that will not fit belongs.
    #[serde(default)]
    pub notes: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Deck {
    pub title: String,
    pub classification: String,
    pub slides: Vec<SlideModel>,
}

/// What fits on a slide before it stops being readable.
///
/// Not style preferences. These are the point at which text runs off the bottom
/// of the layout this product renders, which the reader discovers in the
/// meeting. Derived from the box geometry in `pptx.rs`: a 16:9 content
/// placeholder at the body size holds about seven lines of roughly ninety
/// characters.
pub const MAX_BULLETS_PER_SLIDE: usize = 7;
pub const MAX_BULLET_CHARACTERS: usize = 180;
pub const MAX_HEADING_CHARACTERS: usize = 90;

impl Deck {
    pub fn problems(&self) -> Vec<String> {
        let mut problems = Vec::new();

        if self.title.trim().is_empty() {
            problems.push("the deck has no title".to_string());
        }
        if self.slides.is_empty() {
            problems.push("the deck has no slides".to_string());
            return problems;
        }

        for (index, slide) in self.slides.iter().enumerate() {
            let name = if slide.heading.trim().is_empty() {
                problems.push(format!("slide {} has no heading", index + 1));
                format!("slide {}", index + 1)
            } else {
                format!("the {:?} slide", slide.heading.trim())
            };

            if slide.heading.chars().count() > MAX_HEADING_CHARACTERS {
                problems.push(format!(
                    "{name} has a {}-character heading; more than {MAX_HEADING_CHARACTERS} \
                     overflows the title box",
                    slide.heading.chars().count()
                ));
            }
            if slide.bullets.is_empty() && slide.table.is_none() {
                problems.push(format!("{name} has a title and nothing else on it"));
            }
            if slide.bullets.len() > MAX_BULLETS_PER_SLIDE {
                problems.push(format!(
                    "{name} carries {} bullets; more than {MAX_BULLETS_PER_SLIDE} runs off the \
                     bottom of the slide",
                    slide.bullets.len()
                ));
            }
            for bullet in &slide.bullets {
                if bullet.trim().is_empty() {
                    problems.push(format!("{name} has an empty bullet"));
                } else if bullet.chars().count() > MAX_BULLET_CHARACTERS {
                    problems.push(format!(
                        "{name} has a {}-character bullet; more than {MAX_BULLET_CHARACTERS} \
                         wraps past the end of the box",
                        bullet.chars().count()
                    ));
                }
                if let Some(marker) = placeholder_in(bullet) {
                    problems.push(format!("{name} still contains the placeholder {marker:?}"));
                }
            }
            if let Some(table) = &slide.table {
                problems.extend(table.problems(&name));
            }
        }

        problems
    }

    pub fn is_sound(&self) -> bool {
        self.problems().is_empty()
    }

    /// Splits overflowing slides and drops empty bullets.
    ///
    /// Splitting is the repair that matters: a slide with twelve bullets is not
    /// broken content, it is content that needs two slides, and a person would
    /// fix it the same way. The continuation is named so a reader can see what
    /// happened.
    pub fn repair(&mut self) -> Vec<String> {
        let mut done = Vec::new();

        for slide in &mut self.slides {
            let before = slide.bullets.len();
            slide.bullets.retain(|b| !b.trim().is_empty());
            if slide.bullets.len() != before {
                done.push(format!(
                    "removed {} empty bullet(s) from {:?}",
                    before - slide.bullets.len(),
                    slide.heading
                ));
            }
        }

        let mut split: Vec<SlideModel> = Vec::new();
        for slide in self.slides.drain(..) {
            if slide.bullets.len() <= MAX_BULLETS_PER_SLIDE {
                split.push(slide);
                continue;
            }
            let carried = slide.bullets.len();
            let chunks: Vec<Vec<String>> = slide
                .bullets
                .chunks(MAX_BULLETS_PER_SLIDE)
                .map(<[String]>::to_vec)
                .collect();
            let total = chunks.len();
            done.push(format!(
                "split {:?} across {total} slides; it carried {carried} bullets",
                slide.heading
            ));
            for (index, chunk) in chunks.into_iter().enumerate() {
                split.push(SlideModel {
                    heading: if index == 0 {
                        slide.heading.clone()
                    } else {
                        format!("{} ({} of {total})", slide.heading, index + 1)
                    },
                    bullets: chunk,
                    // The table and the notes belong to the first slide;
                    // duplicating them would show the same table twice.
                    table: if index == 0 { slide.table.clone() } else { None },
                    notes: if index == 0 { slide.notes.clone() } else { None },
                });
            }
        }
        self.slides = split;

        done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn paragraph(text: &str) -> Block {
        Block::Paragraph { text: text.to_string() }
    }

    fn section(heading: &str, level: u8, blocks: Vec<Block>) -> Section {
        Section { heading: heading.to_string(), level, blocks }
    }

    fn document(sections: Vec<Section>) -> Document {
        Document {
            title: "Unit Four Outage".to_string(),
            classification: "OFFICIAL".to_string(),
            sections,
            properties: Properties::default(),
        }
    }

    #[test]
    fn a_well_formed_document_has_no_problems() {
        let doc = document(vec![section(
            "Findings",
            1,
            vec![paragraph("The wear ring clearance was measured at 0.45 mm.")],
        )]);
        assert!(doc.is_sound(), "{:?}", doc.problems());
    }

    /// A jump means every generated table of contents is wrong, and no reader
    /// will see why.
    #[test]
    fn a_heading_level_that_skips_is_a_problem() {
        let doc = document(vec![
            section("Findings", 1, vec![paragraph("The clearance was measured at 0.45 mm.")]),
            section("Detail", 3, vec![paragraph("Measured with a feeler gauge on the day.")]),
        ]);
        assert!(doc.problems().iter().any(|p| p.contains("skips level 2")), "{:?}", doc.problems());
    }

    #[test]
    fn a_ragged_table_is_named_row_by_row() {
        let doc = document(vec![section(
            "Readings",
            1,
            vec![Block::Table {
                header: vec!["Tag".into(), "Reading".into(), "Limit".into()],
                rows: vec![vec!["PV-2201".into(), "0.45".into()]],
                caption: Some("Clearances measured during the outage".into()),
            }],
        )]);
        assert!(
            doc.problems().iter().any(|p| p.contains("row 1 has 2 cell(s) against 3")),
            "{:?}",
            doc.problems()
        );
    }

    #[test]
    fn a_heading_with_nothing_under_it_is_a_problem() {
        let doc = document(vec![section("Findings", 1, vec![])]);
        assert!(doc.problems().iter().any(|p| p.contains("nothing under it")));
    }

    #[test]
    fn a_placeholder_is_a_problem() {
        let doc = document(vec![section(
            "Recommendation",
            1,
            vec![paragraph("TBD — awaiting the vendor's confirmation of the price.")],
        )]);
        assert!(doc.problems().iter().any(|p| p.contains("placeholder")), "{:?}", doc.problems());
    }

    #[test]
    fn repair_fixes_an_outline_without_touching_the_words() {
        let mut doc = document(vec![
            section("Findings", 1, vec![paragraph("The clearance was measured at 0.45 mm.")]),
            section("Detail", 4, vec![paragraph("Measured with a feeler gauge on the day.")]),
        ]);
        let done = doc.repair();
        assert!(done.iter().any(|d| d.contains("does not skip")), "{done:?}");
        assert_eq!(doc.sections[1].level, 2);
        assert_eq!(doc.sections[1].heading, "Detail", "the heading text must not change");
        assert!(doc.is_sound(), "{:?}", doc.problems());
    }

    #[test]
    fn repair_pads_a_short_row_but_never_drops_a_cell() {
        let mut doc = document(vec![section(
            "Readings",
            1,
            vec![Block::Table {
                header: vec!["Tag".into(), "Reading".into()],
                rows: vec![vec!["PV-2201".into()], vec!["A".into(), "B".into(), "C".into()]],
                caption: None,
            }],
        )]);
        doc.repair();
        let problems = doc.problems();
        // The short row was padded; the long row was not truncated, because
        // dropping a cell loses data the model has to be told about.
        assert!(
            problems.iter().any(|p| p.contains("row 2 has 3 cell(s)")),
            "a long row must still be reported: {problems:?}"
        );
        assert!(!problems.iter().any(|p| p.contains("row 1 has")), "{problems:?}");
    }

    // ── Workbooks ───────────────────────────────────────────────────────

    fn workbook(sheets: Vec<Sheet>) -> Workbook {
        Workbook {
            title: "Clearances".to_string(),
            classification: "OFFICIAL".to_string(),
            sheets,
        }
    }

    fn column(header: &str, kind: ColumnType) -> Column {
        Column { header: header.to_string(), kind, width: None }
    }

    #[test]
    fn a_well_formed_workbook_has_no_problems() {
        let book = workbook(vec![Sheet {
            name: "Readings".to_string(),
            columns: vec![column("Tag", ColumnType::Text), column("Clearance", ColumnType::Number)],
            rows: vec![vec!["PV-2201".into(), "0.45".into()]],
            freeze_header: true,
        }]);
        assert!(book.is_sound(), "{:?}", book.problems());
    }

    #[test]
    fn a_number_column_holding_prose_is_a_problem() {
        let book = workbook(vec![Sheet {
            name: "Readings".to_string(),
            columns: vec![column("Clearance", ColumnType::Number)],
            rows: vec![vec!["about half a millimetre".into()]],
            freeze_header: false,
        }]);
        assert!(
            book.problems().iter().any(|p| p.contains("declared a number")),
            "{:?}",
            book.problems()
        );
    }

    #[test]
    fn a_formula_that_is_not_a_formula_is_a_problem() {
        let book = workbook(vec![Sheet {
            name: "Totals".to_string(),
            columns: vec![column("Total", ColumnType::Formula)],
            rows: vec![vec!["42".into()]],
            freeze_header: false,
        }]);
        assert!(book.problems().iter().any(|p| p.contains("does not begin with")));
    }

    /// An ambiguous date is worse than a refused one.
    #[test]
    fn a_date_that_is_not_iso_is_a_problem() {
        let book = workbook(vec![Sheet {
            name: "Log".to_string(),
            columns: vec![column("Inspected", ColumnType::Date)],
            rows: vec![vec!["03/04/2026".into()]],
            freeze_header: false,
        }]);
        assert!(book.problems().iter().any(|p| p.contains("YYYY-MM-DD")));
    }

    #[test]
    fn excels_own_sheet_name_rules_are_enforced() {
        let book = workbook(vec![Sheet {
            name: "Q1/Q2 readings".to_string(),
            columns: vec![column("Tag", ColumnType::Text)],
            rows: vec![vec!["PV-2201".into()]],
            freeze_header: false,
        }]);
        assert!(book.problems().iter().any(|p| p.contains("forbids")), "{:?}", book.problems());
    }

    #[test]
    fn repair_makes_a_sheet_name_legal_and_unique() {
        let mut book = workbook(vec![
            Sheet {
                name: "Q1/Q2".to_string(),
                columns: vec![column("Tag", ColumnType::Text)],
                rows: vec![vec!["A".into()]],
                freeze_header: false,
            },
            Sheet {
                name: "Q1Q2".to_string(),
                columns: vec![column("Tag", ColumnType::Text)],
                rows: vec![vec!["B".into()]],
                freeze_header: false,
            },
        ]);
        book.repair();
        assert_eq!(book.sheets[0].name, "Q1Q2");
        assert_eq!(book.sheets[1].name, "Q1Q2 (2)");
        assert!(book.is_sound(), "{:?}", book.problems());
    }

    // ── Decks ───────────────────────────────────────────────────────────

    fn deck(slides: Vec<SlideModel>) -> Deck {
        Deck {
            title: "Outage briefing".to_string(),
            classification: "OFFICIAL".to_string(),
            slides,
        }
    }

    fn slide(heading: &str, bullets: &[&str]) -> SlideModel {
        SlideModel {
            heading: heading.to_string(),
            bullets: bullets.iter().map(|b| b.to_string()).collect(),
            table: None,
            notes: None,
        }
    }

    #[test]
    fn a_well_formed_deck_has_no_problems() {
        let deck = deck(vec![slide("Findings", &["The clearance was within tolerance."])]);
        assert!(deck.is_sound(), "{:?}", deck.problems());
    }

    /// The failure a reader finds in the meeting.
    #[test]
    fn a_slide_that_will_overflow_is_a_problem() {
        let bullets: Vec<&str> = vec!["a point worth making"; MAX_BULLETS_PER_SLIDE + 3];
        let deck = deck(vec![slide("Findings", &bullets)]);
        assert!(
            deck.problems().iter().any(|p| p.contains("runs off the bottom")),
            "{:?}",
            deck.problems()
        );
    }

    #[test]
    fn repair_splits_an_overflowing_slide_rather_than_dropping_anything() {
        let bullets: Vec<String> = (1..=10).map(|n| format!("Point number {n}")).collect();
        let borrowed: Vec<&str> = bullets.iter().map(String::as_str).collect();
        let mut deck = deck(vec![slide("Findings", &borrowed)]);
        let done = deck.repair();

        assert!(done.iter().any(|d| d.contains("split")), "{done:?}");
        assert_eq!(deck.slides.len(), 2);
        let carried: usize = deck.slides.iter().map(|s| s.bullets.len()).sum();
        assert_eq!(carried, 10, "nothing may be lost in a split");
        assert!(deck.slides[1].heading.contains("2 of 2"), "{}", deck.slides[1].heading);
        assert!(deck.is_sound(), "{:?}", deck.problems());
    }

    #[test]
    fn a_slide_with_only_a_title_is_a_problem() {
        let deck = deck(vec![slide("Findings", &[])]);
        assert!(deck.problems().iter().any(|p| p.contains("nothing else on it")));
    }
}
