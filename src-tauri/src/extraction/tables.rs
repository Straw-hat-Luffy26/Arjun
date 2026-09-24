//! Tables the OCR model transcribed, turned into cells.
//!
//! Unlimited-OCR writes a table region as HTML on one line:
//! `table [78, 304, 785, 460]<table><tr><td>Phase</td>…</tr></table>`. The
//! cells are the part a field lookup needs, so they are parsed here — by a
//! tokenizer over exactly the tags a table uses, not by an HTML engine that
//! would also accept everything else.
//!
//! ## What it will not do
//!
//! Invent structure. A `<table>` whose body carries no row or cell tags (a real
//! capture in `ocr_spans` has `<table>PhaseReadingLimit</table>`) has no cell
//! boundaries to report, and [`parse_ocr_table`] answers `None` for it; the
//! caller keeps the region's text and says the cells were not delimited. Nor
//! does it give a cell a box: OCR locates the table, not its cells.

use super::regions::TableCell;

/// Rows beyond this are dropped and the table is marked cut. A transcription
/// that loops inside a table produces thousands of identical rows.
pub const MAX_TABLE_ROWS: usize = 500;
pub const MAX_TABLE_COLS: usize = 64;

/// The cells, and anything the parse noticed.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedTable {
    pub cells: Vec<TableCell>,
    pub rows: u32,
    pub cols: u32,
    pub notes: Vec<String>,
}

/// Parses the `<table>…</table>` in `text`, if it has delimited cells.
pub fn parse_ocr_table(text: &str) -> Option<ParsedTable> {
    let lower = text.to_ascii_lowercase();
    let start = lower.find("<table")?;
    let end = lower[start..].find("</table>").map(|at| start + at);
    let body = &text[start..end.unwrap_or(text.len())];
    let body_lower = &lower[start..end.unwrap_or(text.len())];
    if !body_lower.contains("<tr") {
        return None;
    }

    let mut notes = Vec::new();
    if end.is_none() {
        notes.push("the table was not closed; the transcription stops inside it".to_string());
    }

    let mut cells = Vec::new();
    let mut row: u32 = 0;
    let mut cols: u32 = 0;
    let mut rows_seen = 0usize;
    let mut spans_ignored = false;

    let mut cursor = 0usize;
    while let Some(tr) = body_lower[cursor..].find("<tr") {
        let row_start = cursor + tr;
        let row_end = body_lower[row_start + 3..]
            .find("<tr")
            .map(|at| row_start + 3 + at)
            .unwrap_or(body.len());
        let row_html = &body[row_start..row_end];
        let row_lower = &body_lower[row_start..row_end];
        rows_seen += 1;
        if rows_seen > MAX_TABLE_ROWS {
            notes.push(format!(
                "only the first {MAX_TABLE_ROWS} rows were kept; the transcription holds more"
            ));
            break;
        }

        let mut col: u32 = 0;
        let mut at = 0usize;
        let mut any = false;
        loop {
            let td = row_lower[at..].find("<td");
            let th = row_lower[at..].find("<th");
            let open = match (td, th) {
                (Some(a), Some(b)) => a.min(b),
                (Some(a), None) | (None, Some(a)) => a,
                (None, None) => break,
            } + at;
            let Some(tag_end) = row_lower[open..].find('>').map(|x| open + x) else {
                break;
            };
            let attributes = &row_lower[open..tag_end];
            let close = ["</td>", "</th>", "<td", "<th"]
                .iter()
                .filter_map(|marker| row_lower[tag_end + 1..].find(marker))
                .min()
                .map(|x| tag_end + 1 + x)
                .unwrap_or(row_html.len());
            let raw = &row_html[tag_end + 1..close];
            if col as usize >= MAX_TABLE_COLS {
                notes.push(format!("columns beyond {MAX_TABLE_COLS} were dropped"));
                break;
            }
            cells.push(TableCell {
                row,
                col,
                text: cell_text(raw),
                bbox: None,
            });
            any = true;
            let span = span_of(attributes, "colspan").unwrap_or(1).clamp(1, 16);
            if attributes.contains("rowspan") {
                spans_ignored = true;
            }
            col += span;
            at = close;
        }
        if any {
            cols = cols.max(col);
            row += 1;
        }
        cursor = row_end;
    }

    if spans_ignored {
        notes.push(
            "a cell spans rows; the cells below it are numbered as written, not shifted".to_string(),
        );
    }
    if cells.is_empty() {
        return None;
    }
    Some(ParsedTable {
        cells,
        rows: row,
        cols,
        notes,
    })
}

fn span_of(attributes: &str, name: &str) -> Option<u32> {
    let at = attributes.find(name)?;
    let rest = &attributes[at + name.len()..];
    let digits: String = rest
        .chars()
        .skip_while(|c| *c == '=' || *c == '"' || *c == '\'' || c.is_whitespace())
        .take_while(|c| c.is_ascii_digit())
        .collect();
    digits.parse().ok()
}

/// Tag-stripped, entity-decoded, whitespace-collapsed.
fn cell_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut in_tag = false;
    for c in raw.chars() {
        match c {
            '<' => in_tag = true,
            '>' if in_tag => {
                in_tag = false;
                out.push(' ');
            }
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    let decoded = out
        .replace("&nbsp;", " ")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", "\"")
        .replace("&#39;", "'")
        .replace("&amp;", "&");
    decoded.split_whitespace().collect::<Vec<_>>().join(" ")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_delimited_table_becomes_cells_in_row_order() {
        let parsed = parse_ocr_table(
            "<table><tr><th>Point</th><th>mm</th></tr><tr><td>A</td><td>9.4</td></tr>\
             <tr><td>C &amp; D</td><td>8.2</td></tr></table>",
        )
        .unwrap();
        assert_eq!(parsed.rows, 3);
        assert_eq!(parsed.cols, 2);
        let at = |r: u32, c: u32| {
            parsed
                .cells
                .iter()
                .find(|cell| cell.row == r && cell.col == c)
                .map(|cell| cell.text.clone())
        };
        assert_eq!(at(0, 0).as_deref(), Some("Point"));
        assert_eq!(at(1, 1).as_deref(), Some("9.4"));
        assert_eq!(at(2, 0).as_deref(), Some("C & D"));
        assert!(parsed.cells.iter().all(|cell| cell.bbox.is_none()));
    }

    #[test]
    fn a_table_with_no_cell_boundaries_is_not_given_any() {
        // Verbatim from the real capture in `ocr_spans`.
        assert_eq!(parse_ocr_table("<table>PhaseReadingLimit</table>"), None);
    }

    #[test]
    fn a_colspan_moves_the_next_cell_and_an_unclosed_table_says_so() {
        let parsed =
            parse_ocr_table("<table><tr><td colspan=\"2\">Head</td><td>X</td></tr><tr><td>1</td>")
                .unwrap();
        assert!(parsed.cells.iter().any(|c| c.row == 0 && c.col == 2 && c.text == "X"));
        assert!(parsed.notes.iter().any(|n| n.contains("not closed")));
    }

    #[test]
    fn a_looping_table_is_cut_at_the_row_limit() {
        let mut html = String::from("<table>");
        for _ in 0..(MAX_TABLE_ROWS + 20) {
            html.push_str("<tr><td>same</td></tr>");
        }
        html.push_str("</table>");
        let parsed = parse_ocr_table(&html).unwrap();
        assert_eq!(parsed.rows as usize, MAX_TABLE_ROWS);
        assert!(parsed.notes.iter().any(|n| n.contains("first")));
    }
}
