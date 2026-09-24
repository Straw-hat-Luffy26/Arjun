//! Requested fields, found in transcribed regions — or honestly not found.
//!
//! A field value reported here is a **checked observation**: its text occurs,
//! character for character, in a region that was transcribed (text layer,
//! embedded table or OCR), and the value carries that region's id, page, box
//! and method. The matcher is deterministic string work, not a model: a model
//! asked "what is the design pressure" answers whether or not the page says,
//! and this answers only when it does.
//!
//! Three shapes are recognised, in this order:
//!
//! 1. a table whose header cell names the field — the values below it, or the
//!    cell to its right in a key/value table;
//! 2. a line `Field: value`, `Field = value` or `Field - value`;
//! 3. a line that starts with the field name followed by a value.
//!
//! A field found nowhere is `not-found`, and the answer says which pages were
//! searched and which were not read at all — "not on the pages that were read"
//! is not "not in the document". A field found in a region whose read was cut
//! (looped, truncated, malformed) is reported with that status and is never
//! the only basis for a checked value.

use serde::Serialize;

use super::regions::{BBox, EvidenceRegion, Method, PageSpace, RegionStatus};

pub const MAX_FIELDS: usize = 12;
pub const MAX_FIELD_CHARS: usize = 80;
pub const MAX_VALUES_PER_FIELD: usize = 6;
const MAX_VALUE_CHARS: usize = 160;

/// One place a field's value was read.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldValue {
    pub value: String,
    pub region_id: String,
    pub page: u32,
    pub bbox: BBox,
    pub coord_space: PageSpace,
    pub method: Method,
    pub status: RegionStatus,
    /// How it was matched: `table-column`, `table-row`, `labelled-line`.
    pub matched_as: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum FieldState {
    /// One value, from a region read cleanly.
    Found,
    /// Several different values. All are reported; none is chosen.
    Conflicting,
    /// Found only in regions whose read was cut or malformed.
    Uncertain,
    NotFound,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FieldResult {
    pub field: String,
    pub state: FieldState,
    pub values: Vec<FieldValue>,
}

/// Validates and normalises what the caller asked for.
pub fn requested(fields: &[String]) -> Result<Vec<String>, String> {
    if fields.len() > MAX_FIELDS {
        return Err(format!(
            "{} fields were asked for and at most {MAX_FIELDS} may be at once",
            fields.len()
        ));
    }
    let mut out: Vec<String> = Vec::new();
    for field in fields {
        let field = field.trim();
        if field.is_empty() {
            continue;
        }
        if field.chars().count() > MAX_FIELD_CHARS {
            return Err(format!(
                "a field name is at most {MAX_FIELD_CHARS} characters; name the field, not the \
                 value you expect"
            ));
        }
        if !out.iter().any(|held| held.eq_ignore_ascii_case(field)) {
            out.push(field.to_string());
        }
    }
    Ok(out)
}

/// Lower case, punctuation to spaces, whitespace collapsed.
fn fold(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_alphanumeric() { c.to_ascii_lowercase() } else { ' ' })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn names(cell: &str, field: &str) -> bool {
    let (cell, field) = (fold(cell), fold(field));
    !field.is_empty() && (cell == field || cell.starts_with(&format!("{field} ")))
}

fn bounded(value: &str) -> String {
    let value = value.trim().trim_matches(|c: char| c == ';' || c == ',' || c == '|').trim();
    value.chars().take(MAX_VALUE_CHARS).collect()
}

/// Finds each field in the regions given. Proposals are never searched: a
/// vision model's description is not where a value is read from.
pub fn find(fields: &[String], regions: &[&EvidenceRegion]) -> Vec<FieldResult> {
    let transcribed: Vec<&EvidenceRegion> = regions
        .iter()
        .copied()
        .filter(|region| region.method.is_transcription())
        .collect();
    fields
        .iter()
        .map(|field| {
            let mut values = Vec::new();
            for region in &transcribed {
                values.extend(in_table(field, region));
                values.extend(in_lines(field, region));
            }
            // One value per region and text: a table found by both routes is
            // one reading, not two.
            let mut seen = std::collections::BTreeSet::new();
            values.retain(|v: &FieldValue| seen.insert((v.region_id.clone(), v.value.clone())));
            values.truncate(MAX_VALUES_PER_FIELD);

            let clean: Vec<&FieldValue> = values
                .iter()
                .filter(|v| v.status == RegionStatus::Read)
                .collect();
            let distinct: std::collections::BTreeSet<String> =
                clean.iter().map(|v| fold(&v.value)).collect();
            let state = if values.is_empty() {
                FieldState::NotFound
            } else if clean.is_empty() {
                FieldState::Uncertain
            } else if distinct.len() > 1 {
                FieldState::Conflicting
            } else {
                FieldState::Found
            };
            FieldResult {
                field: field.clone(),
                state,
                values,
            }
        })
        .collect()
}

fn value_at(region: &EvidenceRegion, value: &str, bbox: Option<BBox>, matched_as: &str) -> FieldValue {
    FieldValue {
        value: bounded(value),
        region_id: region.region_id.clone(),
        page: region.page,
        bbox: bbox.unwrap_or(region.bbox),
        coord_space: region.coord_space,
        method: region.method,
        status: region.status,
        matched_as: matched_as.to_string(),
    }
}

fn in_table(field: &str, region: &EvidenceRegion) -> Vec<FieldValue> {
    let mut out = Vec::new();
    if region.cells.is_empty() {
        return out;
    }
    let cols = region.cells.iter().map(|c| c.col).max().unwrap_or(0) + 1;
    for header in region.cells.iter().filter(|c| names(&c.text, field)) {
        // A column header: the values below it.
        if header.row == 0 && cols > 1 {
            for cell in region
                .cells
                .iter()
                .filter(|c| c.col == header.col && c.row > header.row && !c.text.trim().is_empty())
            {
                out.push(value_at(region, &cell.text, cell.bbox, "table-column"));
            }
        }
        // A key/value row: the cell to the right.
        if header.col == 0 || header.row > 0 {
            if let Some(cell) = region
                .cells
                .iter()
                .find(|c| c.row == header.row && c.col == header.col + 1 && !c.text.trim().is_empty())
            {
                out.push(value_at(region, &cell.text, cell.bbox, "table-row"));
            }
        }
    }
    out
}

fn in_lines(field: &str, region: &EvidenceRegion) -> Vec<FieldValue> {
    let mut out = Vec::new();
    if !region.cells.is_empty() {
        return out;
    }
    let wanted = fold(field);
    if wanted.is_empty() {
        return out;
    }
    for line in region.text.lines() {
        let trimmed = line.trim();
        // `Field: value`, `Field = value`, `Field - value`.
        let split = trimmed
            .char_indices()
            .find(|(_, c)| *c == ':' || *c == '=' || *c == '\u{2013}')
            .or_else(|| trimmed.find(" - ").map(|at| (at + 1, '-')));
        if let Some((at, separator)) = split {
            let (key, rest) = (&trimmed[..at], &trimmed[at + separator.len_utf8()..]);
            if fold(key) == wanted && !rest.trim().is_empty() {
                out.push(value_at(region, rest, None, "labelled-line"));
                continue;
            }
        }
        // `Field value` at the start of a line, when the field is at least two
        // characters and the line carries something after it.
        let folded = fold(trimmed);
        if wanted.chars().count() >= 2 && folded.starts_with(&format!("{wanted} ")) {
            let words = field.split_whitespace().count();
            let rest: Vec<&str> = trimmed.split_whitespace().skip(words).collect();
            if !rest.is_empty() {
                out.push(value_at(region, &rest.join(" "), None, "labelled-line"));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::regions::{Extractor, TableCell};

    fn region(id: &str, method: Method, status: RegionStatus, text: &str, cells: Vec<TableCell>) -> EvidenceRegion {
        EvidenceRegion {
            region_id: id.into(),
            document_sha256: "0".repeat(64),
            page: 2,
            bbox: BBox::new(1.0, 2.0, 3.0, 4.0),
            coord_space: PageSpace::PdfPoints,
            label: "text".into(),
            method,
            status,
            text: text.into(),
            cells,
            notes: Vec::new(),
            crop_id: None,
            image_sha256: None,
            extractor: Extractor::default(),
            cache_key: None,
        }
    }

    fn cell(row: u32, col: u32, text: &str) -> TableCell {
        TableCell { row, col, text: text.into(), bbox: Some(BBox::new(col as f64, row as f64, col as f64 + 1.0, row as f64 + 1.0)) }
    }

    #[test]
    fn a_labelled_line_gives_its_value_and_its_region() {
        let r = region("rg-a", Method::Ocr, RegionStatus::Read, "Tag: PT-2201\nDesign pressure = 10 bar", vec![]);
        let found = find(&["design pressure".into(), "tag".into()], &[&r]);
        assert_eq!(found[0].state, FieldState::Found);
        assert_eq!(found[0].values[0].value, "10 bar");
        assert_eq!(found[0].values[0].region_id, "rg-a");
        assert_eq!(found[0].values[0].method, Method::Ocr);
        assert_eq!(found[1].values[0].value, "PT-2201");
    }

    #[test]
    fn a_table_header_gives_the_column_below_it_with_cell_boxes() {
        let r = region(
            "rg-t",
            Method::EmbeddedTable,
            RegionStatus::Read,
            "",
            vec![cell(0, 0, "Point"), cell(0, 1, "mm"), cell(1, 0, "A"), cell(1, 1, "9.4"), cell(2, 0, "C"), cell(2, 1, "8.2")],
        );
        let found = find(&["mm".into()], &[&r]);
        let values: Vec<&str> = found[0].values.iter().map(|v| v.value.as_str()).collect();
        assert_eq!(values, vec!["9.4", "8.2"]);
        assert_eq!(found[0].state, FieldState::Conflicting, "two readings are not one value");
        assert_eq!(found[0].values[0].bbox, BBox::new(1.0, 1.0, 2.0, 2.0));
    }

    #[test]
    fn a_value_only_in_a_looped_read_is_uncertain_and_a_proposal_is_never_searched() {
        let looped = region("rg-l", Method::Ocr, RegionStatus::Looped, "Tag: PT-9", vec![]);
        let proposal = region("rg-v", Method::VisionInference, RegionStatus::Read, "Tag: PT-1", vec![]);
        let found = find(&["tag".into()], &[&looped, &proposal]);
        assert_eq!(found[0].state, FieldState::Uncertain);
        assert_eq!(found[0].values.len(), 1);
        assert_eq!(found[0].values[0].region_id, "rg-l");
    }

    #[test]
    fn a_field_nowhere_is_not_found_and_a_request_is_bounded() {
        let r = region("rg-a", Method::EmbeddedText, RegionStatus::Read, "Nothing relevant", vec![]);
        assert_eq!(find(&["flow rate".into()], &[&r])[0].state, FieldState::NotFound);
        assert!(requested(&vec!["x".to_string(); MAX_FIELDS + 1]).is_err());
        assert_eq!(requested(&["Tag".into(), "tag".into(), " ".into()]).unwrap(), vec!["Tag"]);
    }
}
