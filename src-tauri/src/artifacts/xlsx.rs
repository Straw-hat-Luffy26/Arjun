//! The calculation workbook — working a reviewer can check without redoing it.
//!
//! ARJUN design rule 27 asks that numerical work be done by a deterministic engine and
//! shown step by step: *"inputs, units, formula, assumptions, intermediate
//! values, result, rounding rule"*. [`orchestrator::calculation`] already
//! produces exactly that record. This module writes it into a spreadsheet,
//! which is where a process engineer actually checks arithmetic.
//!
//! ## Excel is asked to disagree
//!
//! Where the expression is plain arithmetic, the result cell is written as a
//! **live formula**, not the number ARJUN computed. Excel recalculates it on
//! open. If the two ever disagreed, the workbook would show it — which makes
//! the file a check on ARJUN rather than a restatement of it.
//!
//! Where the expression carries units (`120 bar * 0.5 m^2`) no honest
//! translation to an Excel formula exists, so the value is written as a number
//! and the cell beside it says why. A workbook that silently degraded from
//! "Excel confirms this" to "ARJUN asserts this" would be worse than one that
//! never claimed the first thing.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::ooxml::{escape, read_part, write_parts};
use crate::orchestrator::calculation::CalculationRecord;

/// A cell's content, which decides how it is written into the sheet.
enum Cell {
    Text(String),
    Number(f64),
    /// A live formula. `cached` is what ARJUN computed, so the file reads
    /// correctly in viewers that do not recalculate.
    Formula { formula: String, cached: f64 },
    Empty,
}

fn column_letter(index: usize) -> String {
    // A..Z then AA.. — the workbook never needs more than a handful, but
    // getting this wrong produces a file Excel refuses rather than a wide one.
    let mut n = index;
    let mut letters = Vec::new();
    loop {
        letters.push((b'A' + (n % 26) as u8) as char);
        if n < 26 {
            break;
        }
        n = n / 26 - 1;
    }
    letters.iter().rev().collect()
}

fn cell_xml(row: usize, column: usize, cell: &Cell) -> String {
    let reference = format!("{}{}", column_letter(column), row);
    match cell {
        Cell::Empty => String::new(),
        Cell::Text(text) => format!(
            "<c r=\"{reference}\" t=\"inlineStr\"><is><t xml:space=\"preserve\">{}</t></is></c>",
            escape(text)
        ),
        Cell::Number(value) => format!("<c r=\"{reference}\"><v>{value}</v></c>"),
        Cell::Formula { formula, cached } => format!(
            "<c r=\"{reference}\"><f>{}</f><v>{cached}</v></c>",
            escape(formula)
        ),
    }
}

/// Whether an expression can be handed to Excel unchanged.
///
/// Deliberately conservative. Anything carrying a unit, a name or a function
/// call is refused, because a translation that was *nearly* right would produce
/// a workbook that quietly disagrees with the record it is supposed to show.
fn is_plain_arithmetic(expression: &str) -> bool {
    !expression.trim().is_empty()
        && expression
            .chars()
            .all(|c| c.is_ascii_digit() || " .+-*/()^".contains(c))
}

fn sheet_xml(rows: &[Vec<Cell>]) -> String {
    let mut body = String::new();
    for (index, row) in rows.iter().enumerate() {
        let number = index + 1;
        let cells: String = row
            .iter()
            .enumerate()
            .map(|(column, cell)| cell_xml(number, column, cell))
            .collect();
        body.push_str(&format!("<row r=\"{number}\">{cells}</row>"));
    }

    format!(
        r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
<cols><col min="1" max="1" width="34" customWidth="1"/><col min="2" max="2" width="22" customWidth="1"/><col min="3" max="3" width="46" customWidth="1"/></cols>
<sheetData>{body}</sheetData></worksheet>"#
    )
}

const CONTENT_TYPES: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
</Types>"#;

const ROOT_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>"#;

// `fullCalcOnLoad` is the point of the whole design: Excel recomputes every
// formula when the file opens rather than trusting the cached values ARJUN
// wrote, so a disagreement would be visible immediately.
const WORKBOOK: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<sheets><sheet name="Calculation" sheetId="1" r:id="rId1"/></sheets>
<calcPr calcId="0" fullCalcOnLoad="1"/>
</workbook>"#;

const WORKBOOK_RELS: &str = r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
</Relationships>"#;

/// What a produced workbook turned out to contain.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct WorkbookCheck {
    pub opens: bool,
    /// Calculations found in the sheet.
    pub calculations: usize,
    /// How many of those Excel will recompute rather than take on trust.
    pub live_formulas: usize,
    pub problems: Vec<String>,
}

impl WorkbookCheck {
    pub fn is_sound(&self) -> bool {
        self.opens && self.problems.is_empty()
    }
}

fn rows_for(records: &[CalculationRecord], classification: &str) -> Vec<Vec<Cell>> {
    let mut rows: Vec<Vec<Cell>> = Vec::new();

    rows.push(vec![Cell::Text("Calculation record".into())]);
    rows.push(vec![Cell::Text(format!("Classification: {classification}"))]);
    rows.push(vec![Cell::Text(
        "Computed by ARJUN's calculation engine. Formula cells recompute in Excel.".into(),
    )]);
    rows.push(Vec::new());

    for (index, record) in records.iter().enumerate() {
        if index > 0 {
            rows.push(Vec::new());
        }

        rows.push(vec![Cell::Text(format!("Calculation {}", index + 1))]);
        rows.push(vec![
            Cell::Text("Expression".into()),
            Cell::Empty,
            Cell::Text(record.expression.clone()),
        ]);

        for input in &record.inputs {
            rows.push(vec![
                Cell::Text("Input".into()),
                Cell::Empty,
                Cell::Text(input.clone()),
            ]);
        }

        for (step_index, step) in record.steps.iter().enumerate() {
            rows.push(vec![
                Cell::Text(format!("Step {}", step_index + 1)),
                Cell::Text(step.result.clone()),
                Cell::Text(step.description.clone()),
            ]);
        }

        let (result_cell, note) = if is_plain_arithmetic(&record.expression) {
            (
                Cell::Formula {
                    formula: record.expression.trim().to_string(),
                    cached: record.value,
                },
                "Excel recomputes this cell on open. It should equal the result below.".to_string(),
            )
        } else {
            (
                Cell::Number(record.value),
                format!(
                    "Written as a value, not a formula: the expression carries units, which Excel \
                     has no way to evaluate. Computed by ARJUN{}.",
                    if record.deterministic { " deterministically" } else { "" }
                ),
            )
        };

        rows.push(vec![Cell::Text("Recomputed".into()), result_cell, Cell::Text(note)]);
        rows.push(vec![
            Cell::Text("Result".into()),
            Cell::Text(record.formatted.clone()),
            Cell::Text(if record.unit.is_empty() {
                "dimensionless".to_string()
            } else {
                format!("unit: {}", record.unit)
            }),
        ]);
        rows.push(vec![
            Cell::Text("Rounding".into()),
            Cell::Empty,
            Cell::Text(record.rounding.clone()),
        ]);
    }

    rows
}

/// Writes the calculation workbook.
pub fn write_workbook(
    path: &Path,
    records: &[CalculationRecord],
    classification: &str,
) -> Result<(), String> {
    if records.is_empty() {
        // An empty workbook would look like a calculation that produced nothing
        // rather than one that was never run.
        return Err("There are no calculations to write. Nothing was written.".to_string());
    }

    let parts = [
        ("[Content_Types].xml", CONTENT_TYPES.to_string()),
        ("_rels/.rels", ROOT_RELS.to_string()),
        ("xl/workbook.xml", WORKBOOK.to_string()),
        ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS.to_string()),
        ("xl/worksheets/sheet1.xml", sheet_xml(&rows_for(records, classification))),
    ];

    write_parts(path, &parts).map_err(|e| format!("The workbook could not be written: {e}"))
}

/// Re-opens a produced workbook and reports what is in it.
pub fn check_workbook(path: &Path) -> WorkbookCheck {
    let sheet = match read_part(path, "xl/worksheets/sheet1.xml") {
        Ok(sheet) => sheet,
        Err(error) => {
            return WorkbookCheck {
                opens: false,
                calculations: 0,
                live_formulas: 0,
                problems: vec![format!("{}: {error}", path.display())],
            }
        }
    };

    let mut problems = Vec::new();
    for required in ["xl/workbook.xml", "xl/_rels/workbook.xml.rels", "[Content_Types].xml"] {
        if read_part(path, required).is_err() {
            problems.push(format!("the workbook is missing {required}"));
        }
    }

    // "Calculation 1", not the "Calculation record" title — counting the title
    // as a calculation would report an empty workbook as holding one.
    let calculations = sheet
        .split("Calculation ")
        .skip(1)
        .filter(|fragment| fragment.starts_with(|c: char| c.is_ascii_digit()))
        .count();
    let live_formulas = sheet.matches("<f>").count();

    if calculations == 0 {
        problems.push("the workbook contains no calculations".to_string());
    }
    if !sheet.contains("Rounding") {
        problems.push("the workbook does not state its rounding rule".to_string());
    }

    WorkbookCheck { opens: true, calculations, live_formulas, problems }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestrator::calculation::evaluate;

    fn temp() -> tempfile::TempDir {
        tempfile::tempdir().unwrap()
    }

    #[test]
    fn column_letters_run_past_z() {
        assert_eq!(column_letter(0), "A");
        assert_eq!(column_letter(25), "Z");
        assert_eq!(column_letter(26), "AA");
        assert_eq!(column_letter(27), "AB");
    }

    /// The workbook this writes must be readable by the reader that reads it.
    ///
    /// Both halves were correct on their own and did not meet. This writer
    /// emits every label as an *inline* string — `<c t="inlineStr"><is><t>` —
    /// and `attachment_extract.py` looked only in `<v>`, which an inline
    /// string does not have. It also laid cells out by their order in the XML
    /// rather than by the square each one names, and `rows_for` below emits
    /// `Text, Empty, Text` with the empty cell written as nothing at all.
    ///
    /// Together those two facts did more than drop the headings: a row of
    /// label-gap-value came back as `" | "`, which the reader discards as
    /// blank. ARJUN's own calculation workbook — the artifact that exists so a
    /// figure can be checked six months later — read back as almost nothing,
    /// and nothing anywhere said so.
    ///
    /// This is the test that cannot go stale, because it runs the real writer
    /// into the real reader. A unit test on either side would have passed
    /// throughout, and both did.
    #[test]
    fn a_workbook_this_writes_can_be_read_back_by_the_document_extractor() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let record = evaluate("(9.0 - 8.2) / 9.0 * 100").unwrap();
        write_workbook(&path, std::slice::from_ref(&record), "Inspection report").unwrap();

        let script = crate::deployment::require_path("document-extractor")
            .expect("the attachment extractor ships with this build");
        let output = std::process::Command::new(crate::deployment::program("python"))
            .arg(&script)
            .arg(&path)
            .arg(dir.path())
            .output()
            .expect(
                "python is a core dependency of this build (see crate::deployment); \
                 the writer and the reader cannot be checked against each other without it",
            );

        let stdout = String::from_utf8_lossy(&output.stdout);
        let line = stdout
            .lines()
            .rev()
            .find(|line| line.trim_start().starts_with('{'))
            .unwrap_or_else(|| {
                panic!(
                    "the extractor printed no JSON: {}",
                    String::from_utf8_lossy(&output.stderr)
                )
            });
        let parsed: serde_json::Value = serde_json::from_str(line).expect("well-formed JSON");
        let text = parsed["text"].as_str().unwrap_or_default();

        assert_eq!(parsed["kind"], "xlsx");
        // The labels this writer puts in column A, which used to vanish.
        for label in ["Calculation record", "Expression", "Result"] {
            assert!(text.contains(label), "the label {label:?} was lost:\n{text}");
        }
        // This writer emits one sheet, so no sheet header is expected: naming
        // the only tab there is answers a question nobody can ask. Asserting
        // its *absence* rather than its presence, because "Calculation" also
        // appears in the body above — an assertion that passed on the body
        // while believing it had found a header would be worse than none.
        // The multi-sheet case is covered in
        // sidecars/document_sidecar/tests/test_attachment_xlsx.py.
        assert!(
            !text.contains("--- sheet:"),
            "a one-sheet workbook should carry no sheet header:\n{text}"
        );
        // The expression, which sits in column C behind an empty column B —
        // the gap that used to shift every value one place left.
        assert!(
            text.contains("(9.0 - 8.2) / 9.0 * 100"),
            "the expression was lost:\n{text}"
        );
        assert!(
            text.contains(&record.formatted),
            "the result {:?} was lost:\n{text}",
            record.formatted
        );
    }

    #[test]
    fn a_workbook_is_written_and_reopens_soundly() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let record = evaluate("(9.0 - 8.2) / 9.0 * 100").unwrap();

        write_workbook(&path, std::slice::from_ref(&record), "Inspection report").unwrap();

        let check = check_workbook(&path);
        assert!(check.is_sound(), "{:?}", check.problems);
        assert_eq!(check.calculations, 1);
    }

    /// The point of the workbook: Excel recomputes and can disagree.
    #[test]
    fn plain_arithmetic_is_written_as_a_formula_excel_will_recompute() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let record = evaluate("120 * 0.85").unwrap();

        write_workbook(&path, std::slice::from_ref(&record), "Test").unwrap();

        let check = check_workbook(&path);
        assert_eq!(check.live_formulas, 1);

        let sheet = read_part(&path, "xl/worksheets/sheet1.xml").unwrap();
        assert!(sheet.contains("<f>120 * 0.85</f>"));
        assert!(sheet.contains("fullCalcOnLoad") || {
            read_part(&path, "xl/workbook.xml").unwrap().contains("fullCalcOnLoad")
        });
    }

    /// A workbook that silently degraded from "Excel confirms this" to "ARJUN
    /// asserts this" would be worse than one that never claimed the first.
    #[test]
    fn an_expression_with_units_is_a_value_and_says_why() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let record = evaluate("120 bar * 0.5").unwrap();

        write_workbook(&path, std::slice::from_ref(&record), "Test").unwrap();

        let sheet = read_part(&path, "xl/worksheets/sheet1.xml").unwrap();
        assert_eq!(check_workbook(&path).live_formulas, 0);
        assert!(sheet.contains("carries units"));
    }

    #[test]
    fn arithmetic_detection_refuses_anything_it_cannot_hand_to_excel() {
        assert!(is_plain_arithmetic("1 + 2 * (3 - 4)"));
        assert!(is_plain_arithmetic("2^10"));
        assert!(!is_plain_arithmetic("120 bar"));
        assert!(!is_plain_arithmetic("sqrt(4)"));
        assert!(!is_plain_arithmetic(""));
    }

    #[test]
    fn every_calculation_carries_its_rounding_rule() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let record = evaluate("10 / 3").unwrap();

        write_workbook(&path, std::slice::from_ref(&record), "Test").unwrap();
        let sheet = read_part(&path, "xl/worksheets/sheet1.xml").unwrap();
        assert!(sheet.contains(&escape(&record.rounding)));
    }

    #[test]
    fn several_calculations_share_one_sheet() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let records = vec![evaluate("1 + 1").unwrap(), evaluate("2 * 3").unwrap()];

        write_workbook(&path, &records, "Test").unwrap();
        assert_eq!(check_workbook(&path).calculations, 2);
    }

    /// An empty workbook reads as a calculation that produced nothing.
    #[test]
    fn no_calculations_refuses_rather_than_writing_an_empty_workbook() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");

        let error = write_workbook(&path, &[], "Test").unwrap_err();
        assert!(error.contains("no calculations"));
        assert!(!path.exists());
    }

    #[test]
    fn model_content_containing_xml_still_produces_a_readable_workbook() {
        let dir = temp();
        let path = dir.path().join("calc.xlsx");
        let record = evaluate("1 + 1").unwrap();

        write_workbook(&path, std::slice::from_ref(&record), "<classified> & \"secret\"").unwrap();
        assert!(check_workbook(&path).is_sound());
    }

    #[test]
    fn a_file_that_is_not_a_workbook_is_reported_rather_than_panicking() {
        let dir = temp();
        let path = dir.path().join("broken.xlsx");
        std::fs::write(&path, "not a zip").unwrap();

        let check = check_workbook(&path);
        assert!(!check.opens);
        assert!(!check.is_sound());
    }
}

/// Writes an ordinary table as a workbook.
///
/// Reuses the parts and the sheet writer that `write_workbook` uses, because a
/// table of figures and a table of calculation steps are the same file with
/// different rows. The only judgement here is which cells are numbers: a column
/// of quantities that arrives as text sorts and sums wrongly in Excel, and the
/// person who opens it will not think to check.
pub fn write_table(
    path: &Path,
    title: &str,
    header: &[String],
    rows: &[Vec<String>],
    classification: &str,
) -> Result<(), String> {
    if header.is_empty() {
        return Err("A table needs a header row. Nothing was written.".to_string());
    }
    if rows.is_empty() {
        return Err("A table with no rows would be an empty sheet. Nothing was written.".to_string());
    }
    for (index, row) in rows.iter().enumerate() {
        if row.len() != header.len() {
            return Err(format!(
                "Row {} has {} cell(s) for {} column(s). Every row must match the header; \
                 nothing was written.",
                index + 1,
                row.len(),
                header.len()
            ));
        }
    }

    let mut sheet: Vec<Vec<Cell>> = Vec::with_capacity(rows.len() + 3);
    sheet.push(vec![Cell::Text(title.to_string())]);
    if !classification.trim().is_empty() {
        sheet.push(vec![Cell::Text(format!(
            "Classification: {}",
            classification.trim()
        ))]);
    }
    sheet.push(header.iter().map(|h| Cell::Text(h.clone())).collect());

    for row in rows {
        sheet.push(
            row.iter()
                .map(|value| {
                    // Parsed rather than assumed. A value that is a number is
                    // written as one so the sheet can add it up; anything else
                    // stays exactly the text it arrived as.
                    match value.trim().parse::<f64>() {
                        Ok(number) if value.trim().parse::<f64>().is_ok() => Cell::Number(number),
                        _ => Cell::Text(value.clone()),
                    }
                })
                .collect(),
        );
    }

    let parts = [
        ("[Content_Types].xml", CONTENT_TYPES.to_string()),
        ("_rels/.rels", ROOT_RELS.to_string()),
        ("xl/workbook.xml", WORKBOOK.to_string()),
        ("xl/_rels/workbook.xml.rels", WORKBOOK_RELS.to_string()),
        ("xl/worksheets/sheet1.xml", sheet_xml(&sheet)),
    ];
    write_parts(path, &parts).map_err(|e| format!("The table could not be written: {e}"))
}

#[cfg(test)]
mod table_tests {
    use super::*;

    fn rows() -> Vec<Vec<String>> {
        vec![
            vec!["PV-2201".into(), "1".into(), "45000".into()],
            vec!["PT-2201".into(), "2".into(), "8250.5".into()],
        ]
    }

    fn header() -> Vec<String> {
        vec!["Item".into(), "Qty".into(), "Cost".into()]
    }

    #[test]
    fn a_table_becomes_a_workbook_that_opens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.xlsx");
        write_table(&path, "Spares", &header(), &rows(), "OFFICIAL").unwrap();

        // `is_sound` is the *calculation workbook* contract - it wants
        // calculations and a stated rounding rule, and a table of spares has
        // neither by design. What matters here is that the file opens and the
        // rows are in it.
        let check = check_workbook(&path);
        assert!(check.opens, "the workbook does not open: {check:?}");

        let sheet = read_part(&path, "xl/worksheets/sheet1.xml").unwrap();
        assert!(sheet.contains("Spares"), "{sheet}");
        assert!(sheet.contains("Item"), "{sheet}");
        assert!(sheet.contains("OFFICIAL"), "{sheet}");
    }

    /// Numbers must be numbers, or the sheet cannot add them up.
    #[test]
    fn numeric_cells_are_written_as_numbers() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("t.xlsx");
        write_table(&path, "Spares", &header(), &rows(), "OFFICIAL").unwrap();

        let sheet = read_part(&path, "xl/worksheets/sheet1.xml").unwrap();
        assert!(sheet.contains("<v>45000</v>"), "{sheet}");
        assert!(sheet.contains("<v>8250.5</v>"), "{sheet}");
        // And the item code, which merely looks like data, stayed text.
        assert!(sheet.contains("PV-2201"), "{sheet}");
    }

    #[test]
    fn a_ragged_row_is_refused_rather_than_padded() {
        let dir = tempfile::tempdir().unwrap();
        let mut ragged = rows();
        ragged[1].pop();
        let problem = write_table(
            &dir.path().join("t.xlsx"),
            "Spares",
            &header(),
            &ragged,
            "OFFICIAL",
        )
        .unwrap_err();
        assert!(problem.contains("2 cell(s) for 3 column(s)"), "{problem}");
        assert!(problem.contains("nothing was written"));
    }

    #[test]
    fn an_empty_table_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert!(write_table(&dir.path().join("t.xlsx"), "x", &header(), &[], "OFFICIAL")
            .unwrap_err()
            .contains("empty sheet"));
        assert!(write_table(&dir.path().join("t.xlsx"), "x", &[], &rows(), "OFFICIAL")
            .unwrap_err()
            .contains("header row"));
    }
}
