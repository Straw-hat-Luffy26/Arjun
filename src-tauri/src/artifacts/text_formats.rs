//! The text formats: CSV, JSON, YAML, XML, Markdown and HTML.
//!
//! ## What was here before
//!
//! `write_scoped_file` — one untyped text writer, mapped to
//! `workspace.write_text`. It took a string and a path and wrote the bytes. No
//! schema, no escaping rules, no encoding decision, no structural check, and a
//! validator that reported `Present, N byte(s).` A CSV whose rows had different
//! widths, a JSON file with a trailing comma, an HTML page with no `lang` and
//! unescaped content in it — all written, all reported as produced.
//!
//! ## Serialise, then re-parse
//!
//! Every serialiser here has a matching parser, and the tests use both: content
//! is built into a value, serialised, and read back before it is trusted. That
//! is the discipline the OOXML and PDF paths follow, and it catches the failure
//! a writer alone never can — a quoting bug produces a file the writer is
//! perfectly happy with.
//!
//! ## Why the parsers are hand-written
//!
//! Only `serde_json` is in the dependency tree. This product ships an SBOM for
//! an air-gapped deployment and has already refused a PDF crate over its
//! transitive count, so pulling in a YAML, an XML and a CSV crate to validate
//! files this code also writes is a cost with a cheaper alternative: each parser
//! here handles the subset the matching serialiser emits.
//!
//! The limit is stated rather than hidden. [`Check::partial`] is set when a
//! parser recognised the document but knows it is not a complete implementation
//! of the format, so a caller never mistakes "my parser was happy" for "this is
//! valid YAML by the specification".

use std::path::Path;

use serde_json::Value;

use super::doc_model::{Block, Document};

/// A text format this product can write and read back.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Csv,
    Json,
    Yaml,
    Xml,
    Markdown,
    Html,
}

impl Format {
    pub fn of_path(path: &Path) -> Option<Format> {
        let extension = path.extension()?.to_str()?.to_lowercase();
        Some(match extension.as_str() {
            "csv" => Format::Csv,
            "json" => Format::Json,
            "yaml" | "yml" => Format::Yaml,
            "xml" => Format::Xml,
            "md" | "markdown" => Format::Markdown,
            "html" | "htm" => Format::Html,
            _ => return None,
        })
    }

    pub const fn name(self) -> &'static str {
        match self {
            Format::Csv => "CSV",
            Format::Json => "JSON",
            Format::Yaml => "YAML",
            Format::Xml => "XML",
            Format::Markdown => "Markdown",
            Format::Html => "HTML",
        }
    }
}

/// What reading a text artifact back found.
#[derive(Debug, Clone, Default)]
pub struct Check {
    pub parses: bool,
    /// Records, keys, elements or headings, depending on the format. For the
    /// message rather than for a decision.
    pub items: usize,
    /// True when the parser handles only the subset this product writes.
    pub partial: bool,
    pub problems: Vec<String>,
}

impl Check {
    pub fn is_sound(&self) -> bool {
        self.parses && self.problems.is_empty()
    }

    fn bad(problem: String) -> Self {
        Self { parses: false, problems: vec![problem], ..Default::default() }
    }
}

/// Checks text against the format its extension claims.
pub fn check(format: Format, text: &str) -> Check {
    match format {
        Format::Csv => check_csv(text),
        Format::Json => check_json(text),
        Format::Yaml => check_yaml(text),
        Format::Xml => check_xml(text),
        Format::Markdown => check_markdown(text),
        Format::Html => check_html(text),
    }
}

// ── CSV ──────────────────────────────────────────────────────────────────

/// Writes RFC 4180 CSV.
///
/// Quoting is not optional and not a matter of taste: a field containing a
/// comma, a quote, a carriage return or a newline **must** be quoted, and a
/// quote inside a quoted field is doubled. Getting this wrong produces a file
/// that opens and is silently wrong — a description containing a comma becomes
/// two columns, and every row after it is misaligned.
///
/// UTF-8 with no byte-order mark. A BOM makes the first header cell read as
/// `\u{feff}Tag` in most parsers, which is a bug reported as "the first column
/// has a strange name".
pub fn to_csv(header: &[String], rows: &[Vec<String>]) -> String {
    let mut out = String::new();
    out.push_str(&csv_record(header));
    for row in rows {
        out.push_str(&csv_record(row));
    }
    out
}

fn csv_record(fields: &[String]) -> String {
    let mut line = String::new();
    for (index, field) in fields.iter().enumerate() {
        if index > 0 {
            line.push(',');
        }
        line.push_str(&csv_field(field));
    }
    line.push_str("\r\n"); // RFC 4180 says CRLF.
    line
}

fn csv_field(field: &str) -> String {
    if field.contains([',', '"', '\n', '\r']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// Parses CSV, honouring quotes. Returns the records.
pub fn parse_csv(text: &str) -> Result<Vec<Vec<String>>, String> {
    let mut records: Vec<Vec<String>> = Vec::new();
    let mut record: Vec<String> = Vec::new();
    let mut field = String::new();
    let mut quoted = false;
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if quoted {
            match c {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        chars.next();
                        field.push('"');
                    } else {
                        quoted = false;
                    }
                }
                other => field.push(other),
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => quoted = true,
            '"' => return Err("a quote appears in the middle of an unquoted field".to_string()),
            ',' => record.push(std::mem::take(&mut field)),
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            other => field.push(other),
        }
    }
    if quoted {
        return Err("a quoted field is never closed".to_string());
    }
    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    Ok(records)
}

fn check_csv(text: &str) -> Check {
    if text.starts_with('\u{feff}') {
        return Check::bad(
            "the file begins with a byte-order mark, which most readers show as part of the \
             first column's name"
                .to_string(),
        );
    }
    let records = match parse_csv(text) {
        Ok(records) => records,
        Err(problem) => return Check::bad(problem),
    };
    let mut check = Check { parses: true, items: records.len(), ..Default::default() };

    let Some(header) = records.first() else {
        check.problems.push("the file is empty: a CSV needs at least a header row".to_string());
        return check;
    };
    if header.iter().any(|h| h.trim().is_empty()) {
        check.problems.push("a column in the header row has no name".to_string());
    }
    // Every record the same width. This is the failure that makes a spreadsheet
    // silently misaligned rather than visibly broken.
    for (index, record) in records.iter().enumerate().skip(1) {
        if record.len() != header.len() {
            check.problems.push(format!(
                "row {} has {} field(s) against {} column(s)",
                index + 1,
                record.len(),
                header.len()
            ));
        }
    }
    check
}

// ── JSON ─────────────────────────────────────────────────────────────────

pub fn to_json(value: &Value) -> String {
    // Pretty, with a trailing newline: a JSON file a person may open and a tool
    // may diff.
    format!("{}\n", serde_json::to_string_pretty(value).unwrap_or_else(|_| "null".to_string()))
}

fn check_json(text: &str) -> Check {
    match serde_json::from_str::<Value>(text) {
        Ok(value) => {
            let mut check = Check { parses: true, ..Default::default() };
            check.items = match &value {
                Value::Object(map) => map.len(),
                Value::Array(items) => items.len(),
                _ => 1,
            };
            // A JSON artifact that is a bare string is almost always prose that
            // was supposed to be structured.
            if matches!(value, Value::String(_)) {
                check.problems.push(
                    "the document is a single JSON string rather than an object or an array, \
                     which usually means prose was written where structure was wanted"
                        .to_string(),
                );
            }
            check
        }
        Err(error) => Check::bad(format!("it is not valid JSON: {error}")),
    }
}

// ── YAML ─────────────────────────────────────────────────────────────────

/// Writes the YAML subset this product emits: nested maps, sequences and
/// scalars, two-space indentation.
///
/// Every scalar that could be read as something else is quoted — a value like
/// `yes`, `null`, `1.0` or `08:00` means a boolean, a nothing, a number and a
/// sexagesimal in YAML 1.1, and a file that changes meaning because a string
/// looked like a number is the classic YAML defect.
pub fn to_yaml(value: &Value) -> String {
    let mut out = String::new();
    yaml_value(value, 0, &mut out);
    out
}

fn yaml_scalar(value: &Value) -> String {
    match value {
        Value::Null => "null".to_string(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => {
            let ambiguous = s.is_empty()
                || s.parse::<f64>().is_ok()
                || matches!(
                    s.to_lowercase().as_str(),
                    "true" | "false" | "yes" | "no" | "on" | "off" | "null" | "~"
                )
                || s.contains(": ")
                || s.contains(" #")
                || s.contains(':')
                || s.starts_with(['-', '?', ':', '&', '*', '!', '|', '>', '%', '@', '`', '"', '\''])
                || s.contains('\n')
                || s.trim() != s;
            if ambiguous {
                format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""))
            } else {
                s.clone()
            }
        }
        _ => String::new(),
    }
}

fn yaml_value(value: &Value, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    match value {
        Value::Object(map) => {
            if map.is_empty() {
                out.push_str(&format!("{pad}{{}}\n"));
            }
            for (key, child) in map {
                match child {
                    Value::Object(inner) if !inner.is_empty() => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        yaml_value(child, depth + 1, out);
                    }
                    Value::Array(items) if !items.is_empty() => {
                        out.push_str(&format!("{pad}{key}:\n"));
                        yaml_value(child, depth + 1, out);
                    }
                    _ => out.push_str(&format!("{pad}{key}: {}\n", yaml_scalar(child))),
                }
            }
        }
        Value::Array(items) => {
            if items.is_empty() {
                out.push_str(&format!("{pad}[]\n"));
            }
            for item in items {
                match item {
                    Value::Object(_) | Value::Array(_) => {
                        out.push_str(&format!("{pad}-\n"));
                        yaml_value(item, depth + 1, out);
                    }
                    _ => out.push_str(&format!("{pad}- {}\n", yaml_scalar(item))),
                }
            }
        }
        scalar => out.push_str(&format!("{pad}{}\n", yaml_scalar(scalar))),
    }
}

fn check_yaml(text: &str) -> Check {
    let mut check = Check { parses: true, partial: true, ..Default::default() };
    if text.trim().is_empty() {
        return Check::bad("the file is empty".to_string());
    }

    for (number, line) in text.lines().enumerate() {
        let line_number = number + 1;
        if line.trim().is_empty() || line.trim_start().starts_with('#') {
            continue;
        }
        if line.contains('\t') {
            // The commonest YAML failure, and one the specification forbids.
            check
                .problems
                .push(format!("line {line_number} is indented with a tab, which YAML forbids"));
            continue;
        }
        let indent = line.len() - line.trim_start().len();
        if indent % 2 != 0 {
            check.problems.push(format!(
                "line {line_number} is indented {indent} space(s); this writer uses two per level"
            ));
        }

        let body = line.trim_start();
        let item = body.strip_prefix("- ").unwrap_or(body);
        if item == "-" || item.is_empty() {
            continue;
        }
        if item.starts_with('"') || item.starts_with('[') || item.starts_with('{') {
            check.items += 1;
            continue;
        }
        if !item.contains(':') && !body.starts_with('-') {
            check.problems.push(format!(
                "line {line_number} is neither a key, a list item nor a quoted scalar: {item:?}"
            ));
        }
        check.items += 1;
    }
    check
}

// ── XML ──────────────────────────────────────────────────────────────────

/// The five predefined entities, and nothing else.
pub fn xml_escape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

/// Writes a value tree as XML under one root element.
pub fn to_xml(root: &str, value: &Value) -> String {
    let mut out = String::from("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
    xml_element(root, value, 0, &mut out);
    out
}

fn xml_name(raw: &str) -> String {
    // An XML name may not start with a digit and may not contain spaces.
    // Rewriting is safe and silent loss is not, so an unusable name becomes a
    // usable one rather than an invalid document.
    let cleaned: String = raw
        .chars()
        .map(|c| if c.is_alphanumeric() || c == '-' || c == '_' || c == '.' { c } else { '-' })
        .collect();
    if cleaned.is_empty() {
        "item".to_string()
    } else if cleaned.chars().next().is_some_and(|c| c.is_ascii_digit()) {
        format!("_{cleaned}")
    } else {
        cleaned
    }
}

fn xml_element(name: &str, value: &Value, depth: usize, out: &mut String) {
    let pad = "  ".repeat(depth);
    let name = xml_name(name);
    match value {
        Value::Object(map) => {
            out.push_str(&format!("{pad}<{name}>\n"));
            for (key, child) in map {
                xml_element(key, child, depth + 1, out);
            }
            out.push_str(&format!("{pad}</{name}>\n"));
        }
        Value::Array(items) => {
            out.push_str(&format!("{pad}<{name}>\n"));
            for item in items {
                xml_element("item", item, depth + 1, out);
            }
            out.push_str(&format!("{pad}</{name}>\n"));
        }
        Value::Null => out.push_str(&format!("{pad}<{name}/>\n")),
        scalar => {
            let text = match scalar {
                Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            out.push_str(&format!("{pad}<{name}>{}</{name}>\n", xml_escape(&text)));
        }
    }
}

fn check_xml(text: &str) -> Check {
    // The well-formedness scan is the SVG validator's, which was written for
    // exactly this subset. Reusing it keeps one parser rather than two that can
    // disagree about the same document.
    match super::svg_validate::scan_for_wellformedness(text) {
        Err(problem) => Check::bad(problem),
        Ok(elements) => {
            let mut check =
                Check { parses: true, items: elements, partial: true, ..Default::default() };
            if elements == 0 {
                check.problems.push("the document contains no elements".to_string());
            }
            // A raw `&` that is not an entity is the commonest XML defect and
            // the one that makes a file fail in another tool rather than here.
            let mut rest = text;
            while let Some(at) = rest.find('&') {
                let tail = &rest[at + 1..];
                let entity = match tail.find(';') {
                    Some(end) if end <= 8 => &tail[..end],
                    _ => "",
                };
                let known = matches!(entity, "amp" | "lt" | "gt" | "quot" | "apos")
                    || entity.starts_with('#');
                if !known {
                    check.problems.push(
                        "the document contains a bare '&' that is not an entity; it must be \
                         written &amp;"
                            .to_string(),
                    );
                    break;
                }
                rest = tail;
            }
            check
        }
    }
}

// ── Markdown ─────────────────────────────────────────────────────────────

/// Writes a [`Document`] as Markdown.
///
/// The same content model that becomes a `.docx` and a `.pdf`. Heading levels
/// are the model's own, so an outline that is right in Word is right here.
pub fn to_markdown(document: &Document) -> String {
    let mut out = format!("# {}\n\n", document.title.trim());
    if !document.classification.trim().is_empty() {
        out.push_str(&format!("*Classification: {}*\n\n", document.classification.trim()));
    }
    for section in &document.sections {
        let level = (section.level as usize + 1).min(6);
        out.push_str(&format!("{} {}\n\n", "#".repeat(level), section.heading.trim()));
        for block in &section.blocks {
            match block {
                Block::Paragraph { text } => out.push_str(&format!("{}\n\n", text.trim())),
                Block::Bullets { items } => {
                    for item in items {
                        out.push_str(&format!("- {}\n", item.trim()));
                    }
                    out.push('\n');
                }
                Block::Numbered { items } => {
                    for (index, item) in items.iter().enumerate() {
                        out.push_str(&format!("{}. {}\n", index + 1, item.trim()));
                    }
                    out.push('\n');
                }
                Block::Table { header, rows, caption } => {
                    if let Some(caption) = caption.as_deref().filter(|c| !c.trim().is_empty()) {
                        out.push_str(&format!("*{}*\n\n", caption.trim()));
                    }
                    out.push_str(&format!("| {} |\n", header.join(" | ")));
                    out.push_str(&format!(
                        "| {} |\n",
                        header.iter().map(|_| "---").collect::<Vec<_>>().join(" | ")
                    ));
                    for row in rows {
                        out.push_str(&format!("| {} |\n", row.join(" | ")));
                    }
                    out.push('\n');
                }
                Block::PageBreak => out.push_str("---\n\n"),
            }
        }
    }
    out
}

fn check_markdown(text: &str) -> Check {
    let mut check = Check { parses: true, ..Default::default() };
    if text.trim().is_empty() {
        return Check::bad("the file is empty".to_string());
    }

    let mut previous = 0usize;
    let mut in_fence = false;
    let mut fence_without_language = false;
    let lines: Vec<&str> = text.lines().collect();

    for (number, line) in lines.iter().enumerate() {
        let line_number = number + 1;
        if let Some(rest) = line.trim_start().strip_prefix("```") {
            if !in_fence && rest.trim().is_empty() {
                fence_without_language = true;
            }
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }

        if let Some(hashes) = line.split_whitespace().next().filter(|t| t.starts_with('#')) {
            if hashes.chars().all(|c| c == '#') {
                let level = hashes.len();
                check.items += 1;
                if previous > 0 && level > previous + 1 {
                    check.problems.push(format!(
                        "line {line_number}: a level {level} heading under a level {previous} \
                         one skips a level"
                    ));
                }
                previous = level;
            }
        }

        // A table's separator row has to have as many cells as its header, or
        // no reader renders it as a table at all.
        if line.trim_start().starts_with('|') && number + 1 < lines.len() {
            let next = lines[number + 1].trim();
            if next.starts_with('|') && next.contains("---") {
                let columns = line.matches('|').count();
                let separators = next.matches('|').count();
                if columns != separators {
                    check.problems.push(format!(
                        "line {line_number}: the table header has {columns} cell boundaries and \
                         its separator row has {separators}, so it will not render as a table"
                    ));
                }
            }
        }
    }

    if in_fence {
        check.problems.push("a fenced code block is never closed".to_string());
    }
    if fence_without_language {
        check
            .problems
            .push("a fenced code block has no language, so nothing can highlight it".to_string());
    }
    check
}

// ── HTML ─────────────────────────────────────────────────────────────────

/// Writes a [`Document`] as a standalone HTML page.
///
/// Semantic elements, one `h1`, a `lang`, and every piece of interpolated text
/// escaped. The escaping is not a nicety: a finding containing `<` becomes an
/// element in the reader's browser otherwise, and a document assembled from a
/// scanned drawing is exactly where such a character comes from.
pub fn to_html(document: &Document) -> String {
    let mut out = String::from(
        "<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n\
         <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n",
    );
    out.push_str(&format!("<title>{}</title>\n", xml_escape(document.title.trim())));
    out.push_str(
        "<style>body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:48rem;\
         margin:2rem auto;padding:0 1rem;line-height:1.5}\
         table{border-collapse:collapse;width:100%}\
         th,td{border:1px solid #999;padding:.4rem .6rem;text-align:left}\
         caption{text-align:left;font-style:italic;margin-bottom:.4rem}</style>\n",
    );
    out.push_str("</head>\n<body>\n<main>\n");
    out.push_str(&format!("<h1>{}</h1>\n", xml_escape(document.title.trim())));
    if !document.classification.trim().is_empty() {
        out.push_str(&format!(
            "<p><strong>Classification:</strong> {}</p>\n",
            xml_escape(document.classification.trim())
        ));
    }

    for section in &document.sections {
        let level = (section.level as usize + 1).min(6);
        out.push_str(&format!(
            "<section>\n<h{level}>{}</h{level}>\n",
            xml_escape(section.heading.trim())
        ));
        for block in &section.blocks {
            match block {
                Block::Paragraph { text } => {
                    out.push_str(&format!("<p>{}</p>\n", xml_escape(text.trim())));
                }
                Block::Bullets { items } => {
                    out.push_str("<ul>\n");
                    for item in items {
                        out.push_str(&format!("<li>{}</li>\n", xml_escape(item.trim())));
                    }
                    out.push_str("</ul>\n");
                }
                Block::Numbered { items } => {
                    out.push_str("<ol>\n");
                    for item in items {
                        out.push_str(&format!("<li>{}</li>\n", xml_escape(item.trim())));
                    }
                    out.push_str("</ol>\n");
                }
                Block::Table { header, rows, caption } => {
                    out.push_str("<table>\n");
                    if let Some(caption) = caption.as_deref().filter(|c| !c.trim().is_empty()) {
                        out.push_str(&format!("<caption>{}</caption>\n", xml_escape(caption)));
                    }
                    out.push_str("<thead>\n<tr>\n");
                    for cell in header {
                        out.push_str(&format!("<th scope=\"col\">{}</th>\n", xml_escape(cell)));
                    }
                    out.push_str("</tr>\n</thead>\n<tbody>\n");
                    for row in rows {
                        out.push_str("<tr>\n");
                        for cell in row {
                            out.push_str(&format!("<td>{}</td>\n", xml_escape(cell)));
                        }
                        out.push_str("</tr>\n");
                    }
                    out.push_str("</tbody>\n</table>\n");
                }
                Block::PageBreak => out.push_str("<hr>\n"),
            }
        }
        out.push_str("</section>\n");
    }
    out.push_str("</main>\n</body>\n</html>\n");
    out
}

fn check_html(text: &str) -> Check {
    let lowered = text.to_lowercase();
    let mut check = Check { parses: true, partial: true, ..Default::default() };

    if !lowered.contains("<html") {
        return Check::bad("the document has no <html> element".to_string());
    }
    // Accessibility rules that can be checked without rendering.
    if !lowered.contains("<html lang=") {
        check.problems.push(
            "the <html> element has no lang attribute, so a screen reader cannot choose a voice"
                .to_string(),
        );
    }
    if !lowered.contains("<title>") {
        check.problems.push("the page has no <title>".to_string());
    }
    let h1s = lowered.matches("<h1").count();
    check.items = lowered.matches("<h").count();
    if h1s == 0 {
        check.problems.push("the page has no <h1>".to_string());
    } else if h1s > 1 {
        check.problems.push(format!(
            "the page has {h1s} <h1> elements; a document has one top-level heading"
        ));
    }
    // An image with no alternative text is invisible to anyone who cannot see
    // it, and there is no way to add it after the fact.
    let mut rest = lowered.as_str();
    while let Some(at) = rest.find("<img") {
        let tail = &rest[at..];
        let end = tail.find('>').unwrap_or(tail.len());
        if !tail[..end].contains("alt=") {
            check.problems.push("an <img> has no alt attribute".to_string());
            break;
        }
        rest = &tail[end.min(tail.len())..];
    }
    // A form control with no label is one a screen reader announces as
    // "edit text, blank".
    let mut rest = lowered.as_str();
    while let Some(at) = rest.find("<input") {
        let tail = &rest[at..];
        let end = tail.find('>').unwrap_or(tail.len());
        let element = &tail[..end];
        let labelled = element.contains("aria-label")
            || element.contains("aria-labelledby")
            || (element.contains("id=") && lowered.contains("<label"));
        let exempt = element.contains("type=\"hidden\"") || element.contains("type=\"submit\"");
        if !labelled && !exempt {
            check.problems.push("an <input> has no label".to_string());
            break;
        }
        rest = &tail[end.min(tail.len())..];
    }
    check
}

/// Quality notes: a file that is valid but not something to hand over.
pub fn quality(format: Format, text: &str) -> Vec<String> {
    let mut notes = Vec::new();
    if let Some(marker) = super::doc_model::placeholder_in(text) {
        notes.push(format!("contains the placeholder {marker:?}"));
    }
    if matches!(format, Format::Markdown | Format::Html) && text.trim().len() < 80 {
        notes.push("is too short to be a finished document".to_string());
    }
    notes
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifacts::doc_model::{Properties, Section};
    use serde_json::json;

    fn document() -> Document {
        Document {
            title: "Shell thickness inspection".to_string(),
            classification: "OFFICIAL".to_string(),
            properties: Properties::default(),
            sections: vec![
                Section {
                    heading: "Findings".to_string(),
                    level: 1,
                    blocks: vec![
                        Block::Paragraph {
                            text: "One point of sixteen was below the stated minimum.".to_string(),
                        },
                        Block::Table {
                            header: vec!["Point".into(), "Measured".into()],
                            rows: vec![vec!["S-02".into(), "8.7".into()]],
                            caption: Some("Readings below the minimum".into()),
                        },
                    ],
                },
                Section {
                    heading: "Detail".to_string(),
                    level: 2,
                    blocks: vec![Block::Numbered {
                        items: vec![
                            "Re-measure S-02.".to_string(),
                            "Refer to the engineer.".to_string(),
                        ],
                    }],
                },
            ],
        }
    }

    #[test]
    fn a_format_is_recognised_from_its_extension() {
        assert_eq!(Format::of_path(Path::new("a/b.csv")), Some(Format::Csv));
        assert_eq!(Format::of_path(Path::new("a/b.YAML")), Some(Format::Yaml));
        assert_eq!(Format::of_path(Path::new("a/b.htm")), Some(Format::Html));
        assert_eq!(Format::of_path(Path::new("a/b.txt")), None);
    }

    // ── CSV ─────────────────────────────────────────────────────────────

    /// The defect that makes a CSV silently wrong rather than visibly broken.
    #[test]
    fn a_field_containing_a_comma_survives_the_round_trip() {
        let header = vec!["Tag".to_string(), "Note".to_string()];
        let rows =
            vec![vec!["PV-2201".to_string(), "Worn, replace at the next outage".to_string()]];
        let text = to_csv(&header, &rows);
        let parsed = parse_csv(&text).expect("parses");

        assert_eq!(parsed[1][1], "Worn, replace at the next outage");
        assert_eq!(parsed[1].len(), 2, "the comma must not become a column break");
        assert!(check(Format::Csv, &text).is_sound());
    }

    #[test]
    fn a_quote_inside_a_field_is_doubled_and_comes_back() {
        let text = to_csv(
            &["Note".to_string()],
            &[vec!["The tag reads \"PV-2201\" on the drawing".to_string()]],
        );
        let parsed = parse_csv(&text).expect("parses");
        assert_eq!(parsed[1][0], "The tag reads \"PV-2201\" on the drawing");
    }

    #[test]
    fn a_newline_inside_a_field_survives() {
        let text = to_csv(&["Note".to_string()], &[vec!["Line one\nLine two".to_string()]]);
        let parsed = parse_csv(&text).expect("parses");
        assert_eq!(parsed.len(), 2, "an embedded newline must not become a new record");
        assert_eq!(parsed[1][0], "Line one\nLine two");
    }

    #[test]
    fn a_ragged_csv_is_rejected() {
        let check = check(Format::Csv, "Tag,Reading\r\nPV-2201\r\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("row 2")), "{:?}", check.problems);
    }

    #[test]
    fn an_unterminated_quote_is_rejected() {
        let check = check(Format::Csv, "Tag\r\n\"never closed\r\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("never closed")));
    }

    #[test]
    fn a_byte_order_mark_is_rejected() {
        let check = check(Format::Csv, "\u{feff}Tag,Reading\r\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("byte-order mark")));
    }

    // ── JSON ────────────────────────────────────────────────────────────

    #[test]
    fn json_round_trips_and_is_checked_by_parsing() {
        let text = to_json(&json!({"tag": "PV-2201", "readings": [9.4, 8.7]}));
        assert!(check(Format::Json, &text).is_sound());
        let back: Value = serde_json::from_str(&text).expect("parses");
        assert_eq!(back["readings"][1], json!(8.7));
    }

    #[test]
    fn a_trailing_comma_is_rejected() {
        let check = check(Format::Json, "{\"a\": 1,}");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("not valid JSON")));
    }

    /// Prose contamination: the file parses, and is not what was wanted.
    #[test]
    fn a_bare_string_is_reported_as_prose_where_structure_was_wanted() {
        let check = check(Format::Json, "\"Here are the readings you asked for.\"");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("prose")), "{:?}", check.problems);
    }

    // ── YAML ────────────────────────────────────────────────────────────

    /// The classic YAML defect: a string that looks like something else.
    #[test]
    fn an_ambiguous_scalar_is_quoted() {
        let text = to_yaml(&json!({"answer": "yes", "version": "1.0", "at": "08:00"}));
        assert!(text.contains("answer: \"yes\""), "{text}");
        assert!(text.contains("version: \"1.0\""), "{text}");
        assert!(text.contains("at: \"08:00\""), "{text}");
        let check = check(Format::Yaml, &text);
        assert!(check.is_sound(), "{:?}", check.problems);
    }

    #[test]
    fn a_tab_indent_is_rejected() {
        let check = check(Format::Yaml, "root:\n\tchild: 1\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("tab")));
    }

    #[test]
    fn nested_yaml_round_trips_through_its_own_checker() {
        let text = to_yaml(&json!({
            "readings": [{"point": "S-01", "mm": 9.4}, {"point": "S-02", "mm": 8.7}],
            "unit": "Unit Four"
        }));
        let check = check(Format::Yaml, &text);
        assert!(check.is_sound(), "{text}\n{:?}", check.problems);
        assert!(check.partial, "the checker must admit it handles a subset");
    }

    // ── XML ─────────────────────────────────────────────────────────────

    #[test]
    fn xml_escapes_and_round_trips() {
        let text = to_xml("readings", &json!({"note": "thickness < 9.0 and flagged"}));
        assert!(text.contains("&lt;"), "{text}");
        let check = check(Format::Xml, &text);
        assert!(check.is_sound(), "{:?}", check.problems);
    }

    #[test]
    fn an_unclosed_element_is_rejected() {
        let check = check(Format::Xml, "<root><child></root>");
        assert!(!check.is_sound());
    }

    #[test]
    fn a_bare_ampersand_is_rejected() {
        let check = check(Format::Xml, "<root>Jones & Son</root>");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("&amp;")), "{:?}", check.problems);
    }

    #[test]
    fn a_name_that_would_be_invalid_is_made_valid() {
        let text = to_xml("root", &json!({"2nd reading": 9.4}));
        assert!(!text.contains("<2nd reading>"), "{text}");
        let check = check(Format::Xml, &text);
        assert!(check.is_sound(), "{text}\n{:?}", check.problems);
    }

    // ── Markdown ────────────────────────────────────────────────────────

    #[test]
    fn a_document_becomes_markdown_the_checker_accepts() {
        let text = to_markdown(&document());
        let check = check(Format::Markdown, &text);
        assert!(check.is_sound(), "{text}\n{:?}", check.problems);
        assert!(text.contains("| Point | Measured |"), "{text}");
        assert!(text.contains("1. Re-measure S-02."), "{text}");
    }

    #[test]
    fn a_heading_that_skips_a_level_is_rejected() {
        let check = check(Format::Markdown, "# Title\n\n### Too deep\n\nSome text.\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("skips a level")));
    }

    #[test]
    fn a_table_whose_separator_does_not_match_is_rejected() {
        let check = check(Format::Markdown, "# T\n\n| A | B |\n| --- |\n| 1 | 2 |\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("separator")), "{:?}", check.problems);
    }

    #[test]
    fn an_unclosed_code_fence_is_rejected() {
        let check = check(Format::Markdown, "# T\n\n```rust\nlet a = 1;\n");
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("never closed")));
    }

    // ── HTML ────────────────────────────────────────────────────────────

    #[test]
    fn a_document_becomes_html_the_checker_accepts() {
        let text = to_html(&document());
        let check = check(Format::Html, &text);
        assert!(check.is_sound(), "{:?}", check.problems);
        assert!(text.contains("<html lang=\"en\">"));
        assert!(text.contains("<th scope=\"col\">"), "a header cell must be scoped");
        assert!(text.contains("<caption>"), "a table must say what it is");
    }

    /// The failure escaping exists to prevent. A finding from a scanned drawing
    /// is exactly where a `<` comes from.
    #[test]
    fn text_is_escaped_rather_than_becoming_markup() {
        let mut doc = document();
        doc.sections[0].blocks = vec![Block::Paragraph {
            text: "Thickness < 9.0 mm <script>alert(1)</script>".to_string(),
        }];
        let text = to_html(&doc);
        assert!(!text.contains("<script>"), "the page must not carry live markup from content");
        assert!(text.contains("&lt;script&gt;"), "{text}");
    }

    #[test]
    fn a_page_with_no_lang_is_rejected() {
        let check = check(
            Format::Html,
            "<html><head><title>T</title></head><body><h1>H</h1></body></html>",
        );
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("lang")));
    }

    #[test]
    fn an_image_with_no_alternative_text_is_rejected() {
        let check = check(
            Format::Html,
            "<html lang=\"en\"><head><title>T</title></head><body><h1>H</h1>\
             <img src=\"a.png\"></body></html>",
        );
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("alt")));
    }

    #[test]
    fn more_than_one_top_level_heading_is_rejected() {
        let check = check(
            Format::Html,
            "<html lang=\"en\"><head><title>T</title></head><body><h1>A</h1><h1>B</h1>\
             </body></html>",
        );
        assert!(!check.is_sound());
        assert!(check.problems.iter().any(|p| p.contains("<h1>")));
    }

    // ── Quality ─────────────────────────────────────────────────────────

    #[test]
    fn a_placeholder_is_a_quality_failure_in_any_format() {
        assert!(!quality(Format::Markdown, "# T\n\nRecommendation: TBD\n").is_empty());
        assert!(!quality(Format::Json, "{\"note\": \"TODO\"}").is_empty());
    }
}
