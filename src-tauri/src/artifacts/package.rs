//! What a file *is*, and whether it is safe to open, before anything opens it.
//!
//! ## Why not the extension
//!
//! A `.docx` is whatever bytes somebody put under that name. The model chose
//! the name; a file attached from outside chose its own. So the format is read
//! from the bytes — `%PDF-`, a ZIP whose `[Content_Types].xml` names a Word,
//! PowerPoint or Excel main part, an `<svg` root, UTF-8 text — and a file whose
//! bytes disagree with its declared kind is a wrong-type failure, not a pass.
//!
//! ## Why not "the ZIP opened"
//!
//! Every Office file is a ZIP, and so is a ZIP bomb. Opening one proves the
//! central directory parses and nothing else. Before any part is read this
//! checks, against [`Limits`]:
//!
//! - the archive size, the number of entries, each entry's declared
//!   uncompressed size, their total, and the compression ratio of large
//!   entries — the zip-bomb shapes;
//! - every entry name — absolute paths, `..`, backslashes, drive letters and
//!   NULs are the zip-slip shapes, and a package carrying one is refused
//!   whole, never partially extracted;
//! - duplicate names, which two readers can resolve differently.
//!
//! ## What makes a file unsafe to *render*
//!
//! Rendering hands the file to a real office suite. Three things in a package
//! can make that suite do something beyond drawing it, and each is found here
//! and refused before a renderer is started:
//!
//! - **macros** — a `vbaProject.bin`, or a macro-enabled main content type;
//! - **external relationships** — `TargetMode="External"` on an image, an OLE
//!   object, an attached template, a frame or an external workbook link: each
//!   is a fetch. A hyperlink is listed but not refused, because a renderer
//!   draws it and does not follow it;
//! - **remote field codes** in Word — `INCLUDEPICTURE`, `INCLUDETEXT`, `LINK`,
//!   `DDE`, `DDEAUTO`, `IMPORT` — which pull content in when fields update.

use std::io::{Cursor, Read};

use serde::{Deserialize, Serialize};

use super::xml_events::{self, Event};

/// What the bytes are.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum DetectedFormat {
    Docx,
    Pptx,
    Xlsx,
    Pdf,
    Svg,
    /// UTF-8 text with no NUL bytes.
    Text,
    /// A ZIP that is not an Office package this product reads.
    OtherZip,
    /// The legacy binary Office container (`.doc`, `.xls`, `.ppt`).
    LegacyOffice,
    Unknown,
}

impl DetectedFormat {
    pub const fn label(self) -> &'static str {
        match self {
            DetectedFormat::Docx => "Word document (OOXML)",
            DetectedFormat::Pptx => "PowerPoint presentation (OOXML)",
            DetectedFormat::Xlsx => "Excel workbook (OOXML)",
            DetectedFormat::Pdf => "PDF",
            DetectedFormat::Svg => "SVG drawing",
            DetectedFormat::Text => "text",
            DetectedFormat::OtherZip => "ZIP archive that is not an Office document",
            DetectedFormat::LegacyOffice => "legacy binary Office file",
            DetectedFormat::Unknown => "unrecognised bytes",
        }
    }

    /// The media type the bytes justify, not the one the name suggests.
    pub const fn mime(self) -> &'static str {
        match self {
            DetectedFormat::Docx => {
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            DetectedFormat::Pptx => {
                "application/vnd.openxmlformats-officedocument.presentationml.presentation"
            }
            DetectedFormat::Xlsx => {
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            }
            DetectedFormat::Pdf => "application/pdf",
            DetectedFormat::Svg => "image/svg+xml",
            DetectedFormat::Text => "text/plain",
            DetectedFormat::OtherZip => "application/zip",
            DetectedFormat::LegacyOffice => "application/x-ole-storage",
            DetectedFormat::Unknown => "application/octet-stream",
        }
    }

    pub const fn is_office(self) -> bool {
        matches!(self, DetectedFormat::Docx | DetectedFormat::Pptx | DetectedFormat::Xlsx)
    }

    /// The format a media type or extension *claims*, for comparison with what
    /// the bytes say. `None` when the claim names nothing this reads.
    pub fn claimed_by(mime: &str, filename: Option<&str>) -> Option<Self> {
        let by_mime = match mime {
            m if m.contains("wordprocessingml") => Some(DetectedFormat::Docx),
            m if m.contains("presentationml") => Some(DetectedFormat::Pptx),
            m if m.contains("spreadsheetml") => Some(DetectedFormat::Xlsx),
            "application/pdf" => Some(DetectedFormat::Pdf),
            "image/svg+xml" => Some(DetectedFormat::Svg),
            _ => None,
        };
        by_mime.or_else(|| {
            let extension = filename?.rsplit_once('.')?.1.to_ascii_lowercase();
            Some(match extension.as_str() {
                "docx" | "dotx" => DetectedFormat::Docx,
                "pptx" | "potx" => DetectedFormat::Pptx,
                "xlsx" | "xltx" => DetectedFormat::Xlsx,
                "pdf" => DetectedFormat::Pdf,
                "svg" => DetectedFormat::Svg,
                _ => return None,
            })
        })
    }
}

/// The ceilings a package is held to before any part is read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Limits {
    pub max_archive_bytes: u64,
    pub max_entries: usize,
    /// Per entry, decompressed. The same 50 MB [`super::ooxml`] holds a part to.
    pub max_entry_bytes: u64,
    pub max_total_bytes: u64,
    /// Decompressed over compressed, for entries above one megabyte.
    pub max_ratio: u64,
}

/// The limits every artifact tool uses.
///
/// 64 MiB of archive matches `artifact.verify_docx`'s input ceiling; 4096
/// entries is several times a large deck with every slide illustrated; a
/// hundred to one is far past what XML compresses to and well short of what a
/// bomb needs.
pub const LIMITS: Limits = Limits {
    max_archive_bytes: 64 * 1024 * 1024,
    max_entries: 4096,
    max_entry_bytes: 50 * 1024 * 1024,
    max_total_bytes: 512 * 1024 * 1024,
    max_ratio: 100,
};

/// A relationship that points outside the package.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExternalReference {
    /// The `.rels` part that declares it.
    pub part: String,
    /// The last segment of the relationship type — `image`, `hyperlink`, …
    pub kind: String,
    pub target: String,
    /// Whether a renderer would fetch it. A hyperlink is drawn, not followed.
    pub fetched_on_render: bool,
}

/// What inspecting a file's bytes found.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PackageReport {
    pub detected: DetectedFormat,
    pub bytes: u64,
    pub entries: usize,
    pub uncompressed_bytes: u64,
    /// Fatal: the file must not be opened by anything.
    pub problems: Vec<String>,
    pub macros: Vec<String>,
    pub external_references: Vec<ExternalReference>,
    pub remote_fields: Vec<String>,
    /// Embedded OLE objects and ActiveX parts. Listed, not refused: a renderer
    /// draws their preview image and runs nothing.
    pub embedded_objects: Vec<String>,
    /// Every part, in archive order.
    pub parts: Vec<String>,
    /// Whether the main content type is a template (`.dotx`, `.potx`, `.xltx`).
    pub is_template: bool,
}

impl PackageReport {
    /// Nothing about the container stops a reader opening it.
    pub fn safe_to_open(&self) -> bool {
        self.problems.is_empty()
    }

    /// Why a renderer must not be handed this, if anything.
    pub fn render_refusals(&self) -> Vec<String> {
        let mut out = self.problems.clone();
        for part in &self.macros {
            out.push(format!("it carries macros ({part})"));
        }
        for reference in self.external_references.iter().filter(|r| r.fetched_on_render) {
            out.push(format!(
                "{} declares an external {} at {}, which a renderer would fetch",
                reference.part, reference.kind, reference.target
            ));
        }
        for field in &self.remote_fields {
            out.push(format!("it carries the remote field code {field}"));
        }
        out
    }
}

/// Reads the format from the bytes alone.
pub fn sniff(bytes: &[u8]) -> DetectedFormat {
    if bytes.starts_with(b"%PDF-") {
        return DetectedFormat::Pdf;
    }
    if bytes.starts_with(&[0xD0, 0xCF, 0x11, 0xE0, 0xA1, 0xB1, 0x1A, 0xE1]) {
        return DetectedFormat::LegacyOffice;
    }
    if bytes.starts_with(b"PK\x03\x04") || bytes.starts_with(b"PK\x05\x06") {
        return inspect(bytes, &LIMITS).detected;
    }
    match std::str::from_utf8(bytes) {
        Ok(text) if !text.contains('\0') => {
            if first_element_is_svg(text) {
                DetectedFormat::Svg
            } else {
                DetectedFormat::Text
            }
        }
        _ => DetectedFormat::Unknown,
    }
}

fn first_element_is_svg(text: &str) -> bool {
    let mut rest = text.trim_start_matches('\u{feff}').trim_start();
    loop {
        if let Some(after) = rest.strip_prefix("<?") {
            match after.find("?>") {
                Some(end) => rest = after[end + 2..].trim_start(),
                None => return false,
            }
        } else if let Some(after) = rest.strip_prefix("<!--") {
            match after.find("-->") {
                Some(end) => rest = after[end + 3..].trim_start(),
                None => return false,
            }
        } else {
            return rest.starts_with("<svg");
        }
    }
}

/// Main-part content types, and what each makes the package.
const MAIN_TYPES: &[(&str, DetectedFormat, bool, bool)] = &[
    // (content type, format, macro-enabled, template)
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml", DetectedFormat::Docx, false, false),
    ("application/vnd.openxmlformats-officedocument.wordprocessingml.template.main+xml", DetectedFormat::Docx, false, true),
    ("application/vnd.ms-word.document.macroEnabled.main+xml", DetectedFormat::Docx, true, false),
    ("application/vnd.ms-word.template.macroEnabledTemplate.main+xml", DetectedFormat::Docx, true, true),
    ("application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml", DetectedFormat::Pptx, false, false),
    ("application/vnd.openxmlformats-officedocument.presentationml.template.main+xml", DetectedFormat::Pptx, false, true),
    ("application/vnd.ms-powerpoint.presentation.macroEnabled.main+xml", DetectedFormat::Pptx, true, false),
    ("application/vnd.ms-powerpoint.template.macroEnabled.main+xml", DetectedFormat::Pptx, true, true),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml", DetectedFormat::Xlsx, false, false),
    ("application/vnd.openxmlformats-officedocument.spreadsheetml.template.main+xml", DetectedFormat::Xlsx, false, true),
    ("application/vnd.ms-excel.sheet.macroEnabled.main+xml", DetectedFormat::Xlsx, true, false),
    ("application/vnd.ms-excel.template.macroEnabled.main+xml", DetectedFormat::Xlsx, true, true),
];

/// Field codes that pull content from elsewhere when fields update.
const REMOTE_FIELDS: &[&str] = &["INCLUDEPICTURE", "INCLUDETEXT", "LINK", "DDEAUTO", "DDE", "IMPORT"];

/// Checks a ZIP-based file against `limits`, and reports what it is.
pub fn inspect(bytes: &[u8], limits: &Limits) -> PackageReport {
    let mut report = PackageReport {
        detected: DetectedFormat::Unknown,
        bytes: bytes.len() as u64,
        entries: 0,
        uncompressed_bytes: 0,
        problems: Vec::new(),
        macros: Vec::new(),
        external_references: Vec::new(),
        remote_fields: Vec::new(),
        embedded_objects: Vec::new(),
        parts: Vec::new(),
        is_template: false,
    };

    if report.bytes > limits.max_archive_bytes {
        report.problems.push(format!(
            "the file is {} bytes, above the {}-byte limit for a package",
            report.bytes, limits.max_archive_bytes
        ));
        return report;
    }

    let mut archive = match zip::ZipArchive::new(Cursor::new(bytes)) {
        Ok(archive) => archive,
        Err(error) => {
            report.problems.push(format!("it is not a readable ZIP package: {error}"));
            return report;
        }
    };
    report.entries = archive.len();
    if report.entries > limits.max_entries {
        report.problems.push(format!(
            "it holds {} entries, above the limit of {}",
            report.entries, limits.max_entries
        ));
        return report;
    }

    let mut seen = std::collections::BTreeSet::new();
    for index in 0..archive.len() {
        let entry = match archive.by_index_raw(index) {
            Ok(entry) => entry,
            Err(error) => {
                report.problems.push(format!("entry {index} cannot be read: {error}"));
                continue;
            }
        };
        let name = String::from_utf8_lossy(entry.name_raw()).to_string();
        if let Some(why) = unsafe_name(&name) {
            report.problems.push(format!("the entry {name:?} {why}"));
        }
        if !seen.insert(name.to_ascii_lowercase()) {
            report.problems.push(format!("the entry {name:?} appears twice"));
        }
        let size = entry.size();
        let compressed = entry.compressed_size().max(1);
        if size > limits.max_entry_bytes {
            report.problems.push(format!(
                "the entry {name:?} expands to {size} bytes, above the {}-byte limit",
                limits.max_entry_bytes
            ));
        }
        if size > 1024 * 1024 && size / compressed > limits.max_ratio {
            report.problems.push(format!(
                "the entry {name:?} expands {}-fold, the shape of a decompression bomb",
                size / compressed
            ));
        }
        report.uncompressed_bytes = report.uncompressed_bytes.saturating_add(size);
        if !entry.is_dir() {
            report.parts.push(name);
        }
    }
    if report.uncompressed_bytes > limits.max_total_bytes {
        report.problems.push(format!(
            "it expands to {} bytes in total, above the {}-byte limit",
            report.uncompressed_bytes, limits.max_total_bytes
        ));
    }
    if !report.problems.is_empty() {
        // Nothing inside a package that failed these is read.
        return report;
    }

    let Some(types) = read_bounded(&mut archive, "[Content_Types].xml", limits.max_entry_bytes) else {
        report.detected = DetectedFormat::OtherZip;
        return report;
    };
    let types = String::from_utf8_lossy(&types).to_string();
    for (content_type, format, macros, template) in MAIN_TYPES {
        if types.contains(content_type) {
            report.detected = *format;
            report.is_template = *template;
            if *macros {
                report.macros.push(format!("main part declared {content_type}"));
            }
            break;
        }
    }
    if report.detected == DetectedFormat::Unknown {
        report.detected = DetectedFormat::OtherZip;
        return report;
    }
    if types.contains("application/vnd.ms-office.vbaProject") {
        report.macros.push("a VBA project content type".to_string());
    }

    let parts = report.parts.clone();
    for part in &parts {
        let lower = part.to_ascii_lowercase();
        if lower.ends_with("vbaproject.bin") || lower.ends_with("vbadata.xml") {
            report.macros.push(part.clone());
        }
        if lower.contains("/activex/") || lower.contains("/embeddings/") {
            report.embedded_objects.push(part.clone());
        }
        if lower.ends_with(".rels") {
            if let Some(body) = read_bounded(&mut archive, part, limits.max_entry_bytes) {
                external_references(part, &String::from_utf8_lossy(&body), &mut report.external_references);
            }
        }
        if report.detected == DetectedFormat::Docx
            && lower.starts_with("word/")
            && lower.ends_with(".xml")
            && !lower.contains("/_rels/")
        {
            if let Some(body) = read_bounded(&mut archive, part, limits.max_entry_bytes) {
                for field in remote_fields(&String::from_utf8_lossy(&body)) {
                    let named = format!("{field} in {part}");
                    if !report.remote_fields.contains(&named) {
                        report.remote_fields.push(named);
                    }
                }
            }
        }
    }
    report
}

/// Why an entry name is unsafe to extract or resolve, if it is.
fn unsafe_name(name: &str) -> Option<&'static str> {
    if name.is_empty() {
        return Some("has no name");
    }
    if name.contains('\0') {
        return Some("contains a NUL byte");
    }
    if name.contains('\\') {
        return Some("uses a backslash, which some readers treat as a directory separator");
    }
    if name.starts_with('/') {
        return Some("is an absolute path");
    }
    if name.as_bytes().get(1) == Some(&b':') {
        return Some("names a drive");
    }
    if name.split('/').any(|segment| segment == "..") {
        return Some("climbs out of the package with '..'");
    }
    None
}

/// Reads one entry, refusing anything past `limit` whatever its header says.
pub fn read_bounded<R: Read + std::io::Seek>(
    archive: &mut zip::ZipArchive<R>,
    name: &str,
    limit: u64,
) -> Option<Vec<u8>> {
    let entry = archive.by_name(name).ok()?;
    if entry.size() > limit {
        return None;
    }
    let mut out = Vec::with_capacity(entry.size().min(1024 * 1024) as usize);
    let read = entry.take(limit + 1).read_to_end(&mut out).ok()?;
    (read as u64 <= limit).then_some(out)
}

/// Reads one named entry out of package bytes, bounded by [`LIMITS`].
pub fn read_part(bytes: &[u8], name: &str) -> Result<String, String> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes))
        .map_err(|error| format!("it is not a readable package: {error}"))?;
    let body = read_bounded(&mut archive, name, LIMITS.max_entry_bytes)
        .ok_or_else(|| format!("the package has no readable {name}"))?;
    String::from_utf8(body).map_err(|_| format!("{name} is not UTF-8"))
}

fn external_references(part: &str, body: &str, out: &mut Vec<ExternalReference>) {
    let Ok(events) = xml_events::scan(body) else {
        return;
    };
    for event in &events {
        if event.local_name() != Some("Relationship") || !matches!(event, Event::Start { .. }) {
            continue;
        }
        if event.attribute("TargetMode") != Some("External") {
            continue;
        }
        let kind = event
            .attribute("Type")
            .and_then(|t| t.rsplit('/').next())
            .unwrap_or("unknown")
            .to_string();
        out.push(ExternalReference {
            part: part.to_string(),
            fetched_on_render: kind != "hyperlink",
            kind,
            target: event.attribute("Target").unwrap_or_default().to_string(),
        });
    }
}

fn remote_fields(body: &str) -> Vec<&'static str> {
    let Ok(events) = xml_events::scan(body) else {
        return Vec::new();
    };
    let mut instructions = Vec::new();
    let mut in_instr = false;
    for event in &events {
        match event {
            Event::Start { .. } if event.local_name() == Some("instrText") => in_instr = true,
            Event::End { .. } if event.local_name() == Some("instrText") => in_instr = false,
            Event::Text(text) if in_instr => instructions.push(text.clone()),
            Event::Start { .. } if event.local_name() == Some("fldSimple") => {
                if let Some(instr) = event.attribute("w:instr") {
                    instructions.push(instr.to_string());
                }
            }
            _ => {}
        }
    }
    let mut found = Vec::new();
    for instruction in instructions {
        let first = instruction
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_ascii_uppercase();
        if let Some(field) = REMOTE_FIELDS.iter().find(|field| **field == first) {
            if !found.contains(field) {
                found.push(*field);
            }
        }
    }
    found
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn zip_of(entries: &[(&str, &[u8])]) -> Vec<u8> {
        let mut buffer = Cursor::new(Vec::new());
        {
            let mut zip = zip::ZipWriter::new(&mut buffer);
            let options: zip::write::FileOptions<'_, ()> = zip::write::FileOptions::default()
                .compression_method(zip::CompressionMethod::Deflated);
            for (name, body) in entries {
                zip.start_file(*name, options).unwrap();
                zip.write_all(body).unwrap();
            }
            zip.finish().unwrap();
        }
        buffer.into_inner()
    }

    const DOCX_TYPES: &[u8] = br#"<?xml version="1.0"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/></Types>"#;

    #[test]
    fn a_word_package_is_known_by_its_content_type_not_its_name() {
        let bytes = zip_of(&[("[Content_Types].xml", DOCX_TYPES), ("word/document.xml", b"<w:document/>")]);
        assert_eq!(sniff(&bytes), DetectedFormat::Docx);
        let report = inspect(&bytes, &LIMITS);
        assert!(report.safe_to_open(), "{:?}", report.problems);
        assert!(report.render_refusals().is_empty());
    }

    #[test]
    fn a_zip_that_is_not_office_is_not_called_office() {
        let bytes = zip_of(&[("readme.txt", b"hello")]);
        assert_eq!(sniff(&bytes), DetectedFormat::OtherZip);
    }

    #[test]
    fn text_pdf_svg_and_garbage_are_told_apart() {
        assert_eq!(sniff(b"%PDF-1.7\n..."), DetectedFormat::Pdf);
        assert_eq!(sniff(b"<?xml version=\"1.0\"?>\n<svg viewBox=\"0 0 1 1\"/>"), DetectedFormat::Svg);
        assert_eq!(sniff(b"plain notes"), DetectedFormat::Text);
        assert_eq!(sniff(&[0xff, 0xfe, 0x00, 0x01]), DetectedFormat::Unknown);
    }

    #[test]
    fn a_zip_slip_entry_refuses_the_whole_package() {
        let bytes = zip_of(&[("[Content_Types].xml", DOCX_TYPES), ("../../evil.xml", b"x")]);
        let report = inspect(&bytes, &LIMITS);
        assert!(!report.safe_to_open());
        assert!(report.problems.iter().any(|p| p.contains("'..'")), "{:?}", report.problems);
        let absolute = inspect(&zip_of(&[("/etc/passwd", b"x")]), &LIMITS);
        assert!(!absolute.safe_to_open());
    }

    #[test]
    fn a_decompression_bomb_is_refused_before_it_is_read() {
        let zeros = vec![0u8; 4 * 1024 * 1024];
        let bytes = zip_of(&[("[Content_Types].xml", DOCX_TYPES), ("word/document.xml", &zeros)]);
        let report = inspect(&bytes, &LIMITS);
        assert!(report.problems.iter().any(|p| p.contains("bomb")), "{:?}", report.problems);

        let tight = Limits { max_entries: 1, ..LIMITS };
        assert!(!inspect(&bytes, &tight).safe_to_open());
    }

    #[test]
    fn macros_external_fetches_and_remote_fields_block_rendering_and_hyperlinks_do_not() {
        let rels = br#"<?xml version="1.0"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="r1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/hyperlink" Target="https://example.org/" TargetMode="External"/><Relationship Id="r2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="https://tracker.example/pixel.png" TargetMode="External"/></Relationships>"#;
        let body = br#"<w:document xmlns:w="w"><w:body><w:p><w:r><w:instrText> INCLUDEPICTURE "http://x/y.png" </w:instrText></w:r></w:p></w:body></w:document>"#;
        let bytes = zip_of(&[
            ("[Content_Types].xml", DOCX_TYPES),
            ("word/document.xml", body),
            ("word/_rels/document.xml.rels", rels),
            ("word/vbaProject.bin", b"\x00\x01"),
        ]);
        let report = inspect(&bytes, &LIMITS);
        assert!(report.safe_to_open(), "the container itself is sound: {:?}", report.problems);
        assert_eq!(report.external_references.len(), 2);
        let refusals = report.render_refusals();
        assert!(refusals.iter().any(|r| r.contains("macros")), "{refusals:?}");
        assert!(refusals.iter().any(|r| r.contains("pixel.png")), "{refusals:?}");
        assert!(refusals.iter().any(|r| r.contains("INCLUDEPICTURE")), "{refusals:?}");
        assert!(!refusals.iter().any(|r| r.contains("example.org")), "a hyperlink is not fetched: {refusals:?}");
    }

    #[test]
    fn a_claimed_format_is_read_from_the_mime_first_and_the_name_second() {
        assert_eq!(
            DetectedFormat::claimed_by("application/pdf", Some("x.docx")),
            Some(DetectedFormat::Pdf)
        );
        assert_eq!(DetectedFormat::claimed_by("text/plain", Some("x.xlsx")), Some(DetectedFormat::Xlsx));
        assert_eq!(DetectedFormat::claimed_by("text/plain", Some("x.txt")), None);
    }
}
