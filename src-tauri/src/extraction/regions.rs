//! Evidence regions: the unit every extraction result is cited by.
//!
//! A region is a box on one page of one stored document, with what was read
//! there and **how** it was read. The "how" is the part this module exists to
//! keep: a value lifted from a PDF's own text layer, the same value transcribed
//! by the OCR model from pixels, and a vision model's reading of a drawing are
//! three different kinds of claim, and a reader who cannot tell them apart
//! cannot weigh them. So [`Method`] is a closed set, and
//! [`Method::VisionInference`] is never a transcription.
//!
//! ## Identity
//!
//! A region id is derived from what it is — document, page, method, box,
//! the settings it was read under — never generated. The same page read twice
//! under the same settings produces the same ids, so a citation made on
//! Tuesday still resolves on Wednesday, and a cached read is the same evidence
//! rather than a copy of it.
//!
//! ## Storage
//!
//! One JSON file per document under `<documents>/regions/<sha>.json`, written by
//! rename so a crash leaves the old file or the new one and never half of
//! either. The source bytes are never touched: every crop is a new file beside
//! the store, and the attachment the person gave is what every region points
//! back to.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// Bumped when a region's shape changes. A file of another version is read as
/// empty rather than guessed at: regions are derived data and can be read again.
pub const REGION_SCHEMA: u32 = 1;

/// Bounds one document's store, so a runaway loop cannot fill a disk.
pub const MAX_REGIONS_PER_DOCUMENT: usize = 20_000;

/// The coordinate space a box is in. Always named beside the box.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum PageSpace {
    /// PDF user space: points, origin top-left as PyMuPDF reports it.
    PdfPoints,
    /// The stored image's own pixels.
    ImagePixels,
}

impl PageSpace {
    pub fn parse(raw: &str) -> Option<Self> {
        match raw {
            "pdf-points" => Some(Self::PdfPoints),
            "image-pixels" => Some(Self::ImagePixels),
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::PdfPoints => "pdf-points",
            Self::ImagePixels => "image-pixels",
        }
    }
}

/// A box on a page, in that page's [`PageSpace`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BBox {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl BBox {
    pub fn new(x0: f64, y0: f64, x1: f64, y1: f64) -> Self {
        Self { x0, y0, x1, y1 }
    }

    /// From the sidecar's `[x0, y0, x1, y1]`.
    pub fn from_slice(values: &[f64]) -> Option<Self> {
        match values {
            [x0, y0, x1, y1] if x1 > x0 && y1 > y0 => Some(Self::new(*x0, *y0, *x1, *y1)),
            _ => None,
        }
    }

    pub fn width(&self) -> f64 {
        (self.x1 - self.x0).max(0.0)
    }

    pub fn height(&self) -> f64 {
        (self.y1 - self.y0).max(0.0)
    }

    pub fn area(&self) -> f64 {
        self.width() * self.height()
    }

    /// The overlap of two boxes, as a fraction of `self`'s area.
    pub fn covered_by(&self, other: &BBox) -> f64 {
        let x0 = self.x0.max(other.x0);
        let y0 = self.y0.max(other.y0);
        let x1 = self.x1.min(other.x1);
        let y1 = self.y1.min(other.y1);
        if x1 <= x0 || y1 <= y0 || self.area() <= 0.0 {
            return 0.0;
        }
        ((x1 - x0) * (y1 - y0)) / self.area()
    }

    /// Two decimals, the precision the sidecar reports and the ids hash.
    pub fn rounded(&self) -> Self {
        let r = |v: f64| (v * 100.0).round() / 100.0;
        Self::new(r(self.x0), r(self.y0), r(self.x1), r(self.y1))
    }

    pub fn describe(&self) -> String {
        format!(
            "[{:.1}, {:.1}, {:.1}, {:.1}]",
            self.x0, self.y0, self.x1, self.y1
        )
    }

    /// Whether this box lies inside a page of the given size, with a small
    /// tolerance for rounding. A box outside is reported, never clamped.
    pub fn inside(&self, width: f64, height: f64) -> bool {
        let slack = 0.5;
        self.x0 >= -slack && self.y0 >= -slack && self.x1 <= width + slack && self.y1 <= height + slack
    }
}

/// How the text in a region came to exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Method {
    /// The PDF's own text layer. What the author's software wrote.
    EmbeddedText,
    /// A ruled table found in the text layer, cell by cell.
    EmbeddedTable,
    /// Local Unlimited-OCR transcription of pixels.
    Ocr,
    /// A vision model's interpretation of an image. A proposal, never a
    /// transcription, and never published as an observation.
    VisionInference,
}

impl Method {
    pub fn label(self) -> &'static str {
        match self {
            Self::EmbeddedText => "embedded text layer",
            Self::EmbeddedTable => "embedded table",
            Self::Ocr => "local OCR transcription",
            Self::VisionInference => "vision-model inference (proposal, not transcription)",
        }
    }

    pub fn is_transcription(self) -> bool {
        !matches!(self, Self::VisionInference)
    }
}

/// What reading a region produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum RegionStatus {
    /// Text was read and no check objected.
    Read,
    /// Something is there and nothing could be made of it. The text is empty,
    /// and the region says so rather than disappearing.
    Unreadable,
    /// Reading stopped at the decode cap; the text is the part that fitted.
    Truncated,
    /// Reading degenerated into repetition and was cut; the text is what came
    /// before the loop.
    Looped,
    /// The output did not have the shape a region needs: a box outside the
    /// image, a box that runs backwards, text with no box at all.
    Malformed,
}

impl RegionStatus {
    pub fn label(self) -> &'static str {
        match self {
            Self::Read => "read",
            Self::Unreadable => "unreadable",
            Self::Truncated => "truncated",
            Self::Looped => "looped (cut at the repetition)",
            Self::Malformed => "malformed",
        }
    }

    /// Whether the text may be quoted as what the page says.
    pub fn quotable(self) -> bool {
        matches!(self, Self::Read)
    }
}

/// One cell of a table.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct TableCell {
    pub row: u32,
    pub col: u32,
    pub text: String,
    /// Present for an embedded table. OCR reports a box for the table and not
    /// for its cells, and inventing cell boxes by dividing the table would be a
    /// location nobody measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub bbox: Option<BBox>,
}

/// What produced a region, exactly enough to reproduce it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct Extractor {
    /// `pymupdf`, `unlimited-ocr`, or a vision model's registry id.
    pub name: String,
    /// The library version, or the OCR parser version.
    pub version: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_id: Option<String>,
    /// The registry's pinned hash of the weights, when it pins one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weights_sha256: Option<String>,
    /// The projector file and its size, as loaded.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub projector: Option<String>,
    /// The sampler settings and prompt, fingerprinted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile: Option<String>,
}

/// A box on a page, what was read there, and how.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct EvidenceRegion {
    pub region_id: String,
    pub document_sha256: String,
    pub page: u32,
    pub bbox: BBox,
    pub coord_space: PageSpace,
    /// `text`, `title`, `table`, `figure`, `footer`, `image`, `crop`… — the
    /// layout engine's or the model's own word, not a normalised taxonomy.
    pub label: String,
    pub method: Method,
    pub status: RegionStatus,
    pub text: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub cells: Vec<TableCell>,
    /// Why the status is not `read`, and anything a checker noticed.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub notes: Vec<String>,
    /// The crop the OCR or vision model was shown, when one was.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub crop_id: Option<String>,
    /// SHA-256 of that crop's PNG bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_sha256: Option<String>,
    pub extractor: Extractor,
    /// The OCR cache entry this came from, for a model read.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_key: Option<String>,
}

impl EvidenceRegion {
    /// A one-line citation a person can follow to the page.
    pub fn citation(&self) -> String {
        format!(
            "{} p.{} {} {}",
            self.region_id,
            self.page,
            self.bbox.describe(),
            self.coord_space.label()
        )
    }
}

/// Derives a region's id from what it is.
///
/// `discriminator` is whatever distinguishes two reads of the same box: the OCR
/// cache key, the layout engine's version, the vision call's own key. `ordinal`
/// separates two regions a single read reported with the same box.
pub fn region_id(
    document_sha256: &str,
    page: u32,
    method: Method,
    label: &str,
    bbox: &BBox,
    discriminator: &str,
    ordinal: usize,
) -> String {
    let b = bbox.rounded();
    let mut hasher = Sha256::new();
    hasher.update(
        format!(
            "{document_sha256}|{page}|{method:?}|{label}|{:.2},{:.2},{:.2},{:.2}|{discriminator}|{ordinal}",
            b.x0, b.y0, b.x1, b.y1
        )
        .as_bytes(),
    );
    format!("rg-{}", &hex::encode(hasher.finalize())[..16])
}

/// A rendered crop, preserved so the exact pixels a model saw can be shown.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CropRecord {
    pub crop_id: String,
    pub document_sha256: String,
    pub page: u32,
    /// The box the crop covers, clamped to the page by the renderer.
    pub bbox: BBox,
    pub coord_space: PageSpace,
    /// Pixels in the crop per unit of `coord_space`.
    pub pixels_per_unit: f64,
    pub width: u32,
    pub height: u32,
    pub image_sha256: String,
    /// The file, relative to the document's region directory.
    pub file: String,
    /// The render resolution asked for, for a PDF. `None` for an image, which
    /// is never re-sampled.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dpi: Option<u32>,
}

/// What is known about one page, whichever way it was read.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageRecord {
    pub page: u32,
    pub width: f64,
    pub height: f64,
    pub coord_space: PageSpace,
    pub rotation: i32,
    /// Characters in the text layer. Zero on a scan.
    pub text_layer_chars: u32,
    /// Fraction of the page's area under image blocks, 0..=1. A scan is one
    /// picture covering the page; a typed page with a logo is not.
    #[serde(default)]
    pub image_coverage: f64,
    /// Measured skew in degrees, when it was measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skew_degrees: Option<f64>,
    /// `projection-profile` or `unmeasured`, when a measurement was attempted.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub skew_method: Option<String>,
}

impl PageRecord {
    /// Whether the page's own text layer is the reading to use.
    ///
    /// Not a character count alone: a typed page with two short lines has a
    /// small text layer and is still a typed page, while a scanned page with a
    /// stray header string has a small text layer and is a picture. So a page
    /// with fewer than `min_chars` characters is a scan only when it has no text
    /// at all or an image covers at least half of it.
    pub fn text_layer_adequate(&self, min_chars: u32) -> bool {
        self.text_layer_chars >= min_chars
            || (self.text_layer_chars > 0 && self.image_coverage < 0.5)
    }
}

/// One document's regions, crops and page facts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct DocumentRegions {
    pub schema_version: u32,
    pub document_sha256: String,
    #[serde(default)]
    pub pages: BTreeMap<u32, PageRecord>,
    #[serde(default)]
    pub regions: BTreeMap<String, EvidenceRegion>,
    #[serde(default)]
    pub crops: BTreeMap<String, CropRecord>,
}

impl DocumentRegions {
    pub fn empty(document_sha256: &str) -> Self {
        Self {
            schema_version: REGION_SCHEMA,
            document_sha256: document_sha256.to_string(),
            pages: BTreeMap::new(),
            regions: BTreeMap::new(),
            crops: BTreeMap::new(),
        }
    }

    /// The regions on one page, in reading order (top to bottom, then left).
    pub fn on_page(&self, page: u32) -> Vec<&EvidenceRegion> {
        let mut found: Vec<&EvidenceRegion> =
            self.regions.values().filter(|r| r.page == page).collect();
        found.sort_by(|a, b| {
            a.bbox
                .y0
                .partial_cmp(&b.bbox.y0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.bbox.x0.partial_cmp(&b.bbox.x0).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.region_id.cmp(&b.region_id))
        });
        found
    }
}

/// Where the region files live.
pub struct RegionStore {
    root: PathBuf,
    /// One writer at a time. The file is read-modify-written, and two tools
    /// finishing together must not lose each other's regions.
    write: Mutex<()>,
}

impl RegionStore {
    /// `documents_root` is `<app data>/documents`.
    pub fn new(documents_root: &Path) -> Self {
        Self {
            root: documents_root.join("regions"),
            write: Mutex::new(()),
        }
    }

    /// The directory a document's crops are written into.
    pub fn document_dir(&self, sha256: &str) -> Option<PathBuf> {
        is_sha(sha256).then(|| self.root.join(sha256))
    }

    fn file(&self, sha256: &str) -> Option<PathBuf> {
        is_sha(sha256).then(|| self.root.join(format!("{sha256}.json")))
    }

    /// The document's regions, or an empty set.
    pub fn load(&self, sha256: &str) -> Result<DocumentRegions, String> {
        let path = self
            .file(sha256)
            .ok_or_else(|| "that is not a document id".to_string())?;
        let raw = match std::fs::read(&path) {
            Ok(raw) => raw,
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                return Ok(DocumentRegions::empty(sha256))
            }
            Err(error) => return Err(format!("the region store could not be read: {error}")),
        };
        match serde_json::from_slice::<DocumentRegions>(&raw) {
            Ok(held) if held.schema_version == REGION_SCHEMA && held.document_sha256 == sha256 => {
                Ok(held)
            }
            // Derived data under another schema, or for another document: read
            // again rather than trusted.
            Ok(_) => Ok(DocumentRegions::empty(sha256)),
            Err(error) => Err(format!(
                "the region store for this document is unreadable ({error}); it holds derived \
                 data only, and deleting {} lets it be rebuilt",
                path.display()
            )),
        }
    }

    /// Adds regions, crops and page facts. Existing entries with the same id
    /// are replaced — an id is what the entry is, so a replacement is the same
    /// evidence recomputed.
    pub fn merge(
        &self,
        sha256: &str,
        pages: Vec<PageRecord>,
        regions: Vec<EvidenceRegion>,
        crops: Vec<CropRecord>,
    ) -> Result<DocumentRegions, String> {
        let _guard = self
            .write
            .lock()
            .map_err(|_| "the region store lock is poisoned".to_string())?;
        let mut held = self.load(sha256)?;
        for page in pages {
            let previous_skew = held
                .pages
                .get(&page.page)
                .and_then(|p| p.skew_degrees.zip(p.skew_method.clone()));
            let mut page = page;
            if page.skew_method.is_none() {
                if let Some((degrees, method)) = previous_skew {
                    page.skew_degrees = Some(degrees);
                    page.skew_method = Some(method);
                }
            }
            held.pages.insert(page.page, page);
        }
        for region in regions {
            if held.regions.len() >= MAX_REGIONS_PER_DOCUMENT
                && !held.regions.contains_key(&region.region_id)
            {
                return Err(format!(
                    "this document already holds {MAX_REGIONS_PER_DOCUMENT} regions, the store's \
                     limit; nothing further was recorded"
                ));
            }
            held.regions.insert(region.region_id.clone(), region);
        }
        for crop in crops {
            held.crops.insert(crop.crop_id.clone(), crop);
        }
        self.write_file(&held)?;
        Ok(held)
    }

    /// Records a page's measured skew.
    pub fn record_skew(
        &self,
        sha256: &str,
        page: u32,
        degrees: Option<f64>,
        method: &str,
    ) -> Result<(), String> {
        let _guard = self
            .write
            .lock()
            .map_err(|_| "the region store lock is poisoned".to_string())?;
        let mut held = self.load(sha256)?;
        if let Some(record) = held.pages.get_mut(&page) {
            record.skew_degrees = degrees;
            record.skew_method = Some(method.to_string());
            self.write_file(&held)?;
        }
        Ok(())
    }

    fn write_file(&self, held: &DocumentRegions) -> Result<(), String> {
        let path = self
            .file(&held.document_sha256)
            .ok_or_else(|| "that is not a document id".to_string())?;
        std::fs::create_dir_all(&self.root)
            .map_err(|error| format!("the region store could not be created: {error}"))?;
        let bytes = serde_json::to_vec_pretty(held)
            .map_err(|error| format!("the regions could not be encoded: {error}"))?;
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, &bytes)
            .map_err(|error| format!("the region store could not be written: {error}"))?;
        std::fs::rename(&temporary, &path)
            .map_err(|error| format!("the region store could not be replaced: {error}"))
    }
}

/// A lower-case SHA-256 hex string, and nothing that could name another path.
pub fn is_sha(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

/// SHA-256 of a file's bytes.
pub fn sha256_file(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|error| format!("{} could not be opened: {error}", path.display()))?;
    let mut hasher = Sha256::new();
    let mut buffer = vec![0u8; 1 << 20];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("{} could not be read: {error}", path.display()))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(hex::encode(hasher.finalize()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn region(id_ordinal: usize, page: u32, y: f64) -> EvidenceRegion {
        let bbox = BBox::new(10.0, y, 100.0, y + 10.0);
        EvidenceRegion {
            region_id: region_id(SHA, page, Method::EmbeddedText, "text", &bbox, "v1", id_ordinal),
            document_sha256: SHA.to_string(),
            page,
            bbox,
            coord_space: PageSpace::PdfPoints,
            label: "text".into(),
            method: Method::EmbeddedText,
            status: RegionStatus::Read,
            text: format!("line at {y}"),
            cells: Vec::new(),
            notes: Vec::new(),
            crop_id: None,
            image_sha256: None,
            extractor: Extractor::default(),
            cache_key: None,
        }
    }

    #[test]
    fn a_region_id_is_what_the_region_is() {
        let bbox = BBox::new(1.0, 2.0, 3.0, 4.0);
        let a = region_id(SHA, 1, Method::Ocr, "text", &bbox, "key", 0);
        let again = region_id(SHA, 1, Method::Ocr, "text", &bbox, "key", 0);
        let other_method = region_id(SHA, 1, Method::EmbeddedText, "text", &bbox, "key", 0);
        let other_settings = region_id(SHA, 1, Method::Ocr, "text", &bbox, "key2", 0);
        assert_eq!(a, again);
        assert_ne!(a, other_method);
        assert_ne!(a, other_settings);
        assert!(a.starts_with("rg-") && a.len() == 19);
    }

    #[test]
    fn a_store_round_trips_and_orders_a_page_top_to_bottom() {
        let dir = tempfile::tempdir().unwrap();
        let store = RegionStore::new(dir.path());
        store
            .merge(SHA, Vec::new(), vec![region(0, 1, 50.0), region(1, 1, 10.0), region(2, 2, 5.0)], Vec::new())
            .unwrap();
        let held = store.load(SHA).unwrap();
        let page = held.on_page(1);
        assert_eq!(page.len(), 2);
        assert!(page[0].bbox.y0 < page[1].bbox.y0);
    }

    #[test]
    fn an_id_that_is_not_a_hash_reaches_no_file() {
        let dir = tempfile::tempdir().unwrap();
        let store = RegionStore::new(dir.path());
        assert!(store.load("../../etc/passwd").is_err());
        assert!(store.document_dir("ABC").is_none());
    }

    #[test]
    fn coverage_of_one_box_by_another_is_a_fraction_of_the_first() {
        let small = BBox::new(0.0, 0.0, 10.0, 10.0);
        let big = BBox::new(5.0, 0.0, 100.0, 100.0);
        assert!((small.covered_by(&big) - 0.5).abs() < 1e-9);
        assert_eq!(small.covered_by(&BBox::new(50.0, 50.0, 60.0, 60.0)), 0.0);
    }
}
