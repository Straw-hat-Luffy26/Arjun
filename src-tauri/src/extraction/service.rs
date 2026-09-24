//! The analyst's operations, over authorised documents only.
//!
//! Every entry point takes a document id — the SHA-256 of its bytes, which is
//! also its version — and the run asking. The document must have been attached
//! to that run's conversation by the signed-in person
//! ([`crate::agent_runtime::documents::DocumentStore::get`]'s check, reused
//! rather than restated), and the stored bytes must still hash to the id. A
//! document somebody else attached, or one attached to another thread, is "not
//! attached here" in the same words as one that does not exist.
//!
//! Page ranges, crops and OCR batches are bounded and a request over a bound is
//! refused rather than trimmed, so the caller always knows what it asked for.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::agent_runtime::cancellation::CancelToken;
use crate::agent_runtime::documents::DocumentStore;
use crate::ai_engine::ocr_profile::OcrDetent;

use super::fields::{self, FieldResult};
use super::ocr::{read_batch, BatchRead, HeldCard, OcrCache, OcrService, OcrUnit, MAX_OCR_UNITS_PER_CALL};
use super::regions::{
    is_sha, region_id, sha256_file, BBox, CropRecord, DocumentRegions, EvidenceRegion, Extractor,
    Method, PageRecord, PageSpace, RegionStatus, RegionStore, TableCell,
};
use super::sidecar::{LayoutAnswer, Sidecar};
use super::vision::{self, VisionService};

/// Pages one layout, table or findings call may cover.
pub const MAX_PAGES_PER_CALL: u32 = 10;
/// Crops one render call may produce.
pub const MAX_CROPS_PER_CALL: usize = 6;
/// Images one call may send for interpretation.
pub const MAX_INTERPRETATIONS_PER_CALL: usize = 2;
/// Below this many text-layer characters a page is treated as a scan.
///
/// Low on purpose: a drawing sheet whose text layer holds only its title block
/// still carries text, and the question "is the text layer adequate" is asked
/// again per crop — a crop is always read by OCR when asked for.
pub const MIN_TEXT_LAYER_CHARS: u32 = 32;
pub const DEFAULT_DPI: u32 = 200;
/// A page measured as skewed by at least this much says so beside its boxes.
pub const SKEW_NOTE_DEGREES: f64 = 1.0;

/// A document this run may read, with its bytes located and verified.
#[derive(Debug, Clone)]
pub struct AuthorisedDocument {
    pub sha256: String,
    pub name: String,
    pub kind: String,
    pub pages: u32,
    pub source: PathBuf,
    pub is_image: bool,
}

/// How one page was covered.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct PageCoverage {
    pub page: u32,
    /// `embedded-text`, `ocr`, or `unread`.
    pub route: String,
    pub methods: Vec<Method>,
    pub regions: u32,
    pub unreadable: u32,
    pub looped: bool,
    pub truncated: bool,
    pub malformed: u32,
    pub skew_degrees: Option<f64>,
    pub reason: Option<String>,
}

/// What an OCR request did.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrOutcome {
    pub document_sha256: String,
    pub batch: BatchRead,
    /// Pages not sent to OCR because their own text layer is adequate.
    pub skipped: Vec<(u32, String)>,
    pub skew: Vec<(u32, Option<f64>, String)>,
}

/// What an interpretation request did.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct InterpretOutcome {
    pub model_id: Option<String>,
    pub proposals: Vec<EvidenceRegion>,
    pub refused: Option<String>,
    pub transport: String,
}

/// A findings pass over a page range.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct FindingsReport {
    pub document_sha256: String,
    pub name: String,
    pub from_page: u32,
    pub to_page: u32,
    pub pages: u32,
    pub fields: Vec<FieldResult>,
    /// Transcribed regions that could not be quoted, with why.
    pub unreadable: Vec<EvidenceRegion>,
    pub coverage: Vec<PageCoverage>,
    pub ocr: Option<OcrOutcome>,
    pub interpretation: Option<InterpretOutcome>,
    /// Regions read cleanly on these pages, by method.
    pub observations: BTreeMap<String, u32>,
}

/// One table, from the text layer or from OCR.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TableRead {
    pub region_id: String,
    pub page: u32,
    pub bbox: BBox,
    pub coord_space: PageSpace,
    pub method: Method,
    pub status: RegionStatus,
    pub rows: u32,
    pub cols: u32,
    pub cells: Vec<TableCell>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TablesReport {
    pub document_sha256: String,
    pub name: String,
    pub from_page: u32,
    pub to_page: u32,
    pub tables: Vec<TableRead>,
    /// Scanned pages in range nobody has OCR-read yet, so their tables are
    /// unknown rather than absent.
    pub not_read: Vec<u32>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LayoutReport {
    pub document_sha256: String,
    pub name: String,
    pub pages: u32,
    pub from_page: u32,
    pub to_page: u32,
    pub page_records: Vec<PageRecord>,
    pub regions: Vec<EvidenceRegion>,
    pub coverage: Vec<PageCoverage>,
}

/// The service the tools, the worker and the admin commands share.
pub struct ExtractionService {
    documents_root: PathBuf,
    pub regions: RegionStore,
    cache: OcrCache,
    ocr: Arc<dyn OcrService>,
    vision: Arc<dyn VisionService>,
    sidecar: Result<Sidecar, String>,
    /// Source files whose hash was verified, by size and modification time.
    verified: Mutex<std::collections::HashMap<PathBuf, (u64, std::time::SystemTime)>>,
}

impl ExtractionService {
    /// `documents_root` is `<app data>/documents`: attachments are read from
    /// `attachments/<sha>/` beneath it, and regions, crops and OCR reads are
    /// kept beside them.
    pub fn new(
        documents_root: &Path,
        ocr: Arc<dyn OcrService>,
        vision: Arc<dyn VisionService>,
        sidecar: Result<Sidecar, String>,
    ) -> Self {
        Self {
            documents_root: documents_root.to_path_buf(),
            regions: RegionStore::new(documents_root),
            cache: OcrCache::new(documents_root),
            ocr,
            vision,
            sidecar,
            verified: Mutex::new(Default::default()),
        }
    }

    /// No OCR and no vision: every model-backed operation says so. Layout and
    /// embedded text still work when the rasteriser resolves.
    pub fn without_models(documents_root: &Path, why: &str) -> Self {
        Self::new(
            documents_root,
            Arc::new(super::ocr::NoOcr(why.to_string())),
            Arc::new(vision::NoVision(why.to_string())),
            Sidecar::resolve(),
        )
    }

    pub fn documents_root(&self) -> &Path {
        &self.documents_root
    }

    pub fn ocr_service(&self) -> &dyn OcrService {
        self.ocr.as_ref()
    }

    pub fn vision_service(&self) -> &dyn VisionService {
        self.vision.as_ref()
    }

    pub fn sidecar(&self) -> Result<&Sidecar, String> {
        self.sidecar.as_ref().map_err(|why| {
            format!("the page analyser (PyMuPDF) is not usable on this machine: {why}")
        })
    }

    /// Resolves and checks a document for an owner in a conversation.
    ///
    /// `documents` is the store the attachment was recorded in; its
    /// owner-and-conversation check is the authorisation, reused rather than
    /// restated.
    pub fn authorise(
        &self,
        documents: &DocumentStore,
        sha256: &str,
        owner: &str,
        conversation: &str,
    ) -> Result<AuthorisedDocument, String> {
        let not_here = || "No document with that id has been attached to this conversation.".to_string();
        if !is_sha(sha256) {
            return Err(not_here());
        }
        let stored = documents
            .get(sha256, owner, Some(conversation))
            .map_err(|error| format!("that document could not be read back: {error}"))?
            .ok_or_else(not_here)?;
        let source = self.source_of(sha256).ok_or_else(|| {
            format!(
                "{} was read when it was attached, and its original bytes are not on this machine \
                 any more, so its pages cannot be laid out or cropped. Its stored text is still \
                 available through document.read_pages.",
                stored.name
            )
        })?;
        let extension = source
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();
        let is_image = matches!(extension.as_str(), "png" | "jpg" | "jpeg");
        if extension != "pdf" && !is_image {
            return Err(format!(
                "{} is a {} file, which has no pages to lay out or crop. Read its text with \
                 document.read_pages.",
                stored.name, stored.kind
            ));
        }
        self.verify_source(sha256, &source)?;
        Ok(AuthorisedDocument {
            sha256: sha256.to_string(),
            name: stored.name,
            kind: stored.kind,
            pages: stored.pages.max(1),
            source,
            is_image,
        })
    }

    fn source_of(&self, sha256: &str) -> Option<PathBuf> {
        let dir = self.documents_root.join("attachments").join(sha256);
        let entries = std::fs::read_dir(&dir).ok()?;
        let mut found: Vec<PathBuf> = entries
            .flatten()
            .map(|entry| entry.path())
            .filter(|path| {
                path.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n.starts_with("source.") || n.starts_with("page-1."))
                    .unwrap_or(false)
            })
            .collect();
        // `source.*` is the document; `page-1.*` is an image attachment, or a
        // page rendered from a scanned document, which is second choice.
        found.sort_by_key(|path| {
            !path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n.starts_with("source."))
                .unwrap_or(false)
        });
        found.into_iter().next()
    }

    fn verify_source(&self, sha256: &str, source: &Path) -> Result<(), String> {
        let meta = std::fs::metadata(source)
            .map_err(|error| format!("the document's bytes could not be read: {error}"))?;
        let stamp = (meta.len(), meta.modified().unwrap_or(std::time::UNIX_EPOCH));
        if let Ok(held) = self.verified.lock() {
            if held.get(source) == Some(&stamp) {
                return Ok(());
            }
        }
        let actual = sha256_file(source)?;
        if actual != sha256 {
            return Err(
                "the bytes stored for this document no longer hash to its id, so they are not the \
                 version that was attached. Nothing was read."
                    .to_string(),
            );
        }
        if let Ok(mut held) = self.verified.lock() {
            held.insert(source.to_path_buf(), stamp);
        }
        Ok(())
    }

    /// Checks a page range against the bound and the document.
    pub fn range(&self, doc: &AuthorisedDocument, from: u32, to: u32) -> Result<(u32, u32), String> {
        if from == 0 {
            return Err("Pages are numbered from 1.".to_string());
        }
        if to < from {
            return Err(format!("The range {from}-{to} ends before it starts."));
        }
        let width = to - from + 1;
        if width > MAX_PAGES_PER_CALL {
            return Err(format!(
                "That is {width} pages and at most {MAX_PAGES_PER_CALL} may be covered at once. Ask \
                 for the pages you actually need."
            ));
        }
        if from > doc.pages {
            return Err(format!("{} has {} page(s); there is no page {from}.", doc.name, doc.pages));
        }
        Ok((from, to.min(doc.pages)))
    }

    // -- Layout -----------------------------------------------------------

    /// Makes sure the layout of every page in range is in the store.
    pub fn ensure_layout(
        &self,
        doc: &AuthorisedDocument,
        from: u32,
        to: u32,
    ) -> Result<DocumentRegions, String> {
        let held = self.regions.load(&doc.sha256)?;
        let missing: Vec<u32> = (from..=to).filter(|p| !held.pages.contains_key(p)).collect();
        let (Some(first), Some(last)) = (missing.first(), missing.last()) else {
            return Ok(held);
        };
        let answer = self.sidecar()?.layout(&doc.source, *first, *last)?;
        let (pages, regions) = layout_regions(&doc.sha256, &answer)?;
        self.regions.merge(&doc.sha256, pages, regions, Vec::new())
    }

    pub fn layout_map(&self, doc: &AuthorisedDocument, from: u32, to: u32) -> Result<LayoutReport, String> {
        let (from, to) = self.range(doc, from, to)?;
        let held = self.ensure_layout(doc, from, to)?;
        let mut regions = Vec::new();
        for page in from..=to {
            regions.extend(held.on_page(page).into_iter().cloned());
        }
        Ok(LayoutReport {
            document_sha256: doc.sha256.clone(),
            name: doc.name.clone(),
            pages: doc.pages,
            from_page: from,
            to_page: to,
            page_records: (from..=to).filter_map(|p| held.pages.get(&p).cloned()).collect(),
            regions,
            coverage: coverage(&held, from, to),
        })
    }

    // -- Crops ------------------------------------------------------------

    /// Renders one crop, or reuses the preserved one.
    pub fn render_crop(
        &self,
        doc: &AuthorisedDocument,
        page: u32,
        bbox: Option<BBox>,
        dpi: Option<u32>,
    ) -> Result<CropRecord, String> {
        if page == 0 || page > doc.pages {
            return Err(format!("{} has {} page(s); there is no page {page}.", doc.name, doc.pages));
        }
        let dpi = if doc.is_image {
            None
        } else {
            Some(dpi.unwrap_or(DEFAULT_DPI).clamp(72, 400))
        };
        let described = bbox
            .map(|b| {
                let b = b.rounded();
                format!("{:.2},{:.2},{:.2},{:.2}", b.x0, b.y0, b.x1, b.y1)
            })
            .unwrap_or_else(|| "page".to_string());
        let crop_id = format!(
            "cr-{}",
            &hex::encode(Sha256::digest(
                format!("{}|{page}|{described}|{dpi:?}", doc.sha256).as_bytes()
            ))[..16]
        );
        let dir = self
            .regions
            .document_dir(&doc.sha256)
            .ok_or_else(|| "that is not a document id".to_string())?
            .join("crops");
        std::fs::create_dir_all(&dir)
            .map_err(|error| format!("the crop directory could not be created: {error}"))?;
        let file = format!("crops/{crop_id}.png");
        let target = dir.join(format!("{crop_id}.png"));

        // Preserved crops are reused only when the file is still the one that
        // was recorded: a crop is evidence, and a different file under the
        // same name would be different evidence.
        let held = self.regions.load(&doc.sha256)?;
        if let Some(existing) = held.crops.get(&crop_id) {
            if target.is_file() && sha256_file(&target).ok().as_deref() == Some(existing.image_sha256.as_str()) {
                return Ok(existing.clone());
            }
        }

        let answer = self.sidecar()?.crop(&doc.source, &target, page, bbox.as_ref(), dpi.unwrap_or(DEFAULT_DPI))?;
        let space = PageSpace::parse(&answer.coord_space).ok_or_else(|| {
            format!("the renderer reported coordinates in {:?}", answer.coord_space)
        })?;
        let rendered_box = BBox::from_slice(&answer.bbox)
            .ok_or_else(|| "the renderer returned an empty region".to_string())?;
        let record = CropRecord {
            crop_id,
            document_sha256: doc.sha256.clone(),
            page: answer.page,
            bbox: rendered_box,
            coord_space: space,
            pixels_per_unit: answer.pixels_per_unit,
            width: answer.width,
            height: answer.height,
            image_sha256: sha256_file(&target)?,
            file,
            dpi,
        };
        self.regions
            .merge(&doc.sha256, Vec::new(), Vec::new(), vec![record.clone()])?;
        Ok(record)
    }

    /// Renders up to [`MAX_CROPS_PER_CALL`] crops.
    pub fn render_regions(
        &self,
        doc: &AuthorisedDocument,
        targets: &[(u32, Option<BBox>)],
        dpi: Option<u32>,
    ) -> Result<Vec<CropRecord>, String> {
        if targets.is_empty() {
            return Err("Name at least one page or region to render.".to_string());
        }
        if targets.len() > MAX_CROPS_PER_CALL {
            return Err(format!(
                "{} regions were asked for and at most {MAX_CROPS_PER_CALL} may be rendered at once.",
                targets.len()
            ));
        }
        targets
            .iter()
            .map(|(page, bbox)| self.render_crop(doc, *page, *bbox, dpi))
            .collect()
    }

    /// The path of a preserved crop.
    pub fn crop_path(&self, crop: &CropRecord) -> Option<PathBuf> {
        self.regions
            .document_dir(&crop.document_sha256)
            .map(|dir| dir.join(&crop.file))
    }

    /// Boxes for region ids already in the store.
    pub fn region_targets(&self, doc: &AuthorisedDocument, ids: &[String]) -> Result<Vec<(u32, Option<BBox>)>, String> {
        let held = self.regions.load(&doc.sha256)?;
        ids.iter()
            .map(|id| {
                held.regions
                    .get(id)
                    .map(|region| (region.page, Some(region.bbox)))
                    .ok_or_else(|| format!("{id} is not a region of this document"))
            })
            .collect()
    }

    // -- OCR --------------------------------------------------------------

    /// Reads pages or crops with local OCR.
    ///
    /// A whole page whose own text layer is adequate is not sent: the text the
    /// file carries is the better reading, and OCR of it would be a second,
    /// worse transcription of the same words. An explicit crop always is.
    #[allow(clippy::too_many_arguments)]
    pub async fn ocr(
        self: &Arc<Self>,
        doc: &AuthorisedDocument,
        targets: Vec<(u32, Option<BBox>)>,
        dpi: Option<u32>,
        detent: OcrDetent,
        held_card: Option<&HeldCard>,
        cancel: &CancelToken,
        deadline: Instant,
    ) -> Result<OcrOutcome, String> {
        if targets.is_empty() {
            return Err("Name at least one page or region to read.".to_string());
        }
        if targets.len() > MAX_OCR_UNITS_PER_CALL {
            return Err(format!(
                "{} pages or regions were asked for and at most {MAX_OCR_UNITS_PER_CALL} may be \
                 read by OCR in one call — a page is minutes of GPU time. Ask for the rest in the \
                 next call; what is read now is cached.",
                targets.len()
            ));
        }
        let mut pages: Vec<u32> = targets.iter().map(|(p, _)| *p).collect();
        pages.sort_unstable();
        pages.dedup();
        if pages.first() == Some(&0) || pages.last().map_or(false, |p| *p > doc.pages) {
            return Err(format!("{} has {} page(s).", doc.name, doc.pages));
        }
        // Each page asked for, laid out once; a page already in the store is
        // not laid out again.
        let layout = {
            let (this, doc) = (Arc::clone(self), doc.clone());
            tokio::task::spawn_blocking(move || -> Result<DocumentRegions, String> {
                let mut held = this.regions.load(&doc.sha256)?;
                for page in pages {
                    held = this.ensure_layout(&doc, page, page)?;
                }
                Ok(held)
            })
            .await
            .map_err(|error| format!("the layout pass did not finish: {error}"))??
        };

        let mut skipped = Vec::new();
        let mut units = Vec::new();
        let mut skew = Vec::new();
        for (page, bbox) in targets {
            let record = layout.pages.get(&page);
            let chars = record.map(|p| p.text_layer_chars).unwrap_or(0);
            let adequate = record.map(|p| p.text_layer_adequate(MIN_TEXT_LAYER_CHARS)).unwrap_or(false);
            if bbox.is_none() && !doc.is_image && adequate {
                skipped.push((
                    page,
                    format!(
                        "page {page} carries its own text layer ({chars} characters), which is \
                         the better reading; it was not sent to OCR. Read it with \
                         document.layout_map or document.read_pages, or ask for a region of it."
                    ),
                ));
                continue;
            }
            let (this, doc2) = (Arc::clone(self), doc.clone());
            let crop = tokio::task::spawn_blocking(move || this.render_crop(&doc2, page, bbox, dpi))
                .await
                .map_err(|error| format!("rendering did not finish: {error}"))??;
            let already = record.and_then(|p| p.skew_method.clone().map(|m| (p.skew_degrees, m)));
            if let (None, Some((degrees, method))) = (bbox, already) {
                // Measured on an earlier read of the same bytes; the image has
                // not changed, so neither has its skew.
                skew.push((page, degrees, method));
            } else if bbox.is_none() {
                let (this, doc2) = (Arc::clone(self), doc.clone());
                let measured = tokio::task::spawn_blocking(move || -> Result<(Option<f64>, String), String> {
                    let answer = this.sidecar()?.skew(&doc2.source, page)?;
                    this.regions.record_skew(&doc2.sha256, page, answer.degrees, &answer.method)?;
                    Ok((answer.degrees, answer.method))
                })
                .await
                .map_err(|error| format!("the skew measurement did not finish: {error}"))?;
                match measured {
                    Ok((degrees, method)) => skew.push((page, degrees, method)),
                    Err(why) => skew.push((page, None, format!("not measured: {why}"))),
                }
            }
            let image = self
                .crop_path(&crop)
                .ok_or_else(|| "the crop has no path".to_string())?;
            units.push(OcrUnit {
                document_sha256: doc.sha256.clone(),
                crop,
                image,
            });
        }

        let mut batch = if units.is_empty() {
            BatchRead {
                read: Vec::new(),
                unread: Vec::new(),
                identity: self.ocr.identity(detent).ok(),
                transport: self.ocr.transport().to_string(),
                residency: None,
                check: Default::default(),
            }
        } else {
            read_batch(self.ocr.as_ref(), &self.cache, detent, units, held_card, cancel, deadline).await
        };

        // Skew is a property of the image the boxes are on. Said beside each
        // box it affects rather than corrected away: the stored original is
        // what every citation points at.
        for (page, degrees, _) in &skew {
            if let Some(degrees) = degrees.filter(|d| d.abs() >= SKEW_NOTE_DEGREES) {
                for unit in batch.read.iter_mut().filter(|u| u.page == *page) {
                    unit.notes.push(format!(
                        "page {page} is skewed by about {degrees:.2} degrees; the boxes are \
                         axis-aligned on the skewed image, so a box may clip a sloping line"
                    ));
                    for region in &mut unit.regions {
                        region.notes.push(format!("page measured as skewed {degrees:.2} degrees"));
                    }
                }
            }
        }

        let regions: Vec<EvidenceRegion> = batch.read.iter().flat_map(|u| u.regions.clone()).collect();
        if !regions.is_empty() {
            self.regions.merge(&doc.sha256, Vec::new(), regions, Vec::new())?;
        }
        Ok(OcrOutcome {
            document_sha256: doc.sha256.clone(),
            batch,
            skipped,
            skew,
        })
    }

    // -- Interpretation ----------------------------------------------------

    /// Asks a vision-ready model about up to [`MAX_INTERPRETATIONS_PER_CALL`]
    /// crops. Every answer is stored and returned as a proposal.
    pub async fn interpret(
        self: &Arc<Self>,
        doc: &AuthorisedDocument,
        targets: Vec<(u32, Option<BBox>)>,
        question: &str,
        wanted: &[String],
        deadline: Instant,
    ) -> Result<InterpretOutcome, String> {
        let transport = self.vision.transport().to_string();
        if targets.len() > MAX_INTERPRETATIONS_PER_CALL {
            return Err(format!(
                "at most {MAX_INTERPRETATIONS_PER_CALL} images may be interpreted in one call"
            ));
        }
        let ready = match self.vision.ready_model() {
            Ok(ready) => ready,
            Err(why) => {
                return Ok(InterpretOutcome {
                    model_id: None,
                    proposals: Vec::new(),
                    refused: Some(why),
                    transport,
                })
            }
        };
        let wait = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(30));
        let session = match self.vision.open(&ready.model_id, wait).await {
            Ok(session) => session,
            Err(why) => {
                return Ok(InterpretOutcome {
                    model_id: Some(ready.model_id),
                    proposals: Vec::new(),
                    refused: Some(why),
                    transport,
                })
            }
        };
        let prompt = vision::interpretation_prompt(question, wanted);
        let mut proposals = Vec::new();
        let mut refused = None;
        for (page, bbox) in targets {
            let (this, doc2) = (Arc::clone(self), doc.clone());
            let crop = tokio::task::spawn_blocking(move || this.render_crop(&doc2, page, bbox, None))
                .await
                .map_err(|error| format!("rendering did not finish: {error}"))??;
            let Some(image) = self.crop_path(&crop) else { continue };
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining < Duration::from_secs(5) {
                refused = Some("this call's time limit was reached before every image was asked about".to_string());
                break;
            }
            match vision::image_call(&session.endpoint, &image, &prompt, 512, remaining).await {
                Ok(answer) => proposals.push(vision::proposal_region(&crop, &ready, &prompt, &answer)),
                Err(why) => {
                    refused = Some(why);
                    break;
                }
            }
        }
        if !proposals.is_empty() {
            self.regions
                .merge(&doc.sha256, Vec::new(), proposals.clone(), Vec::new())?;
        }
        Ok(InterpretOutcome {
            model_id: Some(ready.model_id),
            proposals,
            refused,
            transport,
        })
    }

    // -- Tables -------------------------------------------------------------

    pub fn tables(&self, doc: &AuthorisedDocument, from: u32, to: u32) -> Result<TablesReport, String> {
        let (from, to) = self.range(doc, from, to)?;
        let held = self.ensure_layout(doc, from, to)?;
        let mut tables = Vec::new();
        let mut not_read = Vec::new();
        for page in from..=to {
            let on_page = held.on_page(page);
            for region in on_page.iter().filter(|r| r.label.eq_ignore_ascii_case("table") && r.method.is_transcription()) {
                tables.push(TableRead {
                    region_id: region.region_id.clone(),
                    page,
                    bbox: region.bbox,
                    coord_space: region.coord_space,
                    method: region.method,
                    status: region.status,
                    rows: region.cells.iter().map(|c| c.row + 1).max().unwrap_or(0),
                    cols: region.cells.iter().map(|c| c.col + 1).max().unwrap_or(0),
                    cells: region.cells.clone(),
                    notes: region.notes.clone(),
                });
            }
            let scan = held
                .pages
                .get(&page)
                .map(|p| doc.is_image || !p.text_layer_adequate(MIN_TEXT_LAYER_CHARS))
                .unwrap_or(true);
            if scan && !on_page.iter().any(|r| r.method == Method::Ocr) {
                not_read.push(page);
            }
        }
        Ok(TablesReport {
            document_sha256: doc.sha256.clone(),
            name: doc.name.clone(),
            from_page: from,
            to_page: to,
            tables,
            not_read,
        })
    }

    // -- Findings -----------------------------------------------------------

    /// Requested fields over a page range, OCR-reading scanned pages first
    /// (bounded), then an optional interpretation labelled as a proposal.
    #[allow(clippy::too_many_arguments)]
    pub async fn findings(
        self: &Arc<Self>,
        doc: &AuthorisedDocument,
        from: u32,
        to: u32,
        wanted: &[String],
        question: Option<&str>,
        interpret_regions: &[String],
        detent: OcrDetent,
        held_card: Option<&HeldCard>,
        cancel: &CancelToken,
        deadline: Instant,
    ) -> Result<FindingsReport, String> {
        let (from, to) = self.range(doc, from, to)?;
        let wanted = fields::requested(wanted)?;
        let layout = {
            let (this, doc2) = (Arc::clone(self), doc.clone());
            tokio::task::spawn_blocking(move || this.ensure_layout(&doc2, from, to))
                .await
                .map_err(|error| format!("the layout pass did not finish: {error}"))??
        };

        // Scanned pages nobody has OCR-read, in page order, bounded.
        let needing: Vec<u32> = (from..=to)
            .filter(|page| {
                let scan = layout
                    .pages
                    .get(page)
                    .map(|p| doc.is_image || !p.text_layer_adequate(MIN_TEXT_LAYER_CHARS))
                    .unwrap_or(true);
                scan && !layout
                    .regions
                    .values()
                    .any(|r| r.page == *page && r.method == Method::Ocr)
            })
            .collect();
        let ocr = if needing.is_empty() {
            None
        } else {
            let batch: Vec<(u32, Option<BBox>)> = needing
                .iter()
                .take(MAX_OCR_UNITS_PER_CALL)
                .map(|p| (*p, None))
                .collect();
            let mut outcome = self
                .ocr(doc, batch, None, detent, held_card, cancel, deadline)
                .await?;
            for page in needing.iter().skip(MAX_OCR_UNITS_PER_CALL) {
                outcome.batch.unread.push(super::ocr::UnitUnread {
                    page: *page,
                    crop_id: String::new(),
                    reason: format!(
                        "at most {MAX_OCR_UNITS_PER_CALL} scanned pages are read per call; ask \
                         for this page in the next call"
                    ),
                });
            }
            Some(outcome)
        };

        let held = self.regions.load(&doc.sha256)?;
        let on_pages: Vec<&EvidenceRegion> = (from..=to).flat_map(|p| held.on_page(p)).collect();
        let found = fields::find(&wanted, &on_pages);
        let unreadable: Vec<EvidenceRegion> = on_pages
            .iter()
            .filter(|r| r.method.is_transcription() && r.status != RegionStatus::Read)
            .filter(|r| !superseded_by_ocr(&held, r))
            .map(|r| (*r).clone())
            .collect();
        let mut observations = BTreeMap::new();
        for region in on_pages
            .iter()
            .filter(|r| r.method.is_transcription() && r.status == RegionStatus::Read && !r.text.is_empty())
        {
            *observations
                .entry(region.method.label().to_string())
                .or_insert(0u32) += 1;
        }

        let interpretation = match question.map(str::trim).filter(|q| !q.is_empty()) {
            None => None,
            Some(question) => {
                let targets = if interpret_regions.is_empty() {
                    vec![(from, None)]
                } else {
                    self.region_targets(doc, interpret_regions)?
                };
                Some(self.interpret(doc, targets, question, &wanted, deadline).await?)
            }
        };

        let mut cover = coverage(&self.regions.load(&doc.sha256)?, from, to);
        if let Some(ocr) = &ocr {
            for unread in &ocr.batch.unread {
                if let Some(page) = cover.iter_mut().find(|c| c.page == unread.page) {
                    if page.route == "unread" {
                        page.reason = Some(unread.reason.clone());
                    }
                }
            }
        }

        Ok(FindingsReport {
            document_sha256: doc.sha256.clone(),
            name: doc.name.clone(),
            from_page: from,
            to_page: to,
            pages: doc.pages,
            fields: found,
            unreadable,
            coverage: cover,
            ocr,
            interpretation,
            observations,
        })
    }

    /// How each page in range was read, from what the store holds.
    pub fn coverage_of(&self, sha256: &str, from: u32, to: u32) -> Vec<PageCoverage> {
        self.regions
            .load(sha256)
            .map(|held| coverage(&held, from, to))
            .unwrap_or_default()
    }
}

/// Page records and embedded regions from a layout answer.
pub fn layout_regions(
    sha256: &str,
    answer: &LayoutAnswer,
) -> Result<(Vec<PageRecord>, Vec<EvidenceRegion>), String> {
    let space = answer.space()?;
    let extractor = Extractor {
        name: "pymupdf".to_string(),
        version: answer.version.clone(),
        ..Default::default()
    };
    let discriminator = format!("pymupdf {}", answer.version);
    let mut pages = Vec::new();
    let mut regions = Vec::new();
    for page in &answer.layout {
        let page_area = (page.width * page.height).max(f64::EPSILON);
        let page_box = BBox::new(0.0, 0.0, page.width, page.height);
        let image_area: f64 = page
            .blocks
            .iter()
            .filter(|block| block.kind != "text")
            .filter_map(|block| BBox::from_slice(&block.bbox))
            .map(|bbox| bbox.area() * bbox.covered_by(&page_box))
            .sum();
        pages.push(PageRecord {
            page: page.page,
            width: page.width,
            height: page.height,
            coord_space: space,
            rotation: page.rotation,
            text_layer_chars: page.text_chars,
            image_coverage: (image_area / page_area).clamp(0.0, 1.0),
            skew_degrees: None,
            skew_method: None,
        });
        for (ordinal, block) in page.blocks.iter().enumerate() {
            let Some(bbox) = BBox::from_slice(&block.bbox) else { continue };
            let is_text = block.kind == "text";
            let mut notes = Vec::new();
            if !bbox.inside(page.width, page.height) {
                notes.push("the layout engine placed this block partly off the page".to_string());
            }
            if !is_text {
                notes.push(
                    "an image on the page: its content is not in the text layer. Read it with \
                     document.ocr_regions, or ask about it with media.extract_findings"
                        .to_string(),
                );
            }
            regions.push(EvidenceRegion {
                region_id: region_id(sha256, page.page, Method::EmbeddedText, &block.kind, &bbox, &discriminator, ordinal),
                document_sha256: sha256.to_string(),
                page: page.page,
                bbox,
                coord_space: space,
                label: block.kind.clone(),
                method: Method::EmbeddedText,
                status: if is_text { RegionStatus::Read } else { RegionStatus::Unreadable },
                text: block.text.clone(),
                cells: Vec::new(),
                notes,
                crop_id: None,
                image_sha256: None,
                extractor: extractor.clone(),
                cache_key: None,
            });
        }
        for (ordinal, table) in page.tables.iter().enumerate() {
            let Some(bbox) = BBox::from_slice(&table.bbox) else { continue };
            let cells: Vec<TableCell> = table
                .cells
                .iter()
                .map(|cell| TableCell {
                    row: cell.row,
                    col: cell.col,
                    text: cell.text.clone(),
                    bbox: cell.bbox.as_deref().and_then(BBox::from_slice),
                })
                .collect();
            let mut text = String::new();
            for row in 0..table.rows {
                let line: Vec<&str> = cells
                    .iter()
                    .filter(|c| c.row == row)
                    .map(|c| c.text.as_str())
                    .collect();
                text.push_str(&line.join(" | "));
                text.push('\n');
            }
            regions.push(EvidenceRegion {
                region_id: region_id(sha256, page.page, Method::EmbeddedTable, "table", &bbox, &discriminator, ordinal),
                document_sha256: sha256.to_string(),
                page: page.page,
                bbox,
                coord_space: space,
                label: "table".to_string(),
                method: Method::EmbeddedTable,
                status: RegionStatus::Read,
                text: text.trim_end().to_string(),
                cells,
                notes: Vec::new(),
                crop_id: None,
                image_sha256: None,
                extractor: extractor.clone(),
                cache_key: None,
            });
        }
    }
    Ok((pages, regions))
}

/// Whether a layout placeholder for a picture has since been read by OCR.
///
/// The layout pass marks an image block unreadable because its content is not
/// in the text layer. Once an OCR crop covering it has been read, the block is
/// no longer a region nobody could read — the OCR regions are its reading, with
/// their own statuses — and listing it as unreadable would tell a reader the
/// page could not be read when it just was.
pub fn superseded_by_ocr(held: &DocumentRegions, region: &EvidenceRegion) -> bool {
    region.method == Method::EmbeddedText
        && region.label != "text"
        && held
            .regions
            .values()
            .filter(|r| r.page == region.page && r.method == Method::Ocr)
            .filter_map(|r| r.crop_id.as_ref())
            .filter_map(|id| held.crops.get(id))
            .any(|crop| region.bbox.covered_by(&crop.bbox) >= 0.9)
}

/// How each page was covered, from the store alone.
pub fn coverage(held: &DocumentRegions, from: u32, to: u32) -> Vec<PageCoverage> {
    (from..=to)
        .map(|page| {
            let record = held.pages.get(&page);
            let on_page = held.on_page(page);
            let transcribed: Vec<&&EvidenceRegion> =
                on_page.iter().filter(|r| r.method.is_transcription()).collect();
            let text_layer = record.map(|r| r.text_layer_adequate(MIN_TEXT_LAYER_CHARS)).unwrap_or(false);
            let ocr = transcribed.iter().any(|r| r.method == Method::Ocr);
            let mut methods: Vec<Method> = transcribed
                .iter()
                .filter(|r| r.status == RegionStatus::Read && !r.text.is_empty())
                .map(|r| r.method)
                .collect();
            methods.sort();
            methods.dedup();
            let (route, reason) = if record.is_none() {
                ("unread".to_string(), Some("the page has not been laid out".to_string()))
            } else if text_layer {
                ("embedded-text".to_string(), None)
            } else if ocr {
                ("ocr".to_string(), None)
            } else {
                (
                    "unread".to_string(),
                    Some("a scanned page with no text layer that has not been OCR-read".to_string()),
                )
            };
            PageCoverage {
                page,
                route,
                methods,
                regions: transcribed.len() as u32,
                unreadable: transcribed.iter().filter(|r| r.status == RegionStatus::Unreadable && r.method == Method::Ocr).count() as u32,
                looped: transcribed.iter().any(|r| r.status == RegionStatus::Looped),
                truncated: transcribed.iter().any(|r| r.status == RegionStatus::Truncated),
                malformed: transcribed.iter().filter(|r| r.status == RegionStatus::Malformed).count() as u32,
                skew_degrees: record.and_then(|r| r.skew_degrees),
                reason,
            }
        })
        .collect()
}
