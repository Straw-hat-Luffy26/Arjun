//! Local Unlimited-OCR, run over bounded crops, checked, and cached.
//!
//! ## What is reused and what is new
//!
//! The transport is `ai_engine::ocr_stream::stream_ocr` unchanged — the same
//! request body, DRY settings, repetition guard, cancellable waits and span
//! parser that `commands::ocr` reads chat attachments with. What this module
//! adds is everything *after* the stream: turning the parser's events into
//! [`EvidenceRegion`]s located on the page, and refusing to call a read
//! complete when it is not.
//!
//! ## The checks, and what each one changes
//!
//! | Signal | Where it comes from | What the region says |
//! |---|---|---|
//! | repetition | the stream guard's `looped_at`, then [`degenerate_tail_start`] over the assembled text | `looped`, text cut at the loop, later regions discarded and counted |
//! | decode cap | `hit_decode_cap` | the last region `truncated`; the page is partial |
//! | no box | text before any region header | `malformed`, located only to the crop |
//! | bad box | coordinates outside 0..=999, or running backwards | `malformed`, located only to the crop |
//! | empty region | a header with no text after it | `unreadable` — something is there and nothing was read |
//! | stop / deadline | the turn's token, the call's time limit | nothing kept, the unit listed as unread |
//!
//! None of these produces a confidence number. A model cannot supply a
//! calibrated one, and a number derived from these flags would be a heuristic
//! wearing a probability's clothes; the flags are reported as themselves.
//!
//! ## The cache
//!
//! Keyed by everything that could change the answer: the document hash, the
//! page, the crop's box and resolution, the crop image's own hash, the weights'
//! pinned hash, the projector file and size, the detent, the request exactly as
//! sent (sampler settings and prompt) and [`OCR_PARSER_VERSION`]. What is stored
//! is the stream's events and summary, not the regions: the regions are derived
//! here, deterministically, so a cached read and a fresh one go through the
//! same checks. A stopped or timed-out read is never cached.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::agent_runtime::cancellation::CancelToken;
use crate::ai_engine::ocr_profile::{to_page, CoordSpace, OcrDetent, PageGeometry, NORMALISED_MAX};
use crate::ai_engine::ocr_repetition::degenerate_tail_start;
use crate::ai_engine::ocr_spans::{OcrEvent, RawBox};
use crate::ai_engine::ocr_stream::{request_body, stream_ocr, StreamSummary};

use super::regions::{
    region_id, BBox, CropRecord, EvidenceRegion, Extractor, Method, RegionStatus,
};
use super::tables::parse_ocr_table;

/// Bumped whenever the event-to-region derivation changes, which invalidates
/// every cached read: the events are kept, the regions are re-derived.
pub const OCR_PARSER_VERSION: &str = "arjun-ocr-regions/1";

/// Pages or crops one tool call may send to the model. A dense page is minutes
/// of GPU time on the target card, and a call is bounded by its deadline; the
/// rest is asked for in the next call, and the cache keeps what was done.
pub const MAX_OCR_UNITS_PER_CALL: usize = 4;

/// Below this much time left, a unit is not started. A read cut by the deadline
/// keeps nothing, so starting one that cannot finish only spends the card.
pub const MIN_TIME_FOR_A_UNIT: Duration = Duration::from_secs(8);

/// The coordinate convention this build's model reports in. Measured — see
/// `commands::ocr::CALIBRATED_COORD_SPACE`, whose value this repeats.
pub const OCR_COORD_SPACE: CoordSpace = CoordSpace::Normalised;

/// Exactly what read a unit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrIdentity {
    pub model_id: String,
    pub weights_file: String,
    /// The registry's pinned hash. `None` when the registry pins none, which
    /// is itself reported.
    pub weights_sha256: Option<String>,
    pub weights_bytes: u64,
    pub projector_file: Option<String>,
    pub projector_bytes: Option<u64>,
    pub detent: String,
    /// SHA-256 of the request body as sent, image aside: model, sampler
    /// settings, decode cap and prompt.
    pub request_fingerprint: String,
    pub prompt: String,
}

impl OcrIdentity {
    /// Built from a registry entry, for a detent.
    pub fn for_entry(entry: &crate::registry::ModelEntry, models_dir: &Path, detent: OcrDetent) -> Self {
        let profile = detent.profile();
        let body = request_body(&entry.id, "image/png", "", &profile);
        let fingerprint = hex::encode(Sha256::digest(body.to_string().as_bytes()));
        let projector = entry.projector.as_ref().map(|p| models_dir.join(p));
        Self {
            model_id: entry.id.clone(),
            weights_file: entry
                .path
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default(),
            weights_sha256: entry.sha256.clone(),
            weights_bytes: entry.weights_bytes,
            projector_file: projector
                .as_ref()
                .and_then(|p| p.file_name())
                .map(|n| n.to_string_lossy().to_string()),
            projector_bytes: projector
                .as_ref()
                .and_then(|p| std::fs::metadata(p).ok())
                .map(|m| m.len()),
            detent: detent.label().to_string(),
            request_fingerprint: fingerprint,
            prompt: profile.prompt().to_string(),
        }
    }

    pub fn extractor(&self) -> Extractor {
        Extractor {
            name: "unlimited-ocr".to_string(),
            version: OCR_PARSER_VERSION.to_string(),
            model_id: Some(self.model_id.clone()),
            weights_sha256: self.weights_sha256.clone(),
            projector: self.projector_file.as_ref().map(|file| match self.projector_bytes {
                Some(bytes) => format!("{file} ({bytes} bytes)"),
                None => format!("{file} (not found on disk)"),
            }),
            profile: Some(format!(
                "{} detent; request {}",
                self.detent,
                &self.request_fingerprint[..12.min(self.request_fingerprint.len())]
            )),
        }
    }
}

/// A local server ready to read, and the card it holds.
pub struct OcrSession {
    pub base_url: String,
    pub served_model_id: String,
    /// What reserved the card, or why nothing had to.
    pub residency: String,
    /// Held for as long as the session lives; dropping it releases the card.
    pub _lease: Option<crate::subagents::scheduling::ModelLease>,
}

/// A card the caller already holds, so a worker leased for the OCR model is
/// not made to wait on itself.
#[derive(Debug, Clone)]
pub struct HeldCard {
    pub model_id: String,
    pub exclusive: bool,
}

/// Where OCR comes from on this deployment.
#[async_trait]
pub trait OcrService: Send + Sync {
    /// What a read at this detent would be, without starting anything. The
    /// cache is consulted with this before any GPU is touched.
    fn identity(&self, detent: OcrDetent) -> Result<OcrIdentity, String>;
    /// Reserves the card through the shared scheduler and starts (or reuses)
    /// the loopback server.
    async fn open(
        &self,
        detent: OcrDetent,
        held: Option<&HeldCard>,
        wait: Duration,
    ) -> Result<OcrSession, String>;
    /// How reads reach the model, for the evidence label.
    fn transport(&self) -> &'static str;
}

/// The production service: the registry's OCR entry, served by `llama-server`,
/// with the card reserved through the same scheduler every worker uses.
pub struct ServedOcr {
    pub registry: Arc<crate::registry::ModelRegistry>,
    pub servers: Arc<crate::serving::ModelServers>,
    pub scheduler: Arc<crate::subagents::scheduling::ModelScheduler>,
}

#[async_trait]
impl OcrService for ServedOcr {
    fn identity(&self, detent: OcrDetent) -> Result<OcrIdentity, String> {
        let model_id = crate::commands::ocr::ocr_model_id(detent);
        let entry = self.registry.find(model_id).ok_or_else(|| {
            format!(
                "{model_id} is not in this machine's model registry, so no page can be read by \
                 OCR. Nothing was sent anywhere else: this product has no hosted OCR fallback."
            )
        })?;
        if !entry.enabled {
            return Err(format!("{model_id} is registered but disabled, so no page can be read by OCR."));
        }
        Ok(OcrIdentity::for_entry(entry, self.registry.models_dir(), detent))
    }

    async fn open(
        &self,
        detent: OcrDetent,
        held: Option<&HeldCard>,
        wait: Duration,
    ) -> Result<OcrSession, String> {
        let model_id = crate::commands::ocr::ocr_model_id(detent);
        let entry = self
            .registry
            .find(model_id)
            .cloned()
            .ok_or_else(|| format!("{model_id} is not in the model registry"))?;

        let (lease, residency) = match held {
            Some(card) if card.model_id == model_id => {
                (None, format!("the caller already holds {model_id}"))
            }
            Some(card) if card.exclusive => {
                return Err(format!(
                    "this worker holds the card for {} and OCR needs {model_id}; waiting for the \
                     card would wait on itself. Run the extraction from a worker routed to the \
                     OCR model, or from the parent's own tool call.",
                    card.model_id
                ))
            }
            _ => {
                let lease = self
                    .scheduler
                    .reserve(model_id, wait)
                    .await
                    .map_err(|refusal| refusal.explain())?;
                let residency = lease.describe();
                (Some(lease), residency)
            }
        };

        let plan = crate::serving::admission::admit(&self.servers, &entry, self.registry.models_dir())
            .await
            .map_err(|error| error.to_string())?
            .plan;
        let endpoint = self
            .servers
            .endpoint_for(&entry, self.registry.models_dir(), &plan)
            .await
            .map_err(|error| error.to_string())?;
        crate::serving::probe::check_loopback(&endpoint.base_url).map_err(|outcome| {
            format!(
                "refusing to send a page off-machine: {}",
                outcome.explain(&endpoint.base_url)
            )
        })?;
        Ok(OcrSession {
            base_url: endpoint.base_url,
            served_model_id: endpoint.served_model_id,
            residency,
            _lease: lease,
        })
    }

    fn transport(&self) -> &'static str {
        "local llama-server (loopback)"
    }
}

/// A deployment with no OCR at all. Every read is refused with the reason.
pub struct NoOcr(pub String);

#[async_trait]
impl OcrService for NoOcr {
    fn identity(&self, _detent: OcrDetent) -> Result<OcrIdentity, String> {
        Err(self.0.clone())
    }

    async fn open(&self, _: OcrDetent, _: Option<&HeldCard>, _: Duration) -> Result<OcrSession, String> {
        Err(self.0.clone())
    }

    fn transport(&self) -> &'static str {
        "none"
    }
}

/// One image to read: a page or a crop of one.
#[derive(Debug, Clone)]
pub struct OcrUnit {
    pub document_sha256: String,
    pub crop: CropRecord,
    pub image: PathBuf,
}

/// What reading one unit produced.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnitRead {
    pub page: u32,
    pub crop_id: String,
    pub cache_key: String,
    pub regions: Vec<EvidenceRegion>,
    pub looped: bool,
    pub truncated: bool,
    pub tokens: u32,
    pub elapsed_ms: u64,
    pub cached: bool,
    /// What the checks noticed, for the unit as a whole.
    pub notes: Vec<String>,
}

/// A unit that produced nothing usable, and why.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct UnitUnread {
    pub page: u32,
    pub crop_id: String,
    pub reason: String,
}

/// What the stream said, as kept in the cache.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CachedRead {
    pub cache_key: String,
    pub inputs: serde_json::Value,
    pub events: Vec<OcrEvent>,
    pub tokens: u32,
    pub elapsed_ms: u64,
    pub hit_decode_cap: bool,
    pub looped_at: Option<usize>,
    pub at: String,
}

/// The cache's key material, kept beside the entry so a reader can see what
/// the key was made of.
pub fn cache_inputs(unit: &OcrUnit, identity: &OcrIdentity) -> serde_json::Value {
    let b = unit.crop.bbox.rounded();
    serde_json::json!({
        "parser": OCR_PARSER_VERSION,
        "documentSha256": unit.document_sha256,
        "page": unit.crop.page,
        "bbox": [b.x0, b.y0, b.x1, b.y1],
        "coordSpace": unit.crop.coord_space.label(),
        "pixelsPerUnit": format!("{:.6}", unit.crop.pixels_per_unit),
        "dpi": unit.crop.dpi,
        "imageSha256": unit.crop.image_sha256,
        "identity": identity,
        "coordConvention": format!("{OCR_COORD_SPACE:?}"),
    })
}

pub fn cache_key(inputs: &serde_json::Value) -> String {
    hex::encode(Sha256::digest(inputs.to_string().as_bytes()))
}

/// OCR reads kept past the call that made them.
pub struct OcrCache {
    root: PathBuf,
}

impl OcrCache {
    /// `documents_root` is `<app data>/documents`.
    pub fn new(documents_root: &Path) -> Self {
        Self {
            root: documents_root.join("ocr-cache"),
        }
    }

    pub fn get(&self, key: &str) -> Option<CachedRead> {
        if !super::regions::is_sha(key) {
            return None;
        }
        let raw = std::fs::read(self.root.join(format!("{key}.json"))).ok()?;
        serde_json::from_slice::<CachedRead>(&raw)
            .ok()
            .filter(|held| held.cache_key == key)
    }

    pub fn put(&self, read: &CachedRead) -> Result<(), String> {
        std::fs::create_dir_all(&self.root)
            .map_err(|error| format!("the OCR cache could not be created: {error}"))?;
        let path = self.root.join(format!("{}.json", read.cache_key));
        let temporary = path.with_extension("json.tmp");
        let bytes = serde_json::to_vec(read)
            .map_err(|error| format!("the OCR read could not be encoded: {error}"))?;
        std::fs::write(&temporary, bytes)
            .map_err(|error| format!("the OCR cache could not be written: {error}"))?;
        std::fs::rename(&temporary, &path)
            .map_err(|error| format!("the OCR cache could not be replaced: {error}"))
    }
}

/// One region as the stream built it, before it is checked.
#[derive(Debug, Clone)]
struct Draft {
    label: String,
    raw: Option<RawBox>,
    text: String,
}

/// Turns one unit's stream into checked regions.
///
/// Pure: the same events and summary always produce the same regions, which
/// is what lets a cached read be re-derived rather than trusted.
pub fn derive_regions(
    unit: &OcrUnit,
    identity: &OcrIdentity,
    cache_key: &str,
    events: &[OcrEvent],
    tokens: u32,
    elapsed_ms: u64,
    hit_decode_cap: bool,
    stream_looped_at: Option<usize>,
    cached: bool,
) -> UnitRead {
    // -- Assemble drafts in stream order ------------------------------------
    let mut drafts: Vec<Draft> = Vec::new();
    let mut by_index: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    let mut unboxed: Option<usize> = None;
    for event in events {
        match event {
            OcrEvent::Region { index, label, bbox } => {
                by_index.insert(*index, drafts.len());
                drafts.push(Draft {
                    label: label.clone(),
                    raw: *bbox,
                    text: String::new(),
                });
            }
            OcrEvent::Text { index: Some(index), delta } => match by_index.get(index) {
                Some(&at) => drafts[at].text.push_str(delta),
                None => {
                    // A text event naming a region that never opened. Kept as
                    // unlocated text rather than dropped.
                    drafts.push(Draft {
                        label: "text".to_string(),
                        raw: None,
                        text: delta.clone(),
                    });
                }
            },
            OcrEvent::Text { index: None, delta } => {
                if delta.trim().is_empty() {
                    continue;
                }
                match unboxed {
                    Some(at) if at + 1 == drafts.len() => drafts[at].text.push_str(delta),
                    _ => {
                        unboxed = Some(drafts.len());
                        drafts.push(Draft {
                            label: "text".to_string(),
                            raw: None,
                            text: delta.clone(),
                        });
                    }
                }
            }
        }
    }

    let mut notes = Vec::new();
    let mut statuses: Vec<(RegionStatus, Vec<String>)> =
        drafts.iter().map(|_| (RegionStatus::Read, Vec::new())).collect();

    // -- Repetition --------------------------------------------------------
    //
    // Measured over the text the parser kept, in stream order, which is the
    // text the regions carry — the stream guard's offset addresses the raw
    // output, headers included, and cannot be used to cut this.
    let mut looped = false;
    let assembled: Vec<(usize, usize)> = {
        let mut spans = Vec::new();
        let mut offset = 0usize;
        for (at, draft) in drafts.iter().enumerate() {
            let length = draft.text.chars().count();
            spans.push((offset, at));
            offset += length;
        }
        spans
    };
    let all: String = drafts.iter().map(|d| d.text.as_str()).collect();
    // Two shapes of loop. The line guard catches the observed one — short
    // region lines cycling a few values — and deliberately ignores long lines,
    // which is where a phrase repeated along one line hides; the period check
    // catches that one. Both report where the loop began.
    let loop_start = degenerate_tail_start(&all)
        .map(|at| (at, "the transcription repeated itself from here and was cut at the repetition".to_string()))
        .or_else(|| {
            periodic_tail(&all).map(|(at, period)| {
                (
                    at,
                    format!(
                        "the transcription repeated one {period}-character pattern from here to \
                         the end and was cut where the repetition began"
                    ),
                )
            })
        });
    if let Some((cut, why)) = loop_start {
        looped = true;
        // The draft the loop starts in.
        let owner = assembled
            .iter()
            .rev()
            .find(|(start, _)| *start <= cut)
            .map(|(start, at)| (*start, *at));
        if let Some((start, at)) = owner {
            let keep = cut - start;
            drafts[at].text = drafts[at].text.chars().take(keep).collect();
            statuses[at].0 = RegionStatus::Looped;
            statuses[at].1.push(why);
            let discarded = drafts.len() - at - 1;
            if discarded > 0 {
                notes.push(format!(
                    "{discarded} region(s) the model emitted after the repetition began were \
                     discarded: they are the loop, not the page"
                ));
                drafts.truncate(at + 1);
                statuses.truncate(at + 1);
            }
        }
    } else if stream_looped_at.is_some() {
        // The guard stopped the stream on repetition the parser stripped (in
        // the region headers). The last region is where it was.
        looped = true;
        if let Some(last) = statuses.last_mut() {
            last.0 = RegionStatus::Looped;
            last.1.push(
                "the read was stopped because the model began repeating region headers"
                    .to_string(),
            );
        }
    }
    if looped {
        notes.push(
            "this read degenerated into repetition; the page below the cut was not read".to_string(),
        );
    }

    // -- Decode cap ----------------------------------------------------------
    let truncated = hit_decode_cap;
    if truncated {
        notes.push(
            "the read stopped at the decode cap, not because the model finished: anything below \
             the last region was not read"
                .to_string(),
        );
        if let Some(last) = statuses.last_mut() {
            if last.0 == RegionStatus::Read {
                last.0 = RegionStatus::Truncated;
            }
            last.1.push("the decode cap was reached inside this region".to_string());
        }
    }

    // -- Per-region shape ----------------------------------------------------
    let crop = &unit.crop;
    let crop_box = crop.bbox;
    let geometry = PageGeometry {
        page_width: crop.width,
        page_height: crop.height,
        input_width: crop.width,
        input_height: crop.height,
    };
    let limit = NORMALISED_MAX as i32;

    let mut regions = Vec::new();
    for (ordinal, (draft, (status, region_notes))) in
        drafts.iter().zip(statuses.into_iter()).enumerate()
    {
        let mut status = status;
        let mut region_notes = region_notes;
        let text = draft.text.trim().to_string();

        let bbox = match draft.raw {
            None => {
                status = RegionStatus::Malformed;
                region_notes.push(
                    "this text arrived with no region box, so it is located only to the crop it \
                     was read from"
                        .to_string(),
                );
                crop_box
            }
            Some(raw)
                if raw.x1 < 0
                    || raw.y1 < 0
                    || raw.x2 > limit
                    || raw.y2 > limit
                    || raw.x2 <= raw.x1
                    || raw.y2 <= raw.y1 =>
            {
                status = RegionStatus::Malformed;
                region_notes.push(format!(
                    "the model's box [{}, {}, {}, {}] is not a box on the 0-{limit} grid, so it \
                     is located only to the crop",
                    raw.x1, raw.y1, raw.x2, raw.y2
                ));
                crop_box
            }
            Some(raw) => {
                let pixels = to_page(raw, OCR_COORD_SPACE, geometry);
                let scale = crop.pixels_per_unit.max(f64::EPSILON);
                BBox::new(
                    crop_box.x0 + pixels.x1 as f64 / scale,
                    crop_box.y0 + pixels.y1 as f64 / scale,
                    crop_box.x0 + pixels.x2 as f64 / scale,
                    crop_box.y0 + pixels.y2 as f64 / scale,
                )
                .rounded()
            }
        };

        // Only a clean read becomes `unreadable` for being empty: a region
        // emptied by the loop cut is the loop's start, and says so.
        if text.is_empty() && status == RegionStatus::Read {
            status = RegionStatus::Unreadable;
            region_notes.push(format!(
                "the model marked a {} region here and transcribed nothing from it",
                draft.label
            ));
        }

        let mut cells = Vec::new();
        if draft.label.eq_ignore_ascii_case("table") && !text.is_empty() {
            match parse_ocr_table(&text) {
                Some(parsed) => {
                    cells = parsed.cells;
                    region_notes.extend(parsed.notes);
                }
                None => region_notes.push(
                    "the transcription did not delimit this table's cells, so it is kept as text \
                     and no cell is reported"
                        .to_string(),
                ),
            }
        }

        regions.push(EvidenceRegion {
            region_id: region_id(
                &unit.document_sha256,
                crop.page,
                Method::Ocr,
                &draft.label,
                &bbox,
                cache_key,
                ordinal,
            ),
            document_sha256: unit.document_sha256.clone(),
            page: crop.page,
            bbox,
            coord_space: crop.coord_space,
            label: draft.label.clone(),
            method: Method::Ocr,
            status,
            text,
            cells,
            notes: region_notes,
            crop_id: Some(crop.crop_id.clone()),
            image_sha256: Some(crop.image_sha256.clone()),
            extractor: identity.extractor(),
            cache_key: Some(cache_key.to_string()),
        });
    }

    if regions.is_empty() {
        // Read, and nothing came back. Recorded as the crop being unreadable
        // rather than as no record at all: "the model saw this and returned
        // nothing" and "nobody has looked" lead to different next steps, and
        // an absent region would say the second.
        notes.push("the model returned nothing for this image".to_string());
        regions.push(EvidenceRegion {
            region_id: region_id(
                &unit.document_sha256,
                crop.page,
                Method::Ocr,
                "page",
                &crop_box,
                cache_key,
                0,
            ),
            document_sha256: unit.document_sha256.clone(),
            page: crop.page,
            bbox: crop_box,
            coord_space: crop.coord_space,
            label: "page".to_string(),
            method: Method::Ocr,
            status: RegionStatus::Unreadable,
            text: String::new(),
            cells: Vec::new(),
            notes: vec!["the model read this image and returned no text for any part of it".to_string()],
            crop_id: Some(crop.crop_id.clone()),
            image_sha256: Some(crop.image_sha256.clone()),
            extractor: identity.extractor(),
            cache_key: Some(cache_key.to_string()),
        });
    }

    UnitRead {
        page: crop.page,
        crop_id: crop.crop_id.clone(),
        cache_key: cache_key.to_string(),
        regions,
        looped,
        truncated,
        tokens,
        elapsed_ms,
        cached,
        notes,
    }
}

/// Where a text ends in one short pattern repeated, and the pattern's length.
///
/// The line-based guard in `ocr_repetition` treats any line longer than 96
/// characters as content, which is right for a table row and wrong for a
/// phrase the model repeats along one line until the decode cap. This finds
/// that shape: a tail at least [`MIN_PERIODIC_RUN`] characters (and twelve
/// periods) long that is exactly periodic with a period of at most
/// [`MAX_PERIOD`]. Text is compared as written, so a real register whose rows
/// differ in their numbers is not periodic.
pub fn periodic_tail(text: &str) -> Option<(usize, usize)> {
    let chars: Vec<char> = text.trim_end().chars().collect();
    let n = chars.len();
    if n < MIN_PERIODIC_RUN {
        return None;
    }
    let mut best: Option<(usize, usize)> = None;
    for period in 1..=MAX_PERIOD.min(n / 4) {
        let mut start = n - period;
        while start > 0 && chars[start - 1] == chars[start - 1 + period] {
            start -= 1;
        }
        let run = n - start;
        if run >= MIN_PERIODIC_RUN.max(12 * period) {
            // A whitespace-only pattern is layout, not a loop.
            if chars[start..start + period].iter().all(|c| c.is_whitespace()) {
                continue;
            }
            if best.map_or(true, |(at, _)| start < at) {
                best = Some((start, period));
            }
        }
    }
    best
}

/// The shortest periodic tail [`periodic_tail`] reports, in characters.
pub const MIN_PERIODIC_RUN: usize = 240;
/// The longest repeated pattern it looks for.
pub const MAX_PERIOD: usize = 64;

/// Pages a batch was asked for and did not get back in order.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchCheck {
    pub missing: Vec<u32>,
    pub duplicated: Vec<u32>,
    pub out_of_order: bool,
}

impl BatchCheck {
    pub fn clean(&self) -> bool {
        self.missing.is_empty() && self.duplicated.is_empty() && !self.out_of_order
    }
}

/// Compares the pages a batch asked for with the pages it produced.
///
/// Applied to every batch, cached or fresh, and to a document's stored page
/// text. A result set that silently lost page 3 or returned page 4 before
/// page 2 is exactly what a citation would then misplace.
pub fn check_batch(requested: &[u32], produced: &[u32]) -> BatchCheck {
    let mut seen = std::collections::BTreeSet::new();
    let mut duplicated = Vec::new();
    for page in produced {
        if !seen.insert(*page) && !duplicated.contains(page) {
            duplicated.push(*page);
        }
    }
    let missing = requested
        .iter()
        .copied()
        .filter(|page| !seen.contains(page))
        .collect();
    let out_of_order = produced.windows(2).any(|pair| pair[1] < pair[0]);
    BatchCheck {
        missing,
        duplicated,
        out_of_order,
    }
}

/// The loopback client OCR is read over.
///
/// `read_timeout` rather than a whole-request timeout: a page streaming
/// correctly but slowly is the normal case on a small GPU; what is ruled out is
/// a server that stops sending forever. The call's own deadline bounds the rest.
fn ocr_client() -> &'static reqwest::Client {
    static CLIENT: std::sync::OnceLock<reqwest::Client> = std::sync::OnceLock::new();
    CLIENT.get_or_init(|| {
        reqwest::Client::builder() // arjun-egress-ok: loopback only; `ServedOcr::open` checks the host first
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(120))
            .no_proxy()
            .build()
            .expect("the OCR http client builds from constants")
    })
}

/// What a batch produced.
#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BatchRead {
    pub read: Vec<UnitRead>,
    pub unread: Vec<UnitUnread>,
    pub identity: Option<OcrIdentity>,
    pub transport: String,
    pub residency: Option<String>,
    pub check: BatchCheck,
}

/// Reads a bounded batch: the cache first, then the model for the rest.
///
/// Units are read one at a time in page order. `deadline` is the call's own
/// limit; a unit that cannot start with [`MIN_TIME_FOR_A_UNIT`] left is not
/// started, and one cut by the deadline keeps nothing.
pub async fn read_batch(
    service: &dyn OcrService,
    cache: &OcrCache,
    detent: OcrDetent,
    units: Vec<OcrUnit>,
    held: Option<&HeldCard>,
    cancel: &CancelToken,
    deadline: Instant,
) -> BatchRead {
    let requested: Vec<u32> = units.iter().map(|u| u.crop.page).collect();
    let mut read = Vec::new();
    let mut unread = Vec::new();
    let transport = service.transport().to_string();

    let identity = match service.identity(detent) {
        Ok(identity) => identity,
        Err(reason) => {
            let check = check_batch(&requested, &[]);
            return BatchRead {
                read,
                unread: units
                    .iter()
                    .map(|u| UnitUnread {
                        page: u.crop.page,
                        crop_id: u.crop.crop_id.clone(),
                        reason: reason.clone(),
                    })
                    .collect(),
                identity: None,
                transport,
                residency: None,
                check,
            };
        }
    };

    // Cache first. Nothing below touches the card for a unit already read
    // under exactly these settings.
    let mut pending = Vec::new();
    for unit in units {
        let inputs = cache_inputs(&unit, &identity);
        let key = cache_key(&inputs);
        match cache.get(&key) {
            Some(hit) => read.push(derive_regions(
                &unit,
                &identity,
                &key,
                &hit.events,
                hit.tokens,
                hit.elapsed_ms,
                hit.hit_decode_cap,
                hit.looped_at,
                true,
            )),
            None => pending.push((unit, inputs, key)),
        }
    }

    let mut residency = None;
    if !pending.is_empty() {
        let wait = deadline
            .saturating_duration_since(Instant::now())
            .min(Duration::from_secs(30));
        match service.open(detent, held, wait).await {
            Err(reason) => {
                for (unit, _, _) in &pending {
                    unread.push(UnitUnread {
                        page: unit.crop.page,
                        crop_id: unit.crop.crop_id.clone(),
                        reason: reason.clone(),
                    });
                }
            }
            Ok(session) => {
                residency = Some(session.residency.clone());
                let profile = detent.profile();
                for (unit, inputs, key) in pending {
                    if cancel.is_cancelled() {
                        unread.push(UnitUnread {
                            page: unit.crop.page,
                            crop_id: unit.crop.crop_id.clone(),
                            reason: "the turn was stopped before this was read".to_string(),
                        });
                        continue;
                    }
                    let remaining = deadline.saturating_duration_since(Instant::now());
                    if remaining < MIN_TIME_FOR_A_UNIT {
                        unread.push(UnitUnread {
                            page: unit.crop.page,
                            crop_id: unit.crop.crop_id.clone(),
                            reason: "this call's time limit was reached before it could start; \
                                     ask for it again and the pages already read come from the \
                                     cache"
                                .to_string(),
                        });
                        continue;
                    }
                    let mut events: Vec<OcrEvent> = Vec::new();
                    let started = Instant::now();
                    let streamed = tokio::time::timeout(
                        remaining,
                        stream_ocr(
                            ocr_client(),
                            &session.base_url,
                            &session.served_model_id,
                            &unit.image,
                            &profile,
                            cancel,
                            |event| events.push(event),
                        ),
                    )
                    .await;
                    let summary: StreamSummary = match streamed {
                        Err(_) => {
                            unread.push(UnitUnread {
                                page: unit.crop.page,
                                crop_id: unit.crop.crop_id.clone(),
                                reason: format!(
                                    "this call's time limit ran out {}s into reading it; nothing \
                                     from a half-read image is kept",
                                    started.elapsed().as_secs()
                                ),
                            });
                            continue;
                        }
                        Ok(Err(error)) => {
                            unread.push(UnitUnread {
                                page: unit.crop.page,
                                crop_id: unit.crop.crop_id.clone(),
                                reason: format!("reading it failed: {error:#}"),
                            });
                            continue;
                        }
                        Ok(Ok(summary)) => summary,
                    };
                    if summary.cancelled {
                        unread.push(UnitUnread {
                            page: unit.crop.page,
                            crop_id: unit.crop.crop_id.clone(),
                            reason: "the turn was stopped while this was being read, so what \
                                     arrived is only part of it"
                                .to_string(),
                        });
                        continue;
                    }
                    let cached = CachedRead {
                        cache_key: key.clone(),
                        inputs,
                        events: events.clone(),
                        tokens: summary.tokens,
                        elapsed_ms: summary.elapsed_ms,
                        hit_decode_cap: summary.hit_decode_cap,
                        looped_at: summary.looped_at,
                        at: chrono::Utc::now().to_rfc3339(),
                    };
                    let mut derived = derive_regions(
                        &unit,
                        &identity,
                        &key,
                        &events,
                        summary.tokens,
                        summary.elapsed_ms,
                        summary.hit_decode_cap,
                        summary.looped_at,
                        false,
                    );
                    if let Err(problem) = cache.put(&cached) {
                        derived.notes.push(format!("{problem}; the next call will read it again"));
                    }
                    read.push(derived);
                }
            }
        }
    }

    read.sort_by_key(|unit| unit.page);
    unread.sort_by_key(|unit| unit.page);
    // Only the pages that were read are checked against the request: an
    // unread page is already named, with its reason.
    let produced: Vec<u32> = read.iter().map(|unit| unit.page).collect();
    let still_asked: Vec<u32> = requested
        .iter()
        .copied()
        .filter(|page| !unread.iter().any(|u| u.page == *page))
        .collect();
    let mut check = check_batch(&still_asked, &produced);
    // Several crops of one page are legitimately several units.
    check.duplicated.retain(|page| requested.iter().filter(|p| *p == page).count() < 2);

    BatchRead {
        read,
        unread,
        identity: Some(identity),
        transport,
        residency,
        check,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::regions::PageSpace;

    const SHA: &str = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef";

    fn identity() -> OcrIdentity {
        OcrIdentity {
            model_id: "unlimited-ocr-q6-k".into(),
            weights_file: "Unlimited-OCR-Q6_K.gguf".into(),
            weights_sha256: Some("a9".repeat(32)),
            weights_bytes: 1,
            projector_file: Some("mmproj-Unlimited-OCR-F16-patched.gguf".into()),
            projector_bytes: Some(2),
            detent: "Detailed".into(),
            request_fingerprint: "f".repeat(64),
            prompt: "Free OCR.".into(),
        }
    }

    /// A 999x999-pixel crop of the box (100,200)-(199.9,299.9) at 10 px/pt.
    fn unit() -> OcrUnit {
        OcrUnit {
            document_sha256: SHA.into(),
            crop: CropRecord {
                crop_id: "cr-1".into(),
                document_sha256: SHA.into(),
                page: 3,
                bbox: BBox::new(100.0, 200.0, 199.9, 299.9),
                coord_space: PageSpace::PdfPoints,
                pixels_per_unit: 10.0,
                width: 999,
                height: 999,
                image_sha256: "e".repeat(64),
                file: "crops/cr-1.png".into(),
                dpi: Some(720),
            },
            image: PathBuf::from("/nonexistent.png"),
        }
    }

    fn events(raw: &str) -> Vec<OcrEvent> {
        let mut parser = crate::ai_engine::ocr_spans::SpanParser::new();
        let mut all = parser.feed(raw);
        all.extend(parser.finish());
        all
    }

    fn derive(raw: &str, cap: bool, guard: Option<usize>) -> UnitRead {
        derive_regions(&unit(), &identity(), &"k".repeat(64), &events(raw), 10, 10, cap, guard, false)
    }

    #[test]
    fn a_normalised_box_lands_on_the_page_through_the_crop() {
        // 0..999 normalised over a 999-pixel crop is one pixel per step; at
        // 10 px per point that is a tenth of a point, offset by the crop.
        let read = derive("text [100, 200, 300, 400]Design pressure: 10 bar\n", false, None);
        let region = &read.regions[0];
        assert_eq!(region.status, RegionStatus::Read);
        assert_eq!(region.bbox, BBox::new(110.0, 220.0, 130.0, 240.0));
        assert_eq!(region.page, 3);
        assert_eq!(region.coord_space, PageSpace::PdfPoints);
        assert_eq!(region.method, Method::Ocr);
        assert_eq!(region.crop_id.as_deref(), Some("cr-1"));
    }

    #[test]
    fn a_region_marked_but_not_read_is_unreadable_not_absent() {
        let read = derive("text [10, 10, 200, 60]\ntext [10, 100, 900, 150]LINE 2\n", false, None);
        assert_eq!(read.regions.len(), 2);
        assert_eq!(read.regions[0].status, RegionStatus::Unreadable);
        assert!(read.regions[0].notes[0].contains("transcribed nothing"));
        assert_eq!(read.regions[1].status, RegionStatus::Read);
    }

    #[test]
    fn a_long_repeated_output_of_region_lines_is_cut_and_the_loop_is_not_kept() {
        // The observed shape: short region lines cycling a few values, their
        // boxes creeping down the page, until the decode cap.
        let mut raw = String::from("title [10, 10, 900, 60]SITE SURVEY\ntext [10, 80, 900, 120]Pump P-101 inlet\n");
        for i in 0..160 {
            let word = ["OVO", "AUDIO", "AVO"][i % 3];
            raw.push_str(&format!("image_caption [{}, {}, {}, {}]{word}\n", 300 + i % 50, 150 + i % 50, 420 + i % 50, 170 + i % 50));
        }
        let read = derive(&raw, true, Some(200));
        assert!(read.looped && read.truncated);
        assert_eq!(read.regions[0].status, RegionStatus::Read);
        assert_eq!(read.regions[1].status, RegionStatus::Read);
        assert_eq!(read.regions[1].text, "Pump P-101 inlet");
        let looped: Vec<_> = read.regions.iter().filter(|r| r.status == RegionStatus::Looped).collect();
        assert_eq!(looped.len(), 1, "{:#?}", read.regions);
        assert!(read.regions.len() < 10, "the loop's regions were kept: {}", read.regions.len());
        assert!(read.notes.iter().any(|n| n.contains("discarded")));
    }

    #[test]
    fn a_phrase_repeated_along_one_line_is_cut_where_it_began() {
        let mut raw = String::from("text [10, 10, 900, 60]Inspection note: ");
        for _ in 0..200 {
            raw.push_str("the valve ");
        }
        let read = derive(&raw, true, None);
        assert!(read.looped);
        let region = &read.regions[0];
        assert_eq!(region.status, RegionStatus::Looped);
        assert!(region.text.starts_with("Inspection note:"));
        assert!(region.text.len() < 40, "{}", region.text);
        assert!(region.notes.iter().any(|n| n.contains("10-character pattern")));
    }

    #[test]
    fn a_register_whose_rows_differ_is_not_a_loop() {
        let mut raw = String::from("table [10, 10, 900, 900]");
        for row in 0..70 {
            raw.push_str(&format!("Phase {} reading {} MOhm; ", row % 3, 380 + row));
        }
        raw.push('\n');
        let read = derive(&raw, false, None);
        assert!(!read.looped, "{:?}", read.notes);
        assert!(periodic_tail("abc").is_none());
    }

    #[test]
    fn text_with_no_box_and_a_box_off_the_grid_are_malformed_and_located_only_to_the_crop() {
        let read = derive("www.free.com\ntext [100, 100, 1400, 200]OFF GRID\n", false, None);
        assert_eq!(read.regions.len(), 2);
        assert!(read.regions.iter().all(|r| r.status == RegionStatus::Malformed));
        assert!(read.regions.iter().all(|r| r.bbox == unit().crop.bbox));
    }

    #[test]
    fn a_decode_cap_marks_the_last_region_truncated() {
        let read = derive("text [10, 10, 900, 60]FIRST\ntext [10, 80, 900, 120]SECO", true, None);
        assert!(read.truncated);
        assert_eq!(read.regions[0].status, RegionStatus::Read);
        assert_eq!(read.regions[1].status, RegionStatus::Truncated);
    }

    #[test]
    fn an_ocr_table_keeps_its_cells_and_a_flat_one_says_it_has_none() {
        let read = derive(
            "table [10, 10, 900, 300]<table><tr><td>Point</td><td>mm</td></tr><tr><td>A</td><td>9.4</td></tr></table>\n\
             table [10, 400, 900, 600]<table>PhaseReadingLimit</table>\n",
            false,
            None,
        );
        assert_eq!(read.regions[0].cells.len(), 4);
        assert!(read.regions[1].cells.is_empty());
        assert!(read.regions[1].notes.iter().any(|n| n.contains("did not delimit")));
    }

    #[test]
    fn the_same_events_always_give_the_same_region_ids() {
        let a = derive("text [100, 200, 300, 400]X\n", false, None);
        let b = derive("text [100, 200, 300, 400]X\n", false, None);
        assert_eq!(a.regions[0].region_id, b.regions[0].region_id);
    }

    #[test]
    fn the_cache_key_moves_with_the_crop_the_image_and_the_model() {
        let base = cache_key(&cache_inputs(&unit(), &identity()));
        let mut moved = unit();
        moved.crop.bbox.x0 += 1.0;
        let mut other_image = unit();
        other_image.crop.image_sha256 = "d".repeat(64);
        let mut other_model = identity();
        other_model.weights_sha256 = Some("c8".repeat(32));
        assert_ne!(base, cache_key(&cache_inputs(&moved, &identity())));
        assert_ne!(base, cache_key(&cache_inputs(&other_image, &identity())));
        assert_ne!(base, cache_key(&cache_inputs(&unit(), &other_model)));
        assert_eq!(base, cache_key(&cache_inputs(&unit(), &identity())));
    }

    #[test]
    fn a_batch_that_lost_a_page_or_reordered_one_is_not_clean() {
        assert!(check_batch(&[1, 2, 3], &[1, 2, 3]).clean());
        let lost = check_batch(&[1, 2, 3], &[1, 3]);
        assert_eq!(lost.missing, vec![2]);
        let reordered = check_batch(&[1, 2], &[2, 1]);
        assert!(reordered.out_of_order);
        let twice = check_batch(&[1, 2], &[1, 1, 2]);
        assert_eq!(twice.duplicated, vec![1]);
    }
}
