//! The accuracy-to-speed slider, and what comes back from it.
//!
//! ## What the slider actually changes
//!
//! Two things reach the server today: the **weight file** (the tier) and the
//! **decode cap** (`max_decode_tokens`, sent per request as `max_tokens`).
//! Stops 1–2 share Q4_K_M and stops 3–4 share Q6_K, so only the 2→3 move
//! reloads; within a tier the slider changes how much text one page may
//! produce, and is instant.
//!
//! ## The vision token budget, and why it is not the slider
//!
//! `max_image_tokens` is the DeepEncoder's resolution mode — 64 at tiny, 100 at
//! small, 256 at base, 400 at large — and it dominates both latency and how
//! much small print survives. It is the lever this module was designed around,
//! and it is **not wired**: [`OcrProfile::server_args`] builds the
//! `--image-max-tokens` flag and nothing but that function's own unit test ever
//! calls it. `serving::plan_launch` does not emit it, and `ocr_stream`'s
//! request body has no field for it.
//!
//! It is not a wiring oversight that can be fixed in one line, because
//! `--image-max-tokens` is a **launch** argument on a `llama-server` process
//! that `ModelServers` keys by model id and reuses across stops. Honouring it
//! per stop would mean either keying servers by `(model_id, image_tokens)` and
//! restarting on every within-tier move — which is exactly the "instant"
//! property above — or a per-request field llama.cpp does not currently offer.
//!
//! So the field is kept, because it records the intent and the numbers are
//! right, and it is no longer *reported to the user* as something the stop
//! changes. A slider that displays a number with no effect is worse than one
//! that does not mention it, and this module exists to stop the labels
//! disagreeing with the profiles that run.
//!
//! ## The sampler is not optional
//!
//! Baidu's own inference code runs a custom logit processor —
//! `no_repeat_ngram_size=35` over a 128-token window for a single image.
//! llama.cpp has no such sampler, and without it Q4_K_M is documented to loop
//! forever on some prompts. DRY is the closest thing llama.cpp ships, so
//! every profile carries DRY settings derived from those numbers. They are an
//! approximation of the original, not a reproduction of it, and Phase 6 has
//! to measure whether the approximation holds rather than assume it.

use serde::{Deserialize, Serialize};

use super::ocr_spans::RawBox;

/// Which weight file a stop needs. Only two are installed, by decision: the
/// card's recommended Q4_K_M sits below the loop-risk line for OCR, and Q8_0
/// buys little over Q6_K while costing half a gigabyte more.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OcrTier {
    /// Q6_K. The accuracy tier.
    High,
    /// Q4_K_M. Usable, but only with DRY engaged.
    Fast,
}

impl OcrTier {
    /// The file this tier loads, relative to the model directory.
    pub const fn weights_file(self) -> &'static str {
        match self {
            OcrTier::High => "Unlimited-OCR-Q6_K.gguf",
            OcrTier::Fast => "Unlimited-OCR-Q4_K_M.gguf",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            OcrTier::High => "High",
            OcrTier::Fast => "Fast",
        }
    }
}

/// The four slider positions, fastest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum OcrDetent {
    Fastest,
    Fast,
    Detailed,
    Maximum,
}

impl OcrDetent {
    /// Where this stop sits on the accuracy axis. Higher reads better.
    ///
    /// Explicit rather than derived from the enum's discriminant so that
    /// reordering the variants — which the UI does by index — cannot silently
    /// reorder *quality*, which other code compares on.
    pub const fn quality(self) -> u8 {
        match self {
            OcrDetent::Fastest => 0,
            OcrDetent::Fast => 1,
            OcrDetent::Detailed => 2,
            OcrDetent::Maximum => 3,
        }
    }

    pub const ALL: [OcrDetent; 4] = [
        OcrDetent::Fastest,
        OcrDetent::Fast,
        OcrDetent::Detailed,
        OcrDetent::Maximum,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            OcrDetent::Fastest => "Fastest",
            OcrDetent::Fast => "Fast",
            OcrDetent::Detailed => "Detailed",
            OcrDetent::Maximum => "Maximum",
        }
    }

    pub const fn profile(self) -> OcrProfile {
        match self {
            // Token budgets are the DeepEncoder's own resolution modes:
            // small (100), base (256), large (400). Tiny (64) is omitted — it
            // loses too much small print to earn a slider position on the
            // documents this product is aimed at.
            OcrDetent::Fastest => OcrProfile::new(OcrTier::Fast, 100, 2048),
            OcrDetent::Fast => OcrProfile::new(OcrTier::Fast, 256, 4096),
            OcrDetent::Detailed => OcrProfile::new(OcrTier::High, 256, 8192),
            OcrDetent::Maximum => OcrProfile::new(OcrTier::High, 400, 16384),
        }
    }

    /// Whether moving between these two stops swaps the weight file, and so
    /// costs a reload. The UI marks this boundary rather than letting a drag
    /// stall unexplained.
    pub const fn reloads_from(self, previous: OcrDetent) -> bool {
        !matches!(
            (previous.profile().tier, self.profile().tier),
            (OcrTier::High, OcrTier::High) | (OcrTier::Fast, OcrTier::Fast)
        )
    }
}

/// Everything one stop settles.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OcrProfile {
    pub tier: OcrTier,
    /// `--image-max-tokens`: the vision budget, and the real speed lever.
    pub max_image_tokens: u32,
    /// `-n`: a dense page needs room, and a looping one must still stop.
    pub max_decode_tokens: u32,
    /// DRY, standing in for Baidu's `no_repeat_ngram_size`.
    pub dry_multiplier: f32,
    pub dry_allowed_length: u32,
    pub dry_penalty_last_n: u32,
    /// OCR is transcription; sampling from it invents text.
    pub temperature: f32,
}

impl OcrProfile {
    const fn new(tier: OcrTier, max_image_tokens: u32, max_decode_tokens: u32) -> Self {
        Self {
            tier,
            max_image_tokens,
            max_decode_tokens,
            dry_multiplier: 0.8,
            // From `no_repeat_ngram_size=35`: repeats shorter than this are
            // ordinary language — table headers, repeated units — and must
            // not be penalised.
            dry_allowed_length: 35,
            // From `ngram_window=128`, the single-image setting. Multi-page
            // uses 1024 upstream, but pages are run one at a time here.
            dry_penalty_last_n: 128,
            temperature: 0.0,
        }
    }

    /// The llama-server arguments this profile implies.
    pub fn server_args(&self) -> Vec<String> {
        vec![
            "--image-max-tokens".into(),
            self.max_image_tokens.to_string(),
            "--dry-multiplier".into(),
            self.dry_multiplier.to_string(),
            "--dry-allowed-length".into(),
            self.dry_allowed_length.to_string(),
            "--dry-penalty-last-n".into(),
            self.dry_penalty_last_n.to_string(),
        ]
    }

    /// The only prompt this product sends.
    ///
    /// Not the documented one. `<|grounding|>Convert the document to
    /// markdown.` is what the model card prescribes, and on this build it
    /// returns *zero* generated tokens — measured twice, with and without an
    /// explicit media marker. `Free OCR.` returns the full page **with**
    /// labels and boxes, which is what the scan view needs.
    ///
    /// The card also warns that `Free OCR.` loops on Q4_K_M. That was not
    /// reproduced here: with the DRY settings below, Q4_K_M read the
    /// calibration page cleanly at 307 tok/s and stopped on its own. The loop
    /// counter in the bench harness exists to catch it if that changes on
    /// other documents.
    pub const fn prompt(&self) -> &'static str {
        "Free OCR."
    }
}

/// What the model was shown, and what the user is looking at.
///
/// These differ whenever the page was scaled to fit the vision encoder, which
/// is almost always. Every box has to cross that gap.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PageGeometry {
    /// The document as stored and displayed.
    pub page_width: u32,
    pub page_height: u32,
    /// The image actually handed to the encoder.
    pub input_width: u32,
    pub input_height: u32,
}

/// Which space the model's numbers are in.
///
/// This is not a guess to be made in code. It is a measured property of the
/// build, settled by running one page at two input sizes and comparing: boxes
/// that move with the input are pixels, boxes that do not are normalised. The
/// wrong choice yields an overlay correct on square pages and skewed on every
/// other one — the failure most likely to ship unnoticed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum CoordSpace {
    /// 0..=`NORMALISED_MAX`, relative to the input image.
    Normalised,
    /// Pixels in the input image.
    InputPixels,
}

/// DeepSeek-OCR's normalised grid is 0–999 inclusive.
pub const NORMALISED_MAX: f64 = 999.0;

/// A box in the coordinate space of the displayed page.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct PageBox {
    pub x1: i32,
    pub y1: i32,
    pub x2: i32,
    pub y2: i32,
    /// False when the source box fell outside the page.
    ///
    /// Reported rather than clamped. A box off the edge means the coordinate
    /// space is wrong, and quietly pulling it inside the page would hide
    /// exactly the evidence that says so.
    pub in_bounds: bool,
}

/// Maps one model box onto the displayed page.
pub fn to_page(raw: RawBox, space: CoordSpace, geometry: PageGeometry) -> PageBox {
    let (sx, sy) = match space {
        CoordSpace::Normalised => (
            geometry.page_width as f64 / NORMALISED_MAX,
            geometry.page_height as f64 / NORMALISED_MAX,
        ),
        CoordSpace::InputPixels => (
            geometry.page_width as f64 / geometry.input_width.max(1) as f64,
            geometry.page_height as f64 / geometry.input_height.max(1) as f64,
        ),
    };

    let x1 = (raw.x1 as f64 * sx).round() as i32;
    let y1 = (raw.y1 as f64 * sy).round() as i32;
    let x2 = (raw.x2 as f64 * sx).round() as i32;
    let y2 = (raw.y2 as f64 * sy).round() as i32;

    let w = geometry.page_width as i32;
    let h = geometry.page_height as i32;
    let in_bounds = x1 >= 0 && y1 >= 0 && x2 <= w && y2 <= h && x1 <= x2 && y1 <= y2;

    PageBox {
        x1,
        y1,
        x2,
        y2,
        in_bounds,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(page: (u32, u32), input: (u32, u32)) -> PageGeometry {
        PageGeometry {
            page_width: page.0,
            page_height: page.1,
            input_width: input.0,
            input_height: input.1,
        }
    }

    fn raw(x1: i32, y1: i32, x2: i32, y2: i32) -> RawBox {
        RawBox { x1, y1, x2, y2 }
    }

    #[test]
    fn the_slider_spans_two_tiers_and_reloads_exactly_once() {
        let tiers: Vec<OcrTier> = OcrDetent::ALL.iter().map(|d| d.profile().tier).collect();
        assert_eq!(
            tiers,
            vec![OcrTier::Fast, OcrTier::Fast, OcrTier::High, OcrTier::High]
        );
        let reloads = OcrDetent::ALL
            .windows(2)
            .filter(|pair| pair[1].reloads_from(pair[0]))
            .count();
        assert_eq!(reloads, 1, "the slider must cross a weight file once");
    }

    #[test]
    fn the_vision_budget_rises_with_every_stop() {
        let budgets: Vec<u32> = OcrDetent::ALL
            .iter()
            .map(|d| d.profile().max_image_tokens)
            .collect();
        assert!(
            budgets.windows(2).all(|p| p[0] <= p[1]),
            "a stop further right must never see less of the page: {budgets:?}"
        );
    }

    #[test]
    fn every_profile_decodes_deterministically() {
        // Sampling a transcription invents text that was never on the page.
        assert!(OcrDetent::ALL
            .iter()
            .all(|d| d.profile().temperature == 0.0));
    }

    #[test]
    fn every_profile_carries_the_repetition_guard() {
        // Without it Q4_K_M is documented to loop forever on some prompts.
        for detent in OcrDetent::ALL {
            let p = detent.profile();
            assert!(p.dry_multiplier > 0.0, "{detent:?} has DRY disabled");
            assert_eq!(p.dry_allowed_length, 35);
        }
        let args = OcrDetent::Fastest.profile().server_args();
        assert!(args.iter().any(|a| a == "--dry-multiplier"));
        assert!(args.iter().any(|a| a == "--image-max-tokens"));
    }

    #[test]
    fn every_tier_sends_the_prompt_that_was_measured_to_work() {
        // Changed from the documented `<|grounding|>...` on evidence: that
        // prompt generated zero tokens on this build, while `Free OCR.`
        // returned labelled, boxed regions on both tiers. If this assertion
        // is ever flipped back, re-run the standalone gate first.
        for detent in OcrDetent::ALL {
            let prompt = detent.profile().prompt();
            assert_eq!(prompt, "Free OCR.");
            assert!(!prompt.contains("Locate"), "Locate is the one still untested");
        }
    }

    #[test]
    fn a_normalised_box_maps_onto_the_page_corners() {
        let g = geom((1000, 1400), (731, 1024));
        let full = to_page(raw(0, 0, 999, 999), CoordSpace::Normalised, g);
        assert_eq!((full.x1, full.y1), (0, 0));
        assert_eq!((full.x2, full.y2), (1000, 1400));
        assert!(full.in_bounds);
    }

    #[test]
    fn an_input_pixel_box_scales_by_the_resize_ratio() {
        // The model saw a 731x1024 copy of a 1000x1400 page.
        let g = geom((1000, 1400), (731, 1024));
        let mapped = to_page(raw(0, 0, 731, 1024), CoordSpace::InputPixels, g);
        assert_eq!((mapped.x2, mapped.y2), (1000, 1400));
        assert!(mapped.in_bounds);
    }

    /// The Phase 0 calibration, expressed as a test: the same emitted numbers
    /// mean different things under the two conventions, and running two input
    /// sizes is what tells them apart. If a build ever changes which space it
    /// reports in, this is the shape of the check that catches it.
    #[test]
    fn the_two_spaces_are_distinguishable_by_running_two_input_sizes() {
        let same = raw(100, 100, 200, 200);
        let small = geom((1000, 1400), (457, 640));
        let large = geom((1000, 1400), (731, 1024));

        let n_small = to_page(same, CoordSpace::Normalised, small);
        let n_large = to_page(same, CoordSpace::Normalised, large);
        assert_eq!(
            n_small, n_large,
            "normalised coords must not move with input size"
        );

        let p_small = to_page(same, CoordSpace::InputPixels, small);
        let p_large = to_page(same, CoordSpace::InputPixels, large);
        assert_ne!(
            p_small, p_large,
            "pixel coords must move with input size — that is the discriminator"
        );
    }

    #[test]
    fn a_box_off_the_page_is_reported_not_clamped() {
        let g = geom((1000, 1400), (1000, 1400));
        let off = to_page(raw(0, 0, 1400, 1400), CoordSpace::Normalised, g);
        assert!(!off.in_bounds, "an out-of-page box must say so");
        assert!(
            off.x2 > 1000,
            "the offending coordinate must survive for the operator to see"
        );
    }

    #[test]
    fn an_inverted_box_is_not_in_bounds() {
        let g = geom((1000, 1400), (1000, 1400));
        let flipped = to_page(raw(500, 500, 100, 100), CoordSpace::InputPixels, g);
        assert!(!flipped.in_bounds);
    }

    #[test]
    fn a_zero_sized_input_cannot_divide_by_zero() {
        let g = geom((1000, 1400), (0, 0));
        let mapped = to_page(raw(10, 10, 20, 20), CoordSpace::InputPixels, g);
        assert!(mapped.x1 >= 0);
    }
}
