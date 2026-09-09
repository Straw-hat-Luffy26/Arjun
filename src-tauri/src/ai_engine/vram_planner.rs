//! GPU Offload Planning — deciding how many layers actually fit in VRAM.
//!
//! Replaces the heuristic previously inlined in `InferenceManager::build_load_config`,
//! which had three defects that all bias toward over-committing VRAM:
//!
//! 1. `vram_gb >= 6.0` forced full offload regardless of model size, so an 8 GB
//!    model on a 6 GB card requested every layer.
//! 2. Partial offload computed `ratio * 32.0`, hardcoding 32 layers. Real models
//!    range from 16 (Llama-3.2-1B) to 36 (Qwen3-4B) to 80 (70B).
//! 3. KV cache was ignored entirely. At 8K context it is often 1 GB+ — larger
//!    than the margin the old check reserved.
//!
//! On a 4 GB card the combination is decisive: a 2.5 GB model passes
//! `vram >= model_size + 1.0`, requests full offload, and then fails or silently
//! spills once the KV cache and compute buffers are allocated.
//!
//! ## Approximation
//!
//! Exact KV cache size needs `n_layer`, `n_head_kv`, and `head_dim` from GGUF
//! metadata, which is not parsed at load time. [`estimate_kv_bytes_per_token`]
//! substitutes a conservative size-banded estimate — deliberately biased high,
//! because under-offloading costs speed while over-offloading costs a crash.
//! `model_recommendation::estimator` has the exact GQA-aware formula for the
//! catalog path, where the metadata is available.

/// VRAM held back for the OS, desktop compositor, and other applications.
///
/// On Windows the desktop alone commonly holds 0.5–1.0 GB on a discrete GPU.
const OS_RESERVE_BYTES: u64 = 900 * 1024 * 1024;

/// Fraction of remaining VRAM reserved for llama.cpp compute buffers
/// (activations, scratch, allocator fragmentation).
const COMPUTE_OVERHEAD_FRACTION: f64 = 0.12;

/// Below this much usable VRAM, GPU offload is not worth attempting.
const MIN_USABLE_VRAM_BYTES: u64 = 512 * 1024 * 1024;

/// Layer count assumed when GGUF metadata is unavailable.
///
/// Only scales the *proportion* offloaded; llama.cpp clamps a too-large value
/// to the real layer count, so erring high is safe.
const ASSUMED_LAYERS_FALLBACK: u32 = 32;

/// Value meaning "offload everything" to llama.cpp.
pub const FULL_OFFLOAD: u32 = 999;

/// The offload decision, with the reasoning that produced it.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuOffloadPlan {
    /// Layers to request. 0 means CPU-only; [`FULL_OFFLOAD`] means all.
    pub gpu_layers: u32,
    /// Whether every layer is expected to fit.
    pub full_offload: bool,
    /// The context window this plan was actually computed for.
    ///
    /// Carried on the plan rather than read separately from the registry entry
    /// because the two must not be able to disagree. The layer count and the
    /// `--ctx-size` the server is launched with are two halves of one decision:
    /// planning for 8K and then serving 32K silently oversubscribes exactly the
    /// VRAM the plan reserved, which is how a full-offload plan ends up paging
    /// over PCIe.
    pub context_length: u32,
    /// Human-readable explanation for the load log.
    pub reason: String,
}

impl GpuOffloadPlan {
    fn cpu_only(context_length: u32, reason: impl Into<String>) -> Self {
        Self {
            gpu_layers: 0,
            full_offload: false,
            context_length,
            reason: reason.into(),
        }
    }
}

/// The smallest context worth serving.
///
/// Shrinking the window is how this module buys layers on the GPU, but it is
/// not free: the assistant reads scanned reports and drafts from them, and
/// below about this much it cannot hold a document and an answer at once. So
/// the search stops here even when a smaller window would fit more layers.
const MIN_SERVING_CONTEXT: u32 = 8192;

/// Context windows to consider, largest first, none above what was asked for.
///
/// The trained window is the ceiling and never exceeded; the ladder below it
/// exists because a window is worth less than the layers it costs. On an 8 GB
/// card a 5 GB model at 32K reserves 5 GB of KV cache and leaves 1.8 GB for
/// weights — a third of the layers on the GPU and the rest crossing PCIe for
/// every token. The same model at 8K fits entirely in VRAM.
fn context_ladder(requested: u32) -> Vec<u32> {
    let mut rungs: Vec<u32> = [requested, 32_768, 16_384, MIN_SERVING_CONTEXT]
        .into_iter()
        .filter(|rung| *rung <= requested && *rung > 0)
        .collect();
    rungs.sort_unstable_by(|a, b| b.cmp(a));
    rungs.dedup();
    // A model trained on less than the floor is served at what it was trained
    // for; the floor lowers windows, it never invents one.
    if rungs.is_empty() {
        rungs.push(requested.max(1));
    }
    rungs
}

/// Conservative KV-cache cost per token, in bytes.
///
/// Banded by model size as a proxy for `layers × kv_heads × head_dim`.
/// Reference points (FP16 KV, GQA):
/// - Llama-3.2-1B: 16 layers × 8 KV heads × 64 dim  →  32 KB/token
/// - Qwen3-4B:     36 layers × 8 KV heads × 128 dim → 144 KB/token
/// - Llama-3-8B:   32 layers × 8 KV heads × 128 dim → 128 KB/token
pub fn estimate_kv_bytes_per_token(model_bytes: u64) -> u64 {
    const GB: u64 = 1024 * 1024 * 1024;
    match model_bytes {
        b if b < GB => 32 * 1024,             // ~1B params
        b if b < 3 * GB => 64 * 1024,         // ~1–3B params
        b if b < 8 * GB => 160 * 1024,        // ~3–8B params (covers Qwen3-4B)
        _ => 256 * 1024,                      // larger
    }
}

/// Plans GPU offload for a model given real VRAM, model size, and context.
///
/// `total_layers` should come from GGUF metadata when known; `None` falls back
/// to [`ASSUMED_LAYERS_FALLBACK`].
pub fn plan_gpu_offload(
    vram_total_bytes: u64,
    model_bytes: u64,
    context_length: u32,
    total_layers: Option<u32>,
) -> GpuOffloadPlan {
    plan_gpu_offload_with(
        vram_total_bytes,
        model_bytes,
        ContextChoice::Planned(context_length),
        total_layers,
        None,
    )
}

/// Where the context window came from, which decides whether it may be lowered.
///
/// The ladder exists because a window is worth less than the GPU layers it
/// costs, and on an 8 GB card walking down from 32K to 8K is the difference
/// between a third of the model on the GPU and all of it. That trade is the
/// planner's to make only while nobody has made it: a window an operator typed
/// into Settings is an answer to the same question, already given.
///
/// Lowering it anyway is not a smaller version of honouring it. A person who
/// raised the window because their turns were refused for exceeding it gets the
/// identical refusal at the identical number, and nothing on screen says the
/// setting was overruled.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextChoice {
    /// Nobody chose. The planner may lower the window to buy GPU layers.
    Planned(u32),
    /// An operator set this window. It is served as asked, and the layer count
    /// is what gives.
    Fixed(u32),
}

/// Plans GPU offload, told where the window came from and what the KV cache
/// actually costs.
///
/// `kv_bytes_per_token` is the model's own geometry from its GGUF header —
/// `block_count * head_count_kv * (key_length + value_length) * 2`, which
/// [`super::gguf_meta::GgufMetadata::kv_bytes_per_token`] already computes and
/// the MoE planner already uses. `None` falls back to
/// [`estimate_kv_bytes_per_token`], the size band, for a caller that has not
/// read the header.
///
/// The band is deliberately high, and on a card with no room to spare that
/// conservatism is not free. Qwen3.5-9B is 32 layers x 4 KV heads x 512, so
/// 128 KB a token; the band charges it 160 KB. At 8 192 tokens that is 0.25 GB
/// of VRAM reserved for a cache that will never use it — and on a 7.9 GB card
/// holding a 5.02 GB model it was the whole difference between all 32 layers on
/// the GPU and 31, with the last one crossing PCIe for every token.
pub fn plan_gpu_offload_with(
    vram_total_bytes: u64,
    model_bytes: u64,
    context: ContextChoice,
    total_layers: Option<u32>,
    kv_bytes_per_token: Option<u64>,
) -> GpuOffloadPlan {
    let ladder = match context {
        // One rung, because the answer is already decided.
        ContextChoice::Fixed(window) => vec![window.max(1)],
        ContextChoice::Planned(window) => context_ladder(window),
    };

    // The largest window that still puts the whole model on the GPU.
    //
    // Tried in order rather than solved for, because `plan_at_context` is the
    // only thing that knows the whole cost model — KV, compute reserve and OS
    // reserve together — and a closed form here would be a second copy of it
    // that could drift.
    for rung in &ladder {
        let plan = plan_at_context(
            vram_total_bytes,
            model_bytes,
            *rung,
            total_layers,
            kv_bytes_per_token,
        );
        if plan.full_offload {
            return plan;
        }
    }

    // Nothing fits whole. Take the smallest window on the ladder anyway: the
    // model is crossing PCIe either way, and this is the rung that keeps the
    // most layers on the GPU while staying above `MIN_SERVING_CONTEXT`. A fixed
    // window has one rung, so this is that window and no reduction happens.
    let smallest = ladder.last().copied().unwrap_or(1);
    plan_at_context(
        vram_total_bytes,
        model_bytes,
        smallest,
        total_layers,
        kv_bytes_per_token,
    )
}

/// The offload arithmetic for one specific context window.
fn plan_at_context(
    vram_total_bytes: u64,
    model_bytes: u64,
    context_length: u32,
    total_layers: Option<u32>,
    kv_bytes_per_token: Option<u64>,
) -> GpuOffloadPlan {
    if vram_total_bytes == 0 {
        return GpuOffloadPlan::cpu_only(context_length, "No GPU VRAM detected");
    }

    // Usable VRAM after the OS and desktop take their share.
    let usable = vram_total_bytes.saturating_sub(OS_RESERVE_BYTES);
    if usable < MIN_USABLE_VRAM_BYTES {
        return GpuOffloadPlan::cpu_only(
            context_length,
            format!(
            "Only {:.2} GB usable VRAM after {:.2} GB OS reserve — below the {:.2} GB minimum",
            as_gb(usable),
            as_gb(OS_RESERVE_BYTES),
            as_gb(MIN_USABLE_VRAM_BYTES)
        ),
        );
    }

    // The model's own geometry where the caller read it, the size band where
    // nobody did. The band errs high on purpose; measured beats conservative
    // whenever the measurement exists.
    let kv_bytes = kv_bytes_per_token
        .filter(|per_token| *per_token > 0)
        .unwrap_or_else(|| estimate_kv_bytes_per_token(model_bytes))
        .saturating_mul(context_length.max(1) as u64);

    // KV cache and compute buffers are charged before any weights.
    let after_kv = usable.saturating_sub(kv_bytes);
    let compute_reserve = (after_kv as f64 * COMPUTE_OVERHEAD_FRACTION) as u64;
    let weight_budget = after_kv.saturating_sub(compute_reserve);

    if weight_budget == 0 {
        return GpuOffloadPlan::cpu_only(
            context_length,
            format!(
            "KV cache for {} tokens (~{:.2} GB) exhausts {:.2} GB usable VRAM",
            context_length,
            as_gb(kv_bytes),
            as_gb(usable)
        ),
        );
    }

    if weight_budget >= model_bytes {
        return GpuOffloadPlan {
            gpu_layers: FULL_OFFLOAD,
            full_offload: true,
            context_length,
            reason: format!(
                "Full offload: {:.2} GB weight budget covers the {:.2} GB model \
                 (VRAM {:.2} GB − {:.2} GB OS − {:.2} GB KV@{} − {:.2} GB compute)",
                as_gb(weight_budget),
                as_gb(model_bytes),
                as_gb(vram_total_bytes),
                as_gb(OS_RESERVE_BYTES),
                as_gb(kv_bytes),
                context_length,
                as_gb(compute_reserve)
            ),
        };
    }

    // Partial offload proportional to the fraction of weights that fit.
    let layers = total_layers.unwrap_or(ASSUMED_LAYERS_FALLBACK).max(1);
    let fraction = weight_budget as f64 / model_bytes as f64;
    let offload = (fraction * layers as f64).floor() as u32;

    if offload == 0 {
        return GpuOffloadPlan::cpu_only(
            context_length,
            format!(
            "Weight budget {:.2} GB covers under one layer of the {:.2} GB model",
            as_gb(weight_budget),
            as_gb(model_bytes)
        ),
        );
    }

    GpuOffloadPlan {
        gpu_layers: offload,
        full_offload: false,
        context_length,
        reason: format!(
            "Partial offload: {}/{} layers ({:.0}% of the {:.2} GB model fits in a \
             {:.2} GB weight budget after {:.2} GB KV@{})",
            offload,
            layers,
            fraction * 100.0,
            as_gb(model_bytes),
            as_gb(weight_budget),
            as_gb(kv_bytes),
            context_length
        ),
    }
}

// ─── MoE expert offload ─────────────────────────────────────────────────────

/// Extra layers of experts pushed to CPU beyond the computed minimum.
///
/// The offload depth is **not monotonic** in speed: performance is worst when
/// VRAM is oversubscribed, best at the smallest depth that genuinely fits, and
/// declines gently past that. The formula below targets that knee, and this
/// biases one layer to the safe side of it — for the same reason the module
/// already errs high on KV cache, an over-commit crashes while an under-commit
/// only costs throughput.
const MOE_SLACK_LAYERS: u32 = 1;

/// Host memory reserved beyond the expert weights themselves — activations,
/// the CPU-side scratch llama.cpp allocates for offloaded matmuls, and page
/// cache pressure from the mapped file.
const MOE_HOST_OVERHEAD_BYTES: u64 = 512 * 1024 * 1024;

/// Model geometry needed to place a MoE model, from
/// [`gguf_meta`](super::gguf_meta).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MoeGeometry {
    pub total_layers: u32,
    /// Routed-expert weights across all layers.
    pub expert_bytes: u64,
    /// Exact f16 KV cost, not the size-banded estimate — that bands on file
    /// size, which for MoE is dominated by experts rather than attention.
    pub kv_bytes_per_token: u64,
    /// Active parameters per token, for the reason string.
    pub active_params: u64,
}

/// Where a MoE model's weights go.
#[derive(Debug, Clone, PartialEq)]
pub struct MoeOffloadPlan {
    /// Always [`FULL_OFFLOAD`] when `fits`. A MoE model is split by *tensor*,
    /// not by layer: attention, KV cache, router and shared experts stay on the
    /// GPU while routed experts move, so reducing the layer count — the dense
    /// lever — would evict exactly the wrong things.
    pub gpu_layers: u32,
    /// N: routed experts of `blk.0` through `blk.(N-1)` are pinned to CPU.
    pub cpu_moe_layers: u32,
    /// False when even all-experts-on-CPU will not fit. The caller falls back
    /// to [`plan_gpu_offload`], which handles dense partial offload and CPU-only.
    pub fits: bool,
    pub reason: String,
}

impl MoeOffloadPlan {
    fn wont_fit(total_layers: u32, reason: impl Into<String>) -> Self {
        Self {
            gpu_layers: 0,
            cpu_moe_layers: total_layers,
            fits: false,
            reason: reason.into(),
        }
    }
}

/// Plans a MoE model's expert offload against real VRAM.
///
/// Returns the smallest number of layers whose routed experts must live in
/// system RAM for the rest of the model to fit on the card, plus one slack
/// layer. Everything else — attention, KV cache, router, shared experts — stays
/// resident, which is what makes a 21B MoE usable on a 4 GB card at all.
pub fn plan_moe_offload(
    model_id: &str,
    vram_total_bytes: u64,
    ram_available_bytes: u64,
    model_bytes: u64,
    context_length: u32,
    geom: &MoeGeometry,
) -> MoeOffloadPlan {
    let layers = geom.total_layers.max(1);

    if vram_total_bytes == 0 {
        return MoeOffloadPlan::wont_fit(layers, "No GPU VRAM detected");
    }
    if geom.expert_bytes == 0 {
        return MoeOffloadPlan::wont_fit(
            layers,
            "No routed-expert weights identified, so there is nothing to offload",
        );
    }

    // Offloaded experts have to live in system RAM, and on the machines this
    // targets that is usually the binding constraint rather than VRAM: a 21B
    // MoE's experts are ~12 GB, which a 16 GB laptop cannot hold however well
    // the VRAM side is planned. Without this check a plan that fits the card
    // is accepted and the host thrashes instead.
    let host_required = geom.expert_bytes.saturating_add(MOE_HOST_OVERHEAD_BYTES);
    if host_required > ram_available_bytes {
        return MoeOffloadPlan::wont_fit(
            layers,
            format!(
                "MoE offload needs {:.2} GB of system RAM for experts (+{:.2} GB overhead) \
                 but only {:.2} GB is usable — this machine is limited by RAM, not VRAM",
                as_gb(geom.expert_bytes),
                as_gb(MOE_HOST_OVERHEAD_BYTES),
                as_gb(ram_available_bytes)
            ),
        );
    }

    let usable = vram_total_bytes.saturating_sub(OS_RESERVE_BYTES);
    if usable < MIN_USABLE_VRAM_BYTES {
        return MoeOffloadPlan::wont_fit(
            layers,
            format!(
                "Only {:.2} GB usable VRAM after {:.2} GB OS reserve — below the {:.2} GB minimum",
                as_gb(usable),
                as_gb(OS_RESERVE_BYTES),
                as_gb(MIN_USABLE_VRAM_BYTES)
            ),
        );
    }

    let kv_bytes = geom
        .kv_bytes_per_token
        .saturating_mul(u64::from(context_length.max(1)));
    let after_kv = usable.saturating_sub(kv_bytes);
    let compute_reserve = (after_kv as f64 * COMPUTE_OVERHEAD_FRACTION) as u64;
    let weight_budget = after_kv.saturating_sub(compute_reserve);

    if weight_budget == 0 {
        return MoeOffloadPlan::wont_fit(
            layers,
            format!(
                "KV cache for {} tokens (~{:.2} GB) exhausts {:.2} GB usable VRAM",
                context_length,
                as_gb(kv_bytes),
                as_gb(usable)
            ),
        );
    }

    let headroom = format!(
        "VRAM {:.2} GB − {:.2} GB OS − {:.2} GB KV@{} − {:.2} GB compute = {:.2} GB weight budget",
        as_gb(vram_total_bytes),
        as_gb(OS_RESERVE_BYTES),
        as_gb(kv_bytes),
        context_length,
        as_gb(compute_reserve),
        as_gb(weight_budget)
    );

    // Everything fits as-is; no reason to send experts across PCIe.
    if weight_budget >= model_bytes {
        return MoeOffloadPlan {
            gpu_layers: FULL_OFFLOAD,
            cpu_moe_layers: 0,
            fits: true,
            reason: format!(
                "MoE resident: {} ({} active) fits entirely in VRAM, no experts offloaded; {headroom}",
                model_id,
                as_billions(geom.active_params)
            ),
        };
    }

    let per_layer = geom.expert_bytes / u64::from(layers);
    if per_layer == 0 {
        return MoeOffloadPlan::wont_fit(
            layers,
            format!(
                "Expert weights ({:.2} GB over {} layers) are too small to offload usefully",
                as_gb(geom.expert_bytes),
                layers
            ),
        );
    }

    let deficit = model_bytes - weight_budget;
    // Ceiling division: a partly-covered layer still has to move.
    let needed = deficit.div_ceil(per_layer);

    if needed > u64::from(layers) {
        return MoeOffloadPlan::wont_fit(
            layers,
            format!(
                "MoE offload insufficient: {} needs {:.2} GB off the card but all {} layers of \
                 experts are only {:.2} GB; {headroom}",
                model_id,
                as_gb(deficit),
                layers,
                as_gb(geom.expert_bytes)
            ),
        );
    }

    let n = (needed as u32 + MOE_SLACK_LAYERS).min(layers);

    MoeOffloadPlan {
        gpu_layers: FULL_OFFLOAD,
        cpu_moe_layers: n,
        fits: true,
        reason: format!(
            "MoE offload: {} ({} active) — experts of {}/{} layers to CPU ({:.2} GB), \
             {} layers resident; {headroom}",
            model_id,
            as_billions(geom.active_params),
            n,
            layers,
            as_gb(u64::from(n) * per_layer),
            layers - n
        ),
    }
}

/// Formats a parameter count as billions, e.g. `3.6B`.
fn as_billions(params: u64) -> String {
    if params == 0 {
        return "unknown".to_string();
    }
    format!("{:.1}B", params as f64 / 1e9)
}

/// Smallest context worth loading. Below this a model is not useful for coding.
const MIN_VIABLE_CONTEXT: u32 = 2048;

/// Largest context length this machine can afford for a model.
///
/// A model's advertised maximum (e.g. Qwen2.5's 32768) says what the *weights*
/// support, not what a given GPU can hold. The KV cache grows linearly with
/// context, so on a small card a full-length context can cost more memory than
/// the model itself. Requesting it anyway is a reliable way to run out of VRAM
/// mid-generation.
///
/// Returns `desired` unchanged when there is no GPU to budget against — in that
/// case the limit is system RAM, which is handled elsewhere.
pub fn max_affordable_context(
    vram_total_bytes: u64,
    model_bytes: u64,
    desired_context: u32,
) -> u32 {
    if vram_total_bytes == 0 || desired_context == 0 {
        return desired_context;
    }

    let usable = vram_total_bytes.saturating_sub(OS_RESERVE_BYTES);
    let compute_reserve = (usable as f64 * COMPUTE_OVERHEAD_FRACTION) as u64;

    // Whatever is left after weights and compute buffers is available for KV.
    let kv_budget = usable
        .saturating_sub(model_bytes)
        .saturating_sub(compute_reserve);

    if kv_budget == 0 {
        // Weights alone fill the card; llama.cpp will spill layers to CPU and
        // the KV for those lives in system RAM. Keep a small usable context
        // rather than reporting zero.
        return MIN_VIABLE_CONTEXT.min(desired_context);
    }

    let per_token = estimate_kv_bytes_per_token(model_bytes).max(1);
    let affordable = (kv_budget / per_token).min(u32::MAX as u64) as u32;

    affordable.clamp(MIN_VIABLE_CONTEXT.min(desired_context), desired_context)
}

fn as_gb(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0 * 1024.0)
}

#[cfg(test)]
mod tests {

    /// The regression that made a 9B model run at CPU speed on an 8 GB card.
    ///
    /// The trained window was read from GGUF (32768) and handed straight to
    /// both the planner and `--ctx-size`. At 160 KB/token that is 5 GB of KV
    /// cache on a card with about 7 GB usable, leaving 1.8 GB for a 5 GB model:
    /// roughly a third of the layers on the GPU and every token crossing PCIe.
    /// Task Manager showed the card full and 1% busy.
    ///
    /// The window is worth less than the layers it costs, so it gives way.
    #[test]
    fn a_trained_window_gives_way_to_getting_the_model_onto_the_gpu() {
        const GB: u64 = 1024 * 1024 * 1024;
        // Qwen3.5-9B-Q4_K_S on an RTX 5060 Laptop.
        let vram = 8151 * 1024 * 1024;
        let model = 5 * GB;

        let plan = plan_gpu_offload(vram, model, 32_768, Some(40));

        assert!(
            plan.full_offload,
            "the whole model fits at a smaller window and must be put there: {}",
            plan.reason
        );
        assert_eq!(plan.gpu_layers, FULL_OFFLOAD);
        assert!(
            plan.context_length < 32_768,
            "the window had to shrink for this to fit"
        );
        assert!(
            plan.context_length >= MIN_SERVING_CONTEXT,
            "shrunk past the floor to {} — below this the assistant cannot hold              a document and an answer at once",
            plan.context_length
        );
    }

    /// A window is never invented, only lowered.
    ///
    /// The ceiling is what the model was trained for; serving beyond it is
    /// garbage output, which no amount of free VRAM makes acceptable.
    #[test]
    fn a_small_trained_window_is_never_raised_to_fill_spare_vram() {
        const GB: u64 = 1024 * 1024 * 1024;
        let plan = plan_gpu_offload(24 * GB, 2 * GB, 4096, Some(32));

        assert!(plan.full_offload);
        assert_eq!(
            plan.context_length, 4096,
            "a model trained on 4K must be served at 4K however much VRAM is spare"
        );
    }

    /// A model that fits at its full window keeps it.
    #[test]
    fn a_window_that_already_fits_is_left_alone() {
        const GB: u64 = 1024 * 1024 * 1024;
        let plan = plan_gpu_offload(24 * GB, 4 * GB, 32_768, Some(32));

        assert!(plan.full_offload);
        assert_eq!(
            plan.context_length, 32_768,
            "nothing was gained by shrinking, so nothing should have been taken"
        );
    }

    /// When the model cannot fit at any window, the layer count still wins.
    ///
    /// The run is crossing PCIe whatever happens; the rung that keeps the most
    /// layers on the GPU is the least bad, and it is still above the floor.
    #[test]
    fn a_model_too_large_for_the_card_is_planned_at_the_smallest_window() {
        const GB: u64 = 1024 * 1024 * 1024;
        let plan = plan_gpu_offload(8 * GB, 20 * GB, 32_768, Some(48));

        assert!(!plan.full_offload, "20 GB cannot fit in 8 GB at any window");
        assert_eq!(plan.context_length, MIN_SERVING_CONTEXT);
    }
    use super::*;

    /// This laptop, with the model's own KV geometry.
    ///
    /// An RTX 5060 Laptop reports 7 899 MB; Qwen3.5-9B Q4_K_S is 5.02 GiB of
    /// weights over 32 layers, 4 KV heads and 256-wide keys and values, so
    /// 128 KB a token. The size band charges it 160 KB, and at 8 192 tokens
    /// that 25% margin is 0.25 GB — enough, on a card with nothing spare, to
    /// put the budget under the model and leave the last layer on the CPU.
    /// The machine that reported this loaded at `gpu_layers=31`.
    #[test]
    fn the_real_kv_geometry_buys_the_last_layer_the_size_band_costs() {
        const VRAM: u64 = 7899 * 1024 * 1024;
        const WEIGHTS: u64 = 5_394_097_376;
        const REAL_KV: u64 = 32 * 4 * (256 + 256) * 2;

        let banded = plan_gpu_offload(VRAM, WEIGHTS, 8192, Some(32));
        assert!(
            !banded.full_offload,
            "the size band is what left a layer on the CPU: {}",
            banded.reason
        );

        let measured = plan_gpu_offload_with(
            VRAM,
            WEIGHTS,
            ContextChoice::Planned(8192),
            Some(32),
            Some(REAL_KV),
        );
        assert!(
            measured.full_offload,
            "the model's own geometry fits the whole model at the same window: {}",
            measured.reason
        );
        assert_eq!(measured.context_length, 8192, "and does not buy it by shrinking the window");
    }

    /// A window an operator set is not walked back down.
    ///
    /// The ladder trades window for layers, which is the right trade to make
    /// for somebody who has not made it themselves. Settings writes a per-model
    /// Context Length; a planner that lowered it anyway would hand back the
    /// same `exceeds the available context size` refusal at the same number the
    /// person raised it above.
    #[test]
    fn a_window_the_operator_set_is_served_at_the_size_they_set() {
        const VRAM: u64 = 7899 * 1024 * 1024;
        const WEIGHTS: u64 = 5_394_097_376;

        let planned = plan_gpu_offload(VRAM, WEIGHTS, 32_768, Some(32));
        assert_eq!(
            planned.context_length, 8192,
            "unchosen, the ladder walks down to buy layers"
        );

        let fixed = plan_gpu_offload_with(
            VRAM,
            WEIGHTS,
            ContextChoice::Fixed(32_768),
            Some(32),
            None,
        );
        assert_eq!(
            fixed.context_length, 32_768,
            "chosen, it is served as asked"
        );
        assert!(
            !fixed.full_offload,
            "and the cost is layers on the CPU, reported rather than hidden"
        );
        assert!(
            fixed.gpu_layers < 32,
            "which is what a window this large costs on this card: {}",
            fixed.reason
        );
    }

    const GB: u64 = 1024 * 1024 * 1024;

    #[test]
    fn no_vram_means_cpu_only() {
        let plan = plan_gpu_offload(0, 2 * GB, 4096, None);
        assert_eq!(plan.gpu_layers, 0);
    }

    #[test]
    fn tiny_gpus_fall_back_to_cpu() {
        // 1 GB card: nothing usable after the OS reserve.
        let plan = plan_gpu_offload(GB, 2 * GB, 4096, None);
        assert_eq!(plan.gpu_layers, 0);
        assert!(plan.reason.contains("usable VRAM"));
    }

    #[test]
    fn regression_4gb_card_does_not_full_offload_a_25gb_model() {
        // The exact case the old heuristic got wrong: 4 GB RTX 3050 laptop.
        // `vram_gb >= model_size_gb + 1.0` → 4.0 >= 3.5 → requested all layers,
        // ignoring ~1.3 GB of KV cache at 8K context.
        let plan = plan_gpu_offload(4 * GB, 2560 * 1024 * 1024, 8192, Some(36));

        assert!(
            !plan.full_offload,
            "must not claim full offload on a 4 GB card: {}",
            plan.reason
        );
        assert!(plan.gpu_layers < FULL_OFFLOAD);
    }

    #[test]
    fn regression_large_vram_does_not_blindly_full_offload() {
        // Old rule: `vram_gb >= 6.0` → full offload regardless of model size.
        // A 20 GB model on an 8 GB card must not request every layer.
        let plan = plan_gpu_offload(8 * GB, 20 * GB, 8192, Some(80));

        assert!(!plan.full_offload, "20 GB model cannot fit 8 GB VRAM");
        assert!(plan.gpu_layers > 0 && plan.gpu_layers < 80);
    }

    #[test]
    fn a_roomy_card_offloads_everything() {
        let plan = plan_gpu_offload(24 * GB, 4 * GB, 8192, Some(36));

        assert!(plan.full_offload);
        assert_eq!(plan.gpu_layers, FULL_OFFLOAD);
    }

    #[test]
    fn context_length_reduces_offloadable_layers() {
        let short = plan_gpu_offload(8 * GB, 6 * GB, 2048, Some(32));
        let long = plan_gpu_offload(8 * GB, 6 * GB, 32768, Some(32));

        assert!(
            long.gpu_layers <= short.gpu_layers,
            "a larger KV cache must not permit more layers ({} vs {})",
            long.gpu_layers,
            short.gpu_layers
        );
    }

    #[test]
    fn layer_count_is_respected_rather_than_assumed_32() {
        // Same memory pressure, different architectures.
        let sixteen = plan_gpu_offload(6 * GB, 8 * GB, 4096, Some(16));
        let eighty = plan_gpu_offload(6 * GB, 8 * GB, 4096, Some(80));

        assert!(
            eighty.gpu_layers > sixteen.gpu_layers,
            "an 80-layer model should offload more layers than a 16-layer one at \
             the same fitting fraction ({} vs {})",
            eighty.gpu_layers,
            sixteen.gpu_layers
        );
        assert!(sixteen.gpu_layers <= 16, "cannot offload more layers than exist");
    }

    #[test]
    fn kv_estimate_grows_with_model_size() {
        assert!(
            estimate_kv_bytes_per_token(512 * 1024 * 1024)
                < estimate_kv_bytes_per_token(4 * GB)
        );
        assert!(estimate_kv_bytes_per_token(4 * GB) < estimate_kv_bytes_per_token(20 * GB));
    }

    #[test]
    fn a_4gb_card_cannot_afford_a_32k_context() {
        // The exact case the certified runtime profile got wrong: it hardcoded
        // context_length 32768 regardless of the machine.
        let affordable = max_affordable_context(4 * GB, 2560 * 1024 * 1024, 32768);

        assert!(
            affordable < 32768,
            "4 GB card must not be told it can hold a 32k context, got {affordable}"
        );
        assert!(affordable >= MIN_VIABLE_CONTEXT, "must stay usable, got {affordable}");
    }

    #[test]
    fn a_large_card_keeps_the_full_context() {
        let affordable = max_affordable_context(24 * GB, 4 * GB, 32768);
        assert_eq!(affordable, 32768, "a 24 GB card can afford the model's full context");
    }

    #[test]
    fn affordable_context_never_exceeds_what_was_asked_for() {
        // Even with enormous VRAM, we never invent context the model lacks.
        assert_eq!(max_affordable_context(80 * GB, GB, 8192), 8192);
    }

    #[test]
    fn no_gpu_leaves_the_request_untouched() {
        // CPU-only is bounded by system RAM, handled elsewhere.
        assert_eq!(max_affordable_context(0, 4 * GB, 16384), 16384);
    }

    #[test]
    fn context_shrinks_as_the_model_grows() {
        let small = max_affordable_context(8 * GB, GB, 32768);
        let large = max_affordable_context(8 * GB, 6 * GB, 32768);

        assert!(
            large <= small,
            "a bigger model leaves less room for KV cache ({large} vs {small})"
        );
    }

    #[test]
    fn plans_never_panic_across_extremes() {
        let cases = [
            (0u64, 0u64, 0u32),
            (u64::MAX, u64::MAX, u32::MAX),
            (4 * GB, 1, 1),
            (1, 4 * GB, 128_000),
        ];

        for (vram, model, ctx) in cases {
            let plan = plan_gpu_offload(vram, model, ctx, None);
            assert!(plan.gpu_layers == 0 || plan.gpu_layers > 0);
        }
    }

    // ─── MoE expert offload ─────────────────────────────────────────────────

    /// gpt-oss-20b: 24 layers, 21B total / 3.6B active, routed experts about
    /// 93% of a 12.8 GB MXFP4 file, and 24 × 8 × (64+64) × 2 = 49,152 bytes of
    /// KV per token.
    fn gpt_oss_geometry() -> MoeGeometry {
        MoeGeometry {
            total_layers: 24,
            expert_bytes: 11_904_000_000,
            kv_bytes_per_token: 49_152,
            active_params: 3_600_000_000,
        }
    }

    const GPT_OSS_BYTES: u64 = 12_800_000_000;
    const WORKING_CONTEXT: u32 = 8192;
    /// Enough host RAM to hold gpt-oss-20b's ~11.9 GB of experts, so VRAM is the
    /// variable under test rather than the RAM gate.
    const ROOMY_RAM: u64 = 24 * GB;

    fn plan_gpt_oss_on(vram: u64) -> MoeOffloadPlan {
        plan_gpt_oss_with(vram, ROOMY_RAM)
    }

    fn plan_gpt_oss_with(vram: u64, ram: u64) -> MoeOffloadPlan {
        plan_moe_offload(
            "gpt-oss-20b",
            vram,
            ram,
            GPT_OSS_BYTES,
            WORKING_CONTEXT,
            &gpt_oss_geometry(),
        )
    }

    /// The minimum-spec target: an RTX 3050 4 GB must run a 12.8 GB MoE model.
    #[test]
    fn the_4gb_minimum_spec_fits_a_21b_moe() {
        let plan = plan_gpt_oss_on(4 * GB);

        assert!(plan.fits, "{}", plan.reason);
        assert_eq!(plan.cpu_moe_layers, 22, "{}", plan.reason);
        assert_eq!(
            plan.gpu_layers, FULL_OFFLOAD,
            "MoE splits by tensor, so every layer still goes to the GPU"
        );
    }

    #[test]
    fn an_8gb_card_keeps_more_experts_resident_than_a_4gb_one() {
        let small = plan_gpt_oss_on(4 * GB);
        let large = plan_gpt_oss_on(8 * GB);

        assert_eq!(large.cpu_moe_layers, 14, "{}", large.reason);
        assert!(
            large.cpu_moe_layers < small.cpu_moe_layers,
            "more VRAM must mean fewer experts on CPU ({} vs {})",
            large.cpu_moe_layers,
            small.cpu_moe_layers
        );
    }

    #[test]
    fn a_roomy_card_offloads_nothing() {
        let plan = plan_gpt_oss_on(24 * GB);

        assert!(plan.fits);
        assert_eq!(plan.cpu_moe_layers, 0, "{}", plan.reason);
        assert_eq!(plan.gpu_layers, FULL_OFFLOAD);
    }

    /// The whole point of the MoE branch: the dense lever (fewer layers) would
    /// evict attention and KV cache, which is exactly what must stay resident.
    #[test]
    fn a_fitting_plan_never_reduces_the_layer_count() {
        for vram in [3 * GB, 4 * GB, 6 * GB, 8 * GB, 12 * GB, 24 * GB] {
            let plan = plan_gpt_oss_on(vram);
            if plan.fits {
                assert_eq!(
                    plan.gpu_layers, FULL_OFFLOAD,
                    "at {:.0} GB: {}",
                    as_gb(vram),
                    plan.reason
                );
            }
        }
    }

    #[test]
    fn offload_depth_never_exceeds_the_layers_that_exist() {
        for vram in [GB, 2 * GB, 3 * GB, 4 * GB, 8 * GB] {
            let plan = plan_gpt_oss_on(vram);
            assert!(
                plan.cpu_moe_layers <= gpt_oss_geometry().total_layers,
                "at {:.0} GB got depth {}",
                as_gb(vram),
                plan.cpu_moe_layers
            );
        }
    }

    #[test]
    fn a_model_too_large_even_with_every_expert_on_cpu_does_not_fit() {
        // Non-expert weights alone dwarf the card.
        let plan = plan_moe_offload(
            "huge-moe",
            4 * GB,
            ROOMY_RAM,
            100 * GB,
            WORKING_CONTEXT,
            &gpt_oss_geometry(),
        );

        assert!(!plan.fits, "{}", plan.reason);
        assert!(plan.reason.contains("insufficient"), "got: {}", plan.reason);
    }

    /// The real limit on the 4 GB minimum spec: a 16 GB laptop cannot hold
    /// ~12 GB of experts however well the card is planned. The message has to
    /// name RAM, because "buy a bigger GPU" would be the wrong conclusion.
    #[test]
    fn a_machine_without_ram_for_the_experts_is_refused_and_says_so() {
        let plan = plan_gpt_oss_with(4 * GB, 6 * GB);

        assert!(!plan.fits, "{}", plan.reason);
        assert!(plan.reason.contains("system RAM"), "got: {}", plan.reason);
        assert!(plan.reason.contains("limited by RAM"), "got: {}", plan.reason);
    }

    #[test]
    fn the_same_card_succeeds_once_there_is_ram_for_the_experts() {
        assert!(!plan_gpt_oss_with(4 * GB, 6 * GB).fits);
        assert!(plan_gpt_oss_with(4 * GB, 24 * GB).fits, "RAM was the only blocker");
    }

    /// Mind the units: `GB` in this file is a GiB (1024³), while the expert
    /// figure is decimal bytes. 11.90 GB of experts is 11.09 GiB, and the host
    /// overhead adds 0.5 GiB — so 11 GiB is short and 12 GiB is not.
    #[test]
    fn ram_just_short_of_the_experts_is_refused_and_just_enough_is_not() {
        let short = plan_gpt_oss_with(4 * GB, 11 * GB);
        assert!(!short.fits, "11 GiB cannot hold 11.09 GiB of experts plus overhead");

        assert!(
            plan_gpt_oss_with(4 * GB, 12 * GB).fits,
            "12 GiB does cover them, so the gate must not be over-eager"
        );
    }

    #[test]
    fn no_gpu_cannot_offload_experts() {
        let plan = plan_gpt_oss_on(0);

        assert!(!plan.fits);
        assert!(plan.reason.contains("No GPU"), "got: {}", plan.reason);
    }

    #[test]
    fn a_dense_model_has_nothing_to_offload() {
        let geom = MoeGeometry {
            expert_bytes: 0,
            ..gpt_oss_geometry()
        };
        let plan = plan_moe_offload("llama-8b", 4 * GB, ROOMY_RAM, 5 * GB, WORKING_CONTEXT, &geom);

        assert!(!plan.fits);
        assert!(plan.reason.contains("nothing to offload"), "got: {}", plan.reason);
    }

    /// The reason string is the only record of why a machine got the depth it
    /// did, and it has to be auditable the same way the dense plans are.
    #[test]
    fn the_reason_names_the_model_the_active_params_and_the_depth() {
        let plan = plan_gpt_oss_on(4 * GB);

        assert!(plan.reason.contains("gpt-oss-20b"), "got: {}", plan.reason);
        assert!(plan.reason.contains("3.6B active"), "got: {}", plan.reason);
        assert!(plan.reason.contains("22/24"), "got: {}", plan.reason);
        assert!(plan.reason.contains("weight budget"), "got: {}", plan.reason);
        assert!(plan.reason.contains("KV@8192"), "got: {}", plan.reason);
    }

    /// Over-committing VRAM crashes; under-committing only costs throughput.
    #[test]
    fn the_depth_is_biased_past_the_bare_minimum() {
        let plan = plan_gpt_oss_on(4 * GB);
        let geom = gpt_oss_geometry();
        let per_layer = geom.expert_bytes / u64::from(geom.total_layers);

        let moved = u64::from(plan.cpu_moe_layers) * per_layer;
        let deficit = GPT_OSS_BYTES - 2_594_764_227; // weight budget on a 4 GB card

        assert!(
            moved > deficit,
            "must move strictly more than the deficit ({moved} vs {deficit})"
        );
    }

    #[test]
    fn a_longer_context_pushes_more_experts_off_the_card() {
        let short = plan_moe_offload("m", 8 * GB, ROOMY_RAM, GPT_OSS_BYTES, 2048, &gpt_oss_geometry());
        let long = plan_moe_offload("m", 8 * GB, ROOMY_RAM, GPT_OSS_BYTES, 32768, &gpt_oss_geometry());

        assert!(
            long.cpu_moe_layers >= short.cpu_moe_layers,
            "a larger KV cache leaves less room for experts ({} vs {})",
            long.cpu_moe_layers,
            short.cpu_moe_layers
        );
    }

    #[test]
    fn moe_plans_never_panic_across_extremes() {
        let geometries = [
            gpt_oss_geometry(),
            MoeGeometry { total_layers: 0, expert_bytes: 0, kv_bytes_per_token: 0, active_params: 0 },
            MoeGeometry {
                total_layers: u32::MAX,
                expert_bytes: u64::MAX,
                kv_bytes_per_token: u64::MAX,
                active_params: u64::MAX,
            },
        ];
        let cases = [(0u64, 0u64, 0u32), (u64::MAX, u64::MAX, u32::MAX), (4 * GB, 1, 1), (1, 4 * GB, 128_000)];

        for geom in &geometries {
            for (vram, model, ctx) in cases {
                for ram in [0u64, ROOMY_RAM, u64::MAX] {
                    let plan = plan_moe_offload("x", vram, ram, model, ctx, geom);
                    assert!(plan.cpu_moe_layers <= geom.total_layers.max(1));
                }
            }
        }
    }
}
