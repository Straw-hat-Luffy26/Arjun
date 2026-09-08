//! Phase 3: Memory & Fit Estimator
//!
//! Calculates exact memory requirements for a model at a given quantization
//! and context length using documented, deterministic formulas.
//!
//! ## Formulas
//!
//! ### Weight Memory
//! `weight_bytes = total_parameters × bits_per_weight / 8`
//!
//! For MoE models, `total_parameters` includes ALL experts, because every
//! weight must be resident *somewhere*. Where it is resident is a separate
//! question: [`split_moe_weights`] divides them into the part that has to be in
//! VRAM and the routed experts llama.cpp can pin to system RAM.
//!
//! ### KV Cache Memory (GQA-Aware)
//! `kv_cache_bytes = 2 × layers × kv_heads × head_dim × context × bytes_per_element`
//!
//! Where:
//! - 2: Key and Value tensors
//! - layers: Number of transformer layers
//! - kv_heads: Number of KV heads (fewer than query heads for GQA/MQA)
//! - head_dim: Per-head dimension
//! - context: Sequence length in tokens
//! - bytes_per_element: 2 for FP16 KV cache (standard in llama.cpp/Ollama/vLLM)
//!
//! ### Runtime/Compute Overhead
//! `overhead_bytes = (weight_bytes + kv_cache_bytes) × overhead_factor`
//!
//! The overhead factor (default 12%) is a configurable conservative heuristic
//! accounting for:
//! - CUDA/Metal/Vulkan context initialization buffers
//! - GGUF metadata and tensor index structures
//! - Activation tensors during forward pass
//! - Memory allocator fragmentation
//!
//! Future runtime measurements can calibrate/replace this value.
//!
//! ### Total Memory
//! `total = weight_bytes + kv_cache_bytes + overhead_bytes`
//!
//! ## Assumptions
//! - KV cache uses FP16 precision regardless of weight quantization
//! - Batch size is 1 (single-user local inference)
//! - MoE: every expert weight is resident in *some* memory tier — see
//!   [`split_moe_weights`] for the VRAM/RAM division; routing overhead negligible

use crate::model_recommendation::traits::*;

/// Estimate weight memory in bytes for a model at a given quantization.
pub fn estimate_weight_memory(model: &ModelMetadata, quant: &QuantizationSpec) -> u64 {
    // For MoE models, ALL parameters (all experts) must be loaded
    let params = model.total_parameters;
    // weight_bytes = params × bits_per_weight / 8
    (params as f64 * quant.bits_per_weight / 8.0) as u64
}

/// How a MoE model's weights divide between the memory that must be on the GPU
/// and the routed experts that can live in system RAM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MoeWeightSplit {
    /// Attention, router, shared experts, embeddings and norms. These stay in
    /// VRAM — they are latency-critical and small.
    pub resident_bytes: u64,
    /// Routed experts, which llama.cpp can pin to system RAM per layer.
    pub expert_bytes: u64,
}

/// Ceiling on the share of weights attributed to routed experts, so a figure
/// that disagrees with the stated totals cannot claim the whole model is
/// offloadable and leave nothing resident.
const MAX_EXPERT_FRACTION: f64 = 0.95;

/// Splits a MoE model's weights into resident and offloadable parts.
///
/// A MoE model does **not** offload proportionally the way a dense one does.
/// Only the routed experts move; attention and the KV cache have to stay on the
/// card. Treating it proportionally overstates the VRAM a machine needs and
/// misreports where the memory actually goes.
///
/// The expert share is solved from figures the catalog already carries. With
/// `D` non-expert parameters and `E` routed-expert parameters:
///
/// ```text
/// D + E                       = total
/// D + E × (active/num experts) = active
/// ⇒ E = (total − active) / (1 − active_experts/num_experts)
/// ```
///
/// Returns `None` for dense models, or when the figures cannot produce a
/// sensible split — the caller then falls back to the proportional estimate.
pub fn split_moe_weights(model: &ModelMetadata, weight_bytes: u64) -> Option<MoeWeightSplit> {
    let (num_experts, active_experts) = match model.architecture {
        ModelArchitecture::MixtureOfExperts { num_experts, active_experts } => {
            (num_experts, active_experts)
        }
        _ => return None,
    };

    if num_experts == 0 || active_experts == 0 || active_experts >= num_experts {
        return None;
    }

    let total = model.total_parameters as f64;
    let active = model.active_parameters? as f64;
    if total <= 0.0 || active <= 0.0 || active >= total {
        return None;
    }

    let used_share = f64::from(active_experts) / f64::from(num_experts);
    let expert_params = (total - active) / (1.0 - used_share);
    if expert_params <= 0.0 {
        return None;
    }

    let expert_fraction = (expert_params / total).min(MAX_EXPERT_FRACTION);
    let expert_bytes = (weight_bytes as f64 * expert_fraction) as u64;

    Some(MoeWeightSplit {
        resident_bytes: weight_bytes.saturating_sub(expert_bytes),
        expert_bytes,
    })
}

/// Estimate KV cache memory in bytes for a given context length.
/// Uses GQA-aware formula: 2 × L × H_kv × D_head × T × E
pub fn estimate_kv_cache_memory(model: &ModelMetadata, context_length: u32, config: &EstimatorConfig) -> u64 {
    let layers = model.num_layers as u64;
    let kv_heads = model.num_kv_heads as u64;
    let head_dim = model.head_dimension as u64;
    let context = context_length as u64;
    let bytes_per_element = config.kv_cache_bytes_per_element as u64;
    let batch_size = config.batch_size as u64;

    // 2 × L × H_kv × D_head × T × B × E
    2 * layers * kv_heads * head_dim * context * batch_size * bytes_per_element
}

/// Estimate total memory requirement for a model configuration.
/// Returns (weight_bytes, kv_cache_bytes, overhead_bytes, total_bytes).
pub fn estimate_total_memory(
    model: &ModelMetadata,
    quant: &QuantizationSpec,
    context_length: u32,
    config: &EstimatorConfig,
) -> (u64, u64, u64, u64) {
    let weight_bytes = estimate_weight_memory(model, quant);
    let kv_cache_bytes = estimate_kv_cache_memory(model, context_length, config);
    let proportional_overhead = ((weight_bytes + kv_cache_bytes) as f64 * config.overhead_factor) as u64;
    let overhead_bytes = proportional_overhead.max(config.min_overhead_bytes);
    let total = weight_bytes + kv_cache_bytes + overhead_bytes;
    (weight_bytes, kv_cache_bytes, overhead_bytes, total)
}

/// Standard quantization levels ordered from highest to lowest quality.
pub fn quantization_hierarchy() -> Vec<QuantizationSpec> {
    vec![
        QuantizationSpec { label: "Q8_0".to_string(),    bits_per_weight: 8.5,  quality_rank: 8 },
        QuantizationSpec { label: "Q6_K".to_string(),    bits_per_weight: 6.5,  quality_rank: 7 },
        QuantizationSpec { label: "Q5_K_M".to_string(),  bits_per_weight: 5.5,  quality_rank: 6 },
        QuantizationSpec { label: "Q4_K_M".to_string(),  bits_per_weight: 4.85, quality_rank: 5 },
        QuantizationSpec { label: "Q4_0".to_string(),    bits_per_weight: 4.5,  quality_rank: 4 },
        QuantizationSpec { label: "Q3_K_M".to_string(),  bits_per_weight: 3.5,  quality_rank: 3 },
        QuantizationSpec { label: "Q2_K".to_string(),    bits_per_weight: 2.5,  quality_rank: 2 },
    ]
}

// ─── Unit Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_llama31_8b() -> ModelMetadata {
        ModelMetadata {
            id: "meta-llama/Llama-3.1-8B".to_string(),
            name: "Llama 3.1 8B".to_string(),
            family: "Llama 3.1".to_string(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 8_030_000_000,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 131072,
            vocab_size: 128256,
            default_dtype: "bf16".to_string(),
            use_cases: vec!["chat".to_string(), "general".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    fn make_mixtral_8x7b() -> ModelMetadata {
        ModelMetadata {
            id: "mistralai/Mixtral-8x7B-v0.1".to_string(),
            name: "Mixtral 8×7B".to_string(),
            family: "Mixtral".to_string(),
            architecture: ModelArchitecture::MixtureOfExperts {
                num_experts: 8,
                active_experts: 2,
            },
            total_parameters: 46_700_000_000,
            active_parameters: Some(12_900_000_000),
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 8,
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 32768,
            vocab_size: 32000,
            default_dtype: "bf16".to_string(),
            use_cases: vec!["chat".to_string(), "general".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    /// gpt-oss-20b: 21B total, 3.6B active, 4 of 32 experts per token.
    fn make_gpt_oss_20b() -> ModelMetadata {
        ModelMetadata {
            id: "openai/gpt-oss-20b".to_string(),
            name: "gpt-oss 20B".to_string(),
            family: "gpt-oss".to_string(),
            architecture: ModelArchitecture::MixtureOfExperts {
                num_experts: 32,
                active_experts: 4,
            },
            total_parameters: 20_900_000_000,
            active_parameters: Some(3_600_000_000),
            num_layers: 24,
            num_attention_heads: 64,
            num_kv_heads: 8,
            head_dimension: 64,
            hidden_size: 2880,
            max_context_length: 131072,
            vocab_size: 201088,
            default_dtype: "mxfp4".to_string(),
            use_cases: vec!["code".to_string(), "chat".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    // ─── MoE weight split ───────────────────────────────────────────────────

    /// The point of the split: a 21B MoE needs only a small resident part in
    /// VRAM, which is what lets it run on a 4 GB card at all.
    #[test]
    fn moe_weights_split_into_a_small_resident_part_and_large_experts() {
        let model = make_gpt_oss_20b();
        let weight_bytes = 12_800_000_000u64;

        let split = split_moe_weights(&model, weight_bytes).expect("gpt-oss is MoE");

        assert!(
            split.resident_bytes < weight_bytes / 5,
            "attention and router should be a small share, got {} of {weight_bytes}",
            split.resident_bytes
        );
        assert!(split.expert_bytes > split.resident_bytes * 4);
    }

    #[test]
    fn a_moe_split_accounts_for_every_weight_byte() {
        for model in [make_gpt_oss_20b(), make_mixtral_8x7b()] {
            let weight_bytes = 12_800_000_000u64;
            let split = split_moe_weights(&model, weight_bytes).unwrap();

            assert_eq!(
                split.resident_bytes + split.expert_bytes,
                weight_bytes,
                "{} lost or invented bytes",
                model.id
            );
        }
    }

    /// Solved from the catalog figures, so it must reproduce the stated active
    /// parameter count rather than using a fixed guess.
    #[test]
    fn the_expert_share_reproduces_the_stated_active_parameters() {
        let model = make_gpt_oss_20b();
        let split = split_moe_weights(&model, model.total_parameters).unwrap();

        // resident + experts × (4/32) should land back on 3.6B.
        let implied_active = split.resident_bytes + split.expert_bytes / 8;
        let stated = model.active_parameters.unwrap();
        let drift = implied_active.abs_diff(stated) as f64 / stated as f64;

        assert!(drift < 0.05, "implied {implied_active} vs stated {stated}");
    }

    /// The recommendation and the load must describe the same model.
    ///
    /// The scorer sizes a model from catalog metadata (total and active
    /// parameters); the loader sizes the same model from its GGUF header (layer
    /// count and expert dimensions). Those are independent derivations of the
    /// same quantity, and if they drift the user is shown one VRAM/RAM split
    /// and gets another.
    #[test]
    fn catalog_and_gguf_geometry_agree_on_the_expert_share() {
        use crate::ai_engine::gguf_meta::GgufMetadata;

        let file_bytes = 12_800_000_000u64;

        // Recommend-time: solved from total and active parameters.
        let from_catalog = split_moe_weights(&make_gpt_oss_20b(), file_bytes)
            .expect("gpt-oss is MoE")
            .expert_bytes;

        // Load-time: computed from the header geometry verified against
        // openai/gpt-oss-20b's published config.json.
        let from_header = GgufMetadata {
            architecture: "gpt-oss".to_string(),
            emits_reasoning: false,
            block_count: 24,
            embedding_length: 2880,
            expert_count: 32,
            expert_used_count: 4,
            expert_ff_length: 2880,
            head_count_kv: 8,
            key_length: 64,
            value_length: 64,
            parameter_count: Some(20_900_000_000),
            // Irrelevant to expert-weight arithmetic; this test compares only
            // that.
            context_length: None,
            supports_toggled_reasoning: false,
        }
        .expert_bytes(file_bytes, None);

        let drift = from_catalog.abs_diff(from_header) as f64 / from_header as f64;
        assert!(
            drift < 0.10,
            "the two paths must not disagree by more than 10%: catalog {from_catalog} vs header {from_header} ({:.1}%)",
            drift * 100.0
        );
    }

    #[test]
    fn a_dense_model_has_no_expert_split() {
        assert!(split_moe_weights(&make_llama31_8b(), 8_000_000_000).is_none());
        assert!(split_moe_weights(&make_codelama_7b(), 7_000_000_000).is_none());
    }

    #[test]
    fn a_moe_model_without_active_parameters_cannot_be_split() {
        // The live HF catalog records these as Dense precisely because the Hub
        // does not expose expert counts; a partial record must not be guessed at.
        let mut model = make_gpt_oss_20b();
        model.active_parameters = None;

        assert!(split_moe_weights(&model, 12_800_000_000).is_none());
    }

    #[test]
    fn nonsense_expert_counts_are_refused_rather_than_extrapolated() {
        let weight_bytes = 12_800_000_000u64;

        for architecture in [
            ModelArchitecture::MixtureOfExperts { num_experts: 0, active_experts: 0 },
            ModelArchitecture::MixtureOfExperts { num_experts: 4, active_experts: 4 },
            ModelArchitecture::MixtureOfExperts { num_experts: 4, active_experts: 9 },
        ] {
            let mut model = make_gpt_oss_20b();
            model.architecture = architecture;
            assert!(split_moe_weights(&model, weight_bytes).is_none(), "{:?}", model.architecture);
        }
    }

    #[test]
    fn something_always_stays_resident() {
        // A split that offloaded everything would leave no attention stack on
        // the card, which is not a configuration llama.cpp can run.
        let mut model = make_gpt_oss_20b();
        model.active_parameters = Some(1); // pathological

        if let Some(split) = split_moe_weights(&model, 12_800_000_000) {
            assert!(split.resident_bytes > 0, "nothing left resident");
        }
    }

    fn make_codelama_7b() -> ModelMetadata {
        ModelMetadata {
            id: "codellama/CodeLlama-7b-hf".to_string(),
            name: "CodeLlama 7B".to_string(),
            family: "CodeLlama".to_string(),
            architecture: ModelArchitecture::Dense,
            total_parameters: 6_740_000_000,
            active_parameters: None,
            num_layers: 32,
            num_attention_heads: 32,
            num_kv_heads: 32, // MHA — no GQA
            head_dimension: 128,
            hidden_size: 4096,
            max_context_length: 16384,
            vocab_size: 32016,
            default_dtype: "bf16".to_string(),
            use_cases: vec!["code".to_string()],
            catalog_version: "1.0".to_string(),
        }
    }

    const GB: u64 = 1_073_741_824;
    const MB: u64 = 1_048_576;

    #[test]
    fn test_weight_memory_7b_q4() {
        let model = make_llama31_8b();
        let q4km = QuantizationSpec { label: "Q4_K_M".to_string(), bits_per_weight: 4.85, quality_rank: 5 };
        let weight_mem = estimate_weight_memory(&model, &q4km);
        // 8.03B × 4.85 / 8 ≈ 4.87 GB
        assert!(weight_mem > 4 * GB);
        assert!(weight_mem < 6 * GB);
    }

    #[test]
    fn test_weight_memory_moe_mixtral() {
        let model = make_mixtral_8x7b();
        let q4km = QuantizationSpec { label: "Q4_K_M".to_string(), bits_per_weight: 4.85, quality_rank: 5 };
        let weight_mem = estimate_weight_memory(&model, &q4km);
        // 46.7B × 4.85 / 8 ≈ 28.3 GB — ALL experts loaded
        assert!(weight_mem > 25 * GB);
        assert!(weight_mem < 32 * GB);
    }

    #[test]
    fn test_kv_cache_gqa_8k() {
        let model = make_llama31_8b();
        let config = EstimatorConfig::default();
        let kv = estimate_kv_cache_memory(&model, 8192, &config);
        // 2 × 32 × 8 × 128 × 8192 × 2 = 1,073,741,824 = 1 GB
        assert_eq!(kv, GB);
    }

    #[test]
    fn test_kv_cache_mha_vs_gqa() {
        let config = EstimatorConfig::default();
        // CodeLlama uses MHA (32 KV heads = 32 query heads)
        let codelama = make_codelama_7b();
        let kv_mha = estimate_kv_cache_memory(&codelama, 8192, &config);
        // Llama 3.1 uses GQA (8 KV heads, 32 query heads)
        let llama = make_llama31_8b();
        let kv_gqa = estimate_kv_cache_memory(&llama, 8192, &config);
        // MHA should be 4× larger than GQA (32/8 = 4)
        assert_eq!(kv_mha, kv_gqa * 4);
    }

    #[test]
    fn test_total_memory_with_overhead() {
        let model = make_llama31_8b();
        let q4km = QuantizationSpec { label: "Q4_K_M".to_string(), bits_per_weight: 4.85, quality_rank: 5 };
        let config = EstimatorConfig::default(); // 12% overhead
        let (w, kv, oh, total) = estimate_total_memory(&model, &q4km, 8192, &config);
        assert_eq!(total, w + kv + oh);
        // Overhead should be ~12% of (weights + kv)
        let expected_oh = ((w + kv) as f64 * 0.12) as u64;
        assert_eq!(oh, expected_oh);
    }

    #[test]
    fn test_quantization_hierarchy_order() {
        let hierarchy = quantization_hierarchy();
        for i in 1..hierarchy.len() {
            assert!(hierarchy[i].quality_rank < hierarchy[i - 1].quality_rank);
            assert!(hierarchy[i].bits_per_weight < hierarchy[i - 1].bits_per_weight);
        }
    }
}
