//! Choosing a model for a task, and being able to say why.
//!
//! PS 26117 asks for *"model auto selection across at least two different task
//! types"* — a coding request handled differently from a document summary. It
//! also asks, in step 10, that the router **record why a model was selected**,
//! and that an uncertain router fall back to something safe rather than quietly
//! picking badly.
//!
//! So the decision is a value, not a side effect: [`RoutingDecision`] carries the
//! model, the intent that led to it, and the reasons in the order they applied.
//! The task trace shows that list verbatim. A router that cannot explain itself
//! is indistinguishable from a coin toss with good luck.
//!
//! ## How it decides
//!
//! 1. **Classify the prompt.** Sarathi's weighted classifier, which scores every
//!    intent and derives confidence from how far ahead the leader is *and* how
//!    much evidence there was at all.
//! 2. **Low confidence routes to reasoning, not to nothing.** A general model
//!    handles a coding question adequately; a coding model handles a summary
//!    badly. When unsure, the cost of being wrong is lower in that direction.
//! 3. **Filter to real candidates** — enabled, right role, above the floor,
//!    cleared for this material.
//! 4. **Prefer the largest that fits.** Capability tracks size within a role, so
//!    the best model is the biggest one whose weights and KV cache fit the GPU
//!    budget that [`crate::ai_engine::vram_planner`] computes.
//! 5. **Fall back rather than fail.** If nothing fits the GPU, the smallest
//!    candidate runs partly on the CPU and the decision says so.

use serde::{Deserialize, Serialize};

use super::{ModelEntry, ModelRegistry, ModelRole, Modality};
use crate::ai_engine::startup::StartupModelTarget;
use crate::ai_engine::vram_planner::{plan_gpu_offload, GpuOffloadPlan};
use crate::capability::classifier::IntentClassifier;
use crate::model_intelligence::intent::PromptIntent;
use crate::policy::Classification;

/// Below this, the classification is not trusted to pick a specialist.
///
/// Chosen to match the classifier's own calibration: it reaches this only when
/// one intent leads clearly *and* several signals supported it. A single
/// incidental keyword cannot get here.
const SPECIALIST_CONFIDENCE: f32 = 0.55;

/// What the router decided, and every reason that led there.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingDecision {
    pub model_id: String,
    pub model_name: String,
    pub role: ModelRole,
    /// What the prompt was taken to be asking for.
    pub intent: String,
    pub confidence: f32,
    /// True when the first choice was unavailable and something else was used.
    pub used_fallback: bool,
    /// Ordered, human-readable. Shown verbatim in the task trace.
    pub reasons: Vec<String>,
    pub gpu_plan_summary: String,
    pub fully_on_gpu: bool,
}

/// What a conversation has already settled on.
///
/// Read from the conversation store at the start of a turn and written back
/// after routing. See [`ModelRouter::route_sticky`].
#[derive(Debug, Clone)]
pub struct StickyRoute {
    pub role: ModelRole,
    pub model_id: String,
}

/// Why no model could be chosen. Each names what would fix it.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RoutingFailure {
    pub role: ModelRole,
    pub reason: String,
}

pub struct ModelRouter;

/// How strong this entry's claim to be the orchestrator is. Higher wins.
///
/// The two claims are not equal, and the difference matters. An administrator
/// choosing a model in Models is a decision made now, about this machine; an
/// `orchestrator.*` id in the manifest is something a deployment (or a stray
/// provisioning script) wrote at some point in the past. When both are present
/// the live decision has to win, or the person who just picked a model watches
/// the chat answer from a different one and has no way to tell why.
///
/// The choice is matched on the exact installed coordinates rather than a name,
/// because two quantizations of one model share a name and are different files
/// — and because no model name belongs in the router at all. Which model runs
/// the chat is a runtime fact about this machine, not a compile-time one.
fn orchestrator_rank(entry: &ModelEntry, chosen: Option<&StartupModelTarget>) -> u8 {
    let matches_choice = chosen
        .and_then(|chosen| entry.load.as_ref().map(|load| (load, chosen)))
        .map(|(load, chosen)| {
            normalized(&load.provider_id) == normalized(&chosen.provider_id)
                && normalized(&load.model_id) == normalized(&chosen.model_id)
                // A choice saved before the installer could read a quantisation
                // out of a file name carries "GGUF", which names the container
                // and identifies no variant. Demanding it match the registry's
                // real label is what made this setting inert: every stored
                // choice failed here, the rank stayed 0, and the chat answered
                // from whichever model the size sort reached first while the
                // Models screen still showed the star beside the chosen one.
                //
                // Such a choice selects the package and leaves the variant
                // open. That is ambiguous only when one package is registered
                // at two quantisations, and a choice saved by this build no
                // longer carries a placeholder at all — the coordinates are
                // resolved to the registry's spelling before they are written.
                && (is_placeholder_quantization(&chosen.quantization)
                    || normalized(&load.quantization) == normalized(&chosen.quantization))
        })
        .unwrap_or(false);
    if matches_choice {
        return 2;
    }
    // A manifest tag only speaks when nobody has chosen. Otherwise a stale tag
    // would quietly outrank the administrator who chose today.
    if chosen.is_none() && (entry.id == "orchestrator" || entry.id.starts_with("orchestrator.")) {
        return 1;
    }
    0
}

/// Whether this entry is the orchestrator by either route.
fn is_orchestrator(entry: &ModelEntry, chosen: Option<&StartupModelTarget>) -> bool {
    orchestrator_rank(entry, chosen) > 0
}

/// Whether a stored quantisation names the container rather than the weights.
///
/// "GGUF" is what a package manifest records when the file name declares
/// nothing it can parse. It picks out no particular variant, so a choice
/// carrying it selects the package and leaves the variant to the registry.
fn is_placeholder_quantization(quantization: &str) -> bool {
    let trimmed = quantization.trim();
    trimmed.is_empty() || trimmed.eq_ignore_ascii_case("gguf")
}

/// Punctuation- and case-insensitive form, so `org/Model-GGUF` and
/// `org/model_gguf` are recognised as the same package id.
fn normalized(value: &str) -> String {
    value
        .chars()
        .filter(|character| character.is_ascii_alphanumeric())
        .flat_map(char::to_lowercase)
        .collect()
}

impl ModelRouter {
    /// Maps a classified intent onto the role that should handle it.
    ///
    /// Only coding gets its own specialist. Mathematics, research and reasoning
    /// all want a strong general model rather than a differently-trained one,
    /// and inventing a role per intent would produce a registry nobody can fill.
    fn role_for(intent: PromptIntent) -> ModelRole {
        match intent {
            PromptIntent::Coding => ModelRole::Coding,
            PromptIntent::Reasoning
            | PromptIntent::Mathematics
            | PromptIntent::ToolCalling
            | PromptIntent::Research
            | PromptIntent::GeneralChat => ModelRole::Reasoning,
        }
    }

    /// Routes a text prompt, with no administrator-configured orchestrator to
    /// honour. Callers that can reach the configuration should prefer
    /// [`Self::route_with_orchestrator`] so the chosen chat model wins.
    pub fn route(
        registry: &ModelRegistry,
        prompt: &str,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> Result<RoutingDecision, RoutingFailure> {
        Self::route_with_orchestrator(
            registry,
            prompt,
            classification,
            vram_total_bytes,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
            None,
        )
    }

    /// Routes a text prompt, preferring the orchestrator an administrator
    /// chose in Models → *Set as orchestrator*.
    ///
    /// `orchestrator` is the exact installed package coordinates from
    /// `ai_settings`, or `None` when nobody has chosen. It is a preference and
    /// not a bypass: the chosen model still has to clear every hard gate — the
    /// right role for the prompt, the parameter floor, the classification it is
    /// cleared for. A choice that fails a gate loses to a model that passes,
    /// and the reasons say which gate.
    #[allow(clippy::too_many_arguments)]
    pub fn route_with_orchestrator(
        registry: &ModelRegistry,
        prompt: &str,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
        orchestrator: Option<&StartupModelTarget>,
    ) -> Result<RoutingDecision, RoutingFailure> {
        let classified = IntentClassifier::classify(prompt);
        // Taken before `intent` is moved into `role_for` below.
        let intent_label = classified.capability_name().to_string();
        let confidence = classified.confidence;
        let mut reasons = Vec::new();

        // Step 1–2: what kind of task is this, and do we trust the answer?
        let confident = classified.confidence >= SPECIALIST_CONFIDENCE;
        let role = if confident {
            reasons.push(format!(
                "Read as a {} request (confidence {:.0}%).",
                intent_label,
                confidence * 100.0
            ));
            Self::role_for(classified.intent)
        } else {
            reasons.push(format!(
                "Intent was unclear (confidence {:.0}%), so it is being handled by a general \
                 reasoning model rather than a specialist.",
                confidence * 100.0
            ));
            ModelRole::Reasoning
        };

        Self::route_to_role(
            registry,
            role,
            classification,
            vram_total_bytes,
            reasons,
            intent_label,
            confidence,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
            orchestrator,
        )
    }

    /// Routes a turn that belongs to a conversation which may already have a
    /// model.
    ///
    /// ## Why a thread keeps its model
    ///
    /// Routing classifies one prompt, and it ran on every turn, so the model
    /// changed whenever the *wording* moved — not when the work did. A follow-up
    /// is the clearest case: "yes, go ahead" and "now check that against the
    /// curve" carry almost no signal, so they scored as unclear, fell to the
    /// general reasoning band, and pulled the thread off whatever specialist had
    /// been doing the job. Every switch also evicted a warm server and threw
    /// away its prompt cache, so the person paid a cold start for it.
    ///
    /// So a settled thread is only overturned by a turn that classifies
    /// *confidently* into a different role. That is precisely the case where the
    /// person has changed the kind of work — "now write me a script to do that"
    /// — and precisely what an unclear follow-up cannot do.
    ///
    /// A sticky model that is no longer in the registry, or no longer a
    /// candidate for its role, is not forced: the role is routed afresh and the
    /// reasons say so. Nothing here can widen what a turn may reach — the sticky
    /// model still has to pass every gate `candidates` applies, including the
    /// classification clearance for *this* turn.
    #[allow(clippy::too_many_arguments)]
    pub fn route_sticky(
        registry: &ModelRegistry,
        prompt: &str,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
        orchestrator: Option<&StartupModelTarget>,
        sticky: Option<&StickyRoute>,
    ) -> Result<RoutingDecision, RoutingFailure> {
        let Some(sticky) = sticky else {
            return Self::route_with_orchestrator(
                registry,
                prompt,
                classification,
                vram_total_bytes,
                required_modality,
                require_structured_output,
                available_runtime_profiles,
                allowed_licenses,
                orchestrator,
            );
        };

        let classified = IntentClassifier::classify(prompt);
        let intent_label = classified.capability_name().to_string();
        let confidence = classified.confidence;
        let confident = confidence >= SPECIALIST_CONFIDENCE;
        let asked_for = if confident {
            Self::role_for(classified.intent)
        } else {
            ModelRole::Reasoning
        };

        if confident && asked_for != sticky.role {
            // The work itself changed. Route it fresh, and say why the thread
            // moved — a person who notices the model changed is owed the reason.
            let mut reasons = vec![format!(
                "This turn reads as {} work (confidence {:.0}%), which is a different kind of task from the rest of this conversation, so the model changed.",
                intent_label,
                confidence * 100.0
            )];
            reasons.push(format!(
                "Read as a {} request (confidence {:.0}%).",
                intent_label,
                confidence * 100.0
            ));
            return Self::route_to_role(
                registry,
                asked_for,
                classification,
                vram_total_bytes,
                reasons,
                intent_label,
                confidence,
                required_modality,
                require_structured_output,
                available_runtime_profiles,
                allowed_licenses,
                orchestrator,
            );
        }

        // The thread keeps its role. Keep the model too, if it is still a
        // candidate for that role under this turn's gates.
        let candidates = registry.candidates(
            sticky.role,
            classification,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
        );
        if let Some(entry) = candidates.iter().find(|e| e.id == sticky.model_id) {
            let plan = plan_gpu_offload(
                vram_total_bytes,
                entry.weights_bytes,
                entry.context_length,
                None,
            );
            let reasons = vec![
                format!(
                    "Kept on {}, which has been answering this conversation. Re-routing every turn changes the model when the wording moves rather than when the work does, and costs a cold model server each time.",
                    entry.name
                ),
                plan.reason.clone(),
            ];
            return Ok(Self::decide(
                entry,
                sticky.role,
                intent_label,
                confidence,
                plan,
                false,
                reasons,
            ));
        }

        let reasons = vec![format!(
            "{} answered this conversation before but is no longer available for {} work, so a model is being chosen again.",
            sticky.model_id,
            sticky.role.label()
        )];
        Self::route_to_role(
            registry,
            sticky.role,
            classification,
            vram_total_bytes,
            reasons,
            intent_label,
            confidence,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
            orchestrator,
        )
    }

    /// Routes to a named role directly, for work whose kind is already known —
    /// OCR on a scanned page, embeddings for retrieval — where classifying the
    /// user's words would be answering the wrong question.
    ///
    /// **No caller in `src/`, and one that matters in `tests/`.** This was
    /// briefly deleted as dead code on the strength of a grep over `src/`
    /// alone. It is not dead: `two_runtimes.rs` uses it to prove the property
    /// PS 26117 asks for — that a coding task and a document task reach
    /// different models on different runtimes — and that is the only place the
    /// claim is checked end to end.
    ///
    /// What is true is that the *product* does not route OCR through here:
    /// `commands::ocr` picks its model by a hardcoded id. That is worth
    /// changing, and until it is, this is the function it would change to use.
    pub fn route_for_role(
        registry: &ModelRegistry,
        role: ModelRole,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> Result<RoutingDecision, RoutingFailure> {
        let reasons = vec![format!(
            "The task needs a {} model, so no classification of the prompt was involved.",
            role.label()
        )];
        Self::route_to_role(
            registry,
            role,
            classification,
            vram_total_bytes,
            reasons,
            role.label().to_string(),
            1.0,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
            None,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn route_to_role(
        registry: &ModelRegistry,
        role: ModelRole,
        classification: Option<Classification>,
        vram_total_bytes: u64,
        mut reasons: Vec<String>,
        intent_label: String,
        confidence: f32,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
        orchestrator: Option<&StartupModelTarget>,
    ) -> Result<RoutingDecision, RoutingFailure> {
        // Step 3: real candidates only.
        let mut candidates = registry.candidates(
            role,
            classification,
            required_modality,
            require_structured_output,
            available_runtime_profiles,
            allowed_licenses,
        );
        if candidates.is_empty() {
            // Nothing at or above the floor for this role, and that is a
            // refusal rather than a rescue.
            //
            // The below-floor rescue at step 5a deliberately does not apply
            // here. It exists for a machine whose registry *has* proper models
            // that will not fit in VRAM — there the floor stops being a quality
            // rule and starts guaranteeing the worst outcome available. A
            // registry that holds nothing meeting the floor at all is a
            // different situation: answering a coding request from a 1.5B model
            // is not a degradation somebody can judge, and "install a model
            // that can do this" is an answer the operator can act on where a
            // bad reply is not. `explain_no_candidates` says which of the two
            // it is.
            return Err(RoutingFailure {
                role,
                reason: Self::explain_no_candidates(
                    registry,
                    role,
                    classification,
                    required_modality,
                    require_structured_output,
                    available_runtime_profiles,
                    allowed_licenses,
                ),
            });
        }

        if let Some(classification) = classification {
            reasons.push(format!(
                "{} of {} registered models are cleared for {} material.",
                candidates.len(),
                registry.all().len(),
                classification.label()
            ));
        }

        // Step 4: orchestrator first, then largest, so the first that fits
        // is the best that fits. The orchestrator is the chat model an
        // administrator chose in Models → "Set as orchestrator" (matched on
        // the exact installed coordinates), or a manifest entry tagged with an
        // id starting "orchestrator.". It should win when it fits; only when
        // it does not fit do we fall back to the largest cleared model that
        // does.
        //
        // Nothing here knows a model by name. A router that carried a
        // compiled-in favourite would answer from that model however loudly
        // the administrator had chosen another one — which is the bug this
        // ordering exists to prevent.
        //
        // The sort key is (is_orchestrator desc, size desc, preferred desc,
        // rank asc). The orchestrator flag is the dominant factor so the
        // user's choice is honoured; the size band is the secondary
        // criterion so capability tracks size within a role; the
        // tie-breakers only matter when two candidates have the same
        // `parameters_b`. `preferred` is the operator-set "use this one"
        // knob, and `rank_within_band` is the telemetry-driven ordering,
        // so the deterministic default is preserved when both are absent.
        candidates.sort_by(|a, b| {
            let a_is_orch = orchestrator_rank(a, orchestrator);
            let b_is_orch = orchestrator_rank(b, orchestrator);
            b_is_orch
                .cmp(&a_is_orch)
                // Active parameters where the model declares them, matching
                // `meets_floor`. See `ModelEntry::effective_parameters_b`.
                .then_with(|| {
                    b.effective_parameters_b()
                        .partial_cmp(&a.effective_parameters_b())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| {
                    b.routing
                        .preferred
                        .cmp(&a.routing.preferred)
                })
                .then_with(|| {
                    a.routing
                        .rank_within_band
                        .cmp(&b.routing.rank_within_band)
                })
        });

        // The administrator's choice is answered before the size search, not
        // inside it.
        //
        // It used to be sorted to the front of `candidates` and then left to
        // the same `full_offload` filter as everything else, which quietly
        // undid it: an orchestrator too large to fit entirely in VRAM failed
        // the `if`, the loop moved on, and the first smaller model that did fit
        // answered instead. The Models screen kept the star beside the chosen
        // model while a different one did the talking — and the reason line
        // said "the largest cleared model that fits in VRAM", which was true
        // and was not the question.
        //
        // A choice that reaches this point has already cleared every hard gate
        // — role, floor, classification, modality, licence — because those are
        // what `candidates` filters on. Fitting in VRAM is not one of them; it
        // is a question of how fast the model will be, and that is the
        // administrator's trade to make. Step 5's own comment already said so.
        if let Some(entry) = candidates
            .iter()
            .find(|entry| is_orchestrator(entry, orchestrator))
            .copied()
        {
            let plan = plan_gpu_offload(
                vram_total_bytes,
                entry.weights_bytes,
                entry.context_length,
                None,
            );
            reasons.push(if plan.full_offload {
                format!(
                    "{} is configured as the orchestrator and fits in VRAM.",
                    entry.name
                )
            } else {
                // Named rather than silently accepted: a partly-offloaded model
                // is markedly slower, and an administrator watching a slow
                // answer should be able to read why it is this model.
                format!(
                    "{} is configured as the orchestrator. It does not fit entirely in this machine's VRAM, so it runs partly on the CPU rather than being replaced by a smaller model. It will be slower.",
                    entry.name
                )
            });
            reasons.push(plan.reason.clone());
            let degraded = !plan.full_offload;
            return Ok(Self::decide(
                entry,
                role,
                intent_label,
                confidence,
                plan,
                degraded,
                reasons,
            ));
        }

        for entry in &candidates {
            let plan = plan_gpu_offload(vram_total_bytes, entry.weights_bytes, entry.context_length, None);
            if plan.full_offload {
                reasons.push(format!(
                    "{} is the largest cleared {} model that fits in VRAM.",
                    entry.name,
                    role.label()
                ));
                reasons.push(plan.reason.clone());
                return Ok(Self::decide(entry, role, intent_label, confidence, plan, false, reasons));
            }
        }

        // Step 5a: nothing at or above the floor fits in VRAM. Before accepting
        // a partial offload, look *below* the floor for something that fits
        // entirely on the GPU.
        //
        // The floor assumes that a model meeting it can run. Where that is
        // false the floor stops being a quality rule and becomes a guarantee of
        // the worst outcome available: on an 8 GB laptop GPU every cleared
        // coding model was 9B or larger, so every coding request ran partly on
        // the CPU at about 0.4 tokens a second and was stopped as stuck before
        // it finished a sentence. A 4B model in the same registry would have
        // fitted entirely and answered in seconds.
        //
        // Ordered largest-first, so this takes the best model that fits rather
        // than the smallest. `used_fallback` is set and the reason says plainly
        // that the floor was crossed, because a smaller model answering is a
        // fact about the answer's quality and the reader is owed it.
        // An operator who marked a model preferred for this role has asked for
        // that model, and asked knowing this machine.
        //
        // Crossing the floor is a rescue for a deployment that has no usable
        // option at all — it is not a licence to overrule a deliberate choice.
        // The orchestrator already wins this argument for the same reason;
        // `preferred` is the per-role form of the same intent, and a fallback
        // that quietly answered from a different model would leave the Models
        // screen showing one choice while another did the talking.
        let preferred_above_floor = candidates.iter().any(|entry| entry.routing.preferred);
        let mut below_floor = if preferred_above_floor {
            Vec::new()
        } else {
            registry.candidates_ignoring_floor(
                role,
                classification,
                required_modality,
                require_structured_output,
                available_runtime_profiles,
                allowed_licenses,
            )
        };
        below_floor.retain(|entry| !entry.meets_floor(role));
        below_floor.sort_by(|a, b| {
            b.effective_parameters_b()
                .partial_cmp(&a.effective_parameters_b())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        for entry in &below_floor {
            let plan = plan_gpu_offload(
                vram_total_bytes,
                entry.weights_bytes,
                entry.context_length,
                None,
            );
            if plan.full_offload {
                reasons.push(format!(
                    "No cleared {} model at or above the {}B floor fits in this machine's VRAM. \
                     {} is below the floor but fits entirely on the GPU, so it answers rather \
                     than a larger model running partly on the CPU at a few tokens a second.",
                    role.label(),
                    role.minimum_parameters_b(),
                    entry.name
                ));
                reasons.push(plan.reason.clone());
                return Ok(Self::decide(
                    entry,
                    role,
                    intent_label,
                    confidence,
                    plan,
                    true,
                    reasons,
                ));
            }
        }

        // Step 5: nothing fits entirely, at or below the floor. The configured
        // orchestrator still wins
        // here — an administrator who chose a model slightly too big for this
        // GPU asked for that model, not for a different one — and otherwise the
        // smallest candidate runs partly on the CPU. Either way ARJUN keeps
        // working on a machine that can still do the job slowly, rather than
        // refusing.
        let chosen = candidates
            .iter()
            .find(|entry| is_orchestrator(entry, orchestrator))
            .copied();
        let fallback = chosen.unwrap_or_else(|| {
            candidates
                .last()
                .copied()
                .expect("candidates was checked non-empty above")
        });
        let plan = plan_gpu_offload(
            vram_total_bytes,
            fallback.weights_bytes,
            fallback.context_length,
            None,
        );

        reasons.push(if chosen.is_some() {
            format!(
                "No cleared {} model fits entirely in this machine's VRAM. {} is configured as \
                 the orchestrator, so it runs partly on the CPU rather than being replaced. It \
                 will run more slowly.",
                role.label(),
                fallback.name
            )
        } else {
            format!(
                "No cleared {} model fits entirely in this machine's VRAM, so ARJUN fell back to \
                 {}, the smallest one available. It will run more slowly.",
                role.label(),
                fallback.name
            )
        });
        reasons.push(plan.reason.clone());

        Ok(Self::decide(fallback, role, intent_label, confidence, plan, true, reasons))
    }

    /// Says which filter emptied the candidate list, so the fix is obvious.
    ///
    /// "No model available" is useless to an administrator. Whether the registry
    /// is empty, the model is disabled, everything is below the floor, or nothing
    /// is cleared for this material are four different problems with four
    /// different remedies.
    fn explain_no_candidates(
        registry: &ModelRegistry,
        role: ModelRole,
        classification: Option<Classification>,
        required_modality: Option<Modality>,
        require_structured_output: bool,
        available_runtime_profiles: &[String],
        allowed_licenses: &[String],
    ) -> String {
        if registry.all().is_empty() {
            return "No models are registered yet. An administrator imports one in \
                    Provisioning mode."
                .to_string();
        }

        let serving: Vec<_> = registry
            .all()
            .iter()
            .filter(|e| e.serves(role))
            .collect();

        if serving.is_empty() {
            return format!(
                "No registered model is set up for {} work. Register one, or enable the role \
                 on an existing model.",
                role.label()
            );
        }

        if !serving.iter().any(|e| e.enabled) {
            return format!(
                "Every {} model is currently disabled. An administrator re-enables one in \
                 Models.",
                role.label()
            );
        }

        if !serving.iter().any(|e| e.meets_floor(role)) {
            return format!(
                "The registered {} models are all below {:.0}B parameters, which is too small \
                 to be reliable at this kind of work. A larger model is needed.",
                role.label(),
                role.minimum_parameters_b()
            );
        }

        // Check modality filter
        if let Some(modality) = required_modality {
            if !serving.iter().any(|e| e.supports_modality(modality)) {
                return format!(
                    "No {} model supports the required modality ({}). A model with {} capability is needed.",
                    role.label(),
                    modality.label(),
                    modality.label()
                );
            }
        }

        // Check structured output filter
        if require_structured_output {
            if !serving.iter().any(|e| e.supports_structured_output()) {
                return format!(
                    "No {} model supports structured output / tool calling. A model with this capability is needed.",
                    role.label()
                );
            }
        }

        // Check runtime profile filter
        if !available_runtime_profiles.is_empty() {
            if !serving.iter().any(|e| e.runtime_profile_available(available_runtime_profiles)) {
                return format!(
                    "No {} model is compatible with the available runtime profiles ({}). A model compatible with one of these profiles is needed.",
                    role.label(),
                    available_runtime_profiles.join(", ")
                );
            }
        }

        // Check license filter
        if !allowed_licenses.is_empty() {
            if !serving.iter().any(|e| e.license_allowed(allowed_licenses)) {
                return format!(
                    "No {} model has an allowed license. Allowed licenses: {}. A model with an allowed license is needed.",
                    role.label(),
                    allowed_licenses.join(", ")
                );
            }
        }

        match classification {
            Some(c) => {
                // Name the model that needs administrator review rather
                // than asking the operator to find one. The closest
                // candidate is the one that clears every other gate
                // (role, modality, floor, runtime, license) but not the
                // classification — that is the one to inspect and
                // either clear or refuse to clear on the record.
                let closest = serving
                    .iter()
                    .find(|e| {
                        e.enabled
                            && e.meets_floor(role)
                            && (required_modality.is_none()
                                || e.supports_modality(required_modality.unwrap()))
                            && (!require_structured_output || e.supports_structured_output())
                            && (available_runtime_profiles.is_empty()
                                || e.runtime_profile_available(available_runtime_profiles))
                            && (allowed_licenses.is_empty()
                                || e.license_allowed(allowed_licenses))
                    })
                    .map(|e| e.id.as_str());
                match closest {
                    Some(model_id) => format!(
                        "No {} model is cleared for {} material. Ask an administrator to \
                         review {} and add {} to its permitted classifications.",
                        role.label(),
                        c.label(),
                        model_id,
                        c.label()
                    ),
                    None => format!(
                        "No {} model is cleared for {} material. An administrator clears one, \
                         having checked it is appropriate for data of that sensitivity.",
                        role.label(),
                        c.label()
                    ),
                }
            }
            None => format!("No {} model is available.", role.label()),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn decide(
        entry: &ModelEntry,
        role: ModelRole,
        intent: String,
        confidence: f32,
        plan: GpuOffloadPlan,
        used_fallback: bool,
        reasons: Vec<String>,
    ) -> RoutingDecision {
        RoutingDecision {
            model_id: entry.id.clone(),
            model_name: entry.name.clone(),
            role,
            intent,
            confidence,
            used_fallback,
            reasons,
            gpu_plan_summary: plan.reason,
            fully_on_gpu: plan.full_offload,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::registry::tests::entry;
    use crate::registry::{ModelManifest, ModelRegistry};
    use std::path::PathBuf;

    const GB: u64 = 1024 * 1024 * 1024;

    fn registry(entries: Vec<ModelEntry>) -> ModelRegistry {
        ModelRegistry::from_manifest(ModelManifest { models: entries }, PathBuf::from("registry.json"))
            .unwrap()
    }

    // ── Stickiness ───────────────────────────────────────────────────────
    //
    // A thread keeps its model until a turn confidently asks for different
    // work. See `ModelRouter::route_sticky`.

    fn sticky(role: ModelRole, model_id: &str) -> StickyRoute {
        StickyRoute { role, model_id: model_id.to_string() }
    }

    fn route_with_sticky<'a>(
        registry: &ModelRegistry,
        prompt: &str,
        held: Option<&'a StickyRoute>,
    ) -> RoutingDecision {
        ModelRouter::route_sticky(
            registry, prompt, None, 24 * GB, None, false, &[], &[], None, held,
        )
        .unwrap()
    }

    /// The failure this exists for: a follow-up carries no signal, so it used
    /// to fall to the general band and pull the thread off its coding model.
    #[test]
    fn an_unclear_follow_up_keeps_the_conversations_model() {
        let registry = stocked();
        let held = sticky(ModelRole::Coding, "qwen-coder-14b");

        for follow_up in ["yes, go ahead", "now do the other one too", "thanks, and the rest?"] {
            let decision = route_with_sticky(&registry, follow_up, Some(&held));
            assert_eq!(decision.model_id, "qwen-coder-14b", "{follow_up:?}");
            assert_eq!(decision.role, ModelRole::Coding, "{follow_up:?}");
        }
    }

    /// And the person is told why the model did not change.
    #[test]
    fn keeping_the_model_says_so() {
        let registry = stocked();
        let held = sticky(ModelRole::Coding, "qwen-coder-14b");
        let decision = route_with_sticky(&registry, "yes, go ahead", Some(&held));
        assert!(
            decision.reasons.iter().any(|r| r.contains("has been answering this conversation")),
            "{:?}",
            decision.reasons
        );
    }

    /// Stickiness is not a cage: work that has genuinely changed re-routes.
    #[test]
    fn a_confident_change_of_work_moves_the_thread() {
        let registry = stocked();
        let held = sticky(ModelRole::Reasoning, "qwen-32b");

        let decision = route_with_sticky(
            &registry,
            "Refactor this Python function and fix the stack trace",
            Some(&held),
        );

        assert_eq!(decision.role, ModelRole::Coding);
        assert!(
            decision.reasons.iter().any(|r| r.contains("different kind of task")),
            "the reader is owed the reason the model changed: {:?}",
            decision.reasons
        );
    }

    /// A held model that has left the registry does not pin the thread to
    /// nothing — the role is routed again.
    #[test]
    fn a_held_model_that_is_gone_routes_the_role_afresh() {
        let registry = stocked();
        let held = sticky(ModelRole::Coding, "a-model-that-was-deleted");

        let decision = route_with_sticky(&registry, "yes, go ahead", Some(&held));

        assert_eq!(decision.role, ModelRole::Coding);
        assert_eq!(decision.model_id, "qwen-coder-14b");
        assert!(
            decision.reasons.iter().any(|r| r.contains("no longer available")),
            "{:?}",
            decision.reasons
        );
    }

    /// With nothing held, this is exactly the old behaviour.
    #[test]
    fn a_first_turn_routes_as_it_always_did() {
        let registry = stocked();
        let fresh = route_with_sticky(&registry, "Summarise this report", None);
        let direct = ModelRouter::route(
            &registry, "Summarise this report", None, 24 * GB, None, false, &[], &[],
        )
        .unwrap();
        assert_eq!(fresh.model_id, direct.model_id);
        assert_eq!(fresh.role, direct.role);
    }

    fn stocked() -> ModelRegistry {
        registry(vec![
            entry("qwen-coder-14b", 14.0, vec![ModelRole::Coding]),
            entry("qwen-coder-7b", 7.0, vec![ModelRole::Coding]),
            entry("qwen-32b", 32.0, vec![ModelRole::Reasoning]),
            entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]),
            entry("surya", 0.65, vec![ModelRole::DocumentOcr]),
        ])
    }

    /// The 8 GB laptop, reproduced.
    ///
    /// Measured on an RTX 5060 Laptop (7.9 GB): every cleared coding model was
    /// 9B or larger, all of them above the 7B floor and none of them a fit. The
    /// router took the smallest *candidate* — 9B, 5.4 GB — and ran it at 31/32
    /// layers with the last on the CPU. It decoded at about 0.4 tokens a second,
    /// produced 116 characters in five minutes, and was stopped as stuck.
    ///
    /// A 4B model was sitting in the same registry, excluded for being below the
    /// coding floor. It fits entirely on the GPU. The floor is a quality rule
    /// and it was buying nothing: the larger model did not answer at all.
    #[test]
    fn a_model_below_the_floor_that_fits_beats_a_larger_one_on_the_cpu() {
        let mut small = entry(
            "nemotron-nano-4b",
            4.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        small.context_length = 8192;
        let mut large = entry("qwen-9b", 9.0, vec![ModelRole::Reasoning, ModelRole::Coding]);
        large.context_length = 8192;
        let registry = registry(vec![large, small]);

        // 6 GB rather than the laptop's nominal 8: the measured weight budget
        // there was 4.91 GB once the 1.25 GB KV cache was taken out, and the 9B
        // is 5.4 GB. This is the same relationship — one model over the budget,
        // one under it — expressed against the fixture's own geometry.
        let coding = ModelRouter::route(
            &registry,
            "Write the example code for a linked list in cpp",
            None,
            6 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(
            coding.model_id, "nemotron-nano-4b",
            "the 9B does not fit and decodes at a few tokens a second; the 4B fits entirely"
        );
        assert!(
            coding.fully_on_gpu,
            "the whole point of crossing the floor is that this one actually fits"
        );
        assert!(
            coding.used_fallback,
            "a model below the floor answering is a degradation and must be reported as one"
        );
        assert!(
            coding.reasons.iter().any(|reason| reason.contains("below the floor")),
            "the reader is owed the reason the smaller model answered: {:?}",
            coding.reasons
        );
    }

    /// A deliberate choice is not overruled by the rescue.
    ///
    /// The below-floor fallback exists for a machine with no usable option. An
    /// operator who marked the 9B preferred for coding has looked at this
    /// machine and accepted that it runs partly on the CPU — answering from a
    /// different model anyway would leave the Models screen showing one choice
    /// while another did the talking.
    #[test]
    fn a_preferred_model_above_the_floor_is_not_replaced_by_one_below_it() {
        let mut small = entry(
            "nemotron-nano-4b",
            4.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        small.context_length = 8192;
        let mut large = entry("qwen-9b", 9.0, vec![ModelRole::Reasoning, ModelRole::Coding]);
        large.context_length = 8192;
        large.routing.preferred = true;
        let registry = registry(vec![large, small]);

        // The same 6 GB budget that sends coding to the 4B without a preference.
        let coding = ModelRouter::route(
            &registry,
            "Write the example code for a linked list in cpp",
            None,
            6 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(
            coding.model_id, "qwen-9b",
            "the operator's preferred coding model must answer, slow or not"
        );
        assert!(
            !coding.fully_on_gpu,
            "and the decision must still say it does not fit, rather than pretending"
        );
    }

    /// This laptop, with the coding model chosen.
    ///
    /// The RTX 5060 Laptop reports 7.9 GB of dedicated VRAM. Installed are a
    /// Nemotron 4B (below the 7B coding floor, fits entirely), a Qwen3.5-9B
    /// (above the floor, 5.4 GB of weights) and a Gemma 12B (larger still).
    /// Coding was being answered by the 4B, because nothing above the floor
    /// fitted in VRAM and the below-floor rescue took over.
    ///
    /// With Qwen3.5-9B chosen in Models -> *Set as orchestrator*, that choice
    /// is answered before the VRAM search and before the rescue, so coding
    /// goes to the 9B whether or not it fits, and the trace says which.
    #[test]
    fn the_chosen_9b_answers_coding_on_the_8gb_laptop() {
        let mut nemotron = entry(
            "NVIDIA_Nemotron3-Nano-4B_Q4_K_M",
            4.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        nemotron.weights_bytes = 2_500_000_000;
        let mut qwen = entry(
            "Qwen_Qwen3.5-9B_Q4_K_S",
            9.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        qwen.weights_bytes = 5_394_097_376;
        qwen.load = Some(crate::registry::LoadSpec {
            provider_id: "local".into(),
            model_id: "Qwen/Qwen3.5-9B".into(),
            quantization: "Q4_K_S".into(),
        });
        let mut gemma = entry(
            "google_gemma-4-12b-it_Q4_K_M",
            12.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        gemma.weights_bytes = 7_300_000_000;
        let registry = registry(vec![nemotron, qwen, gemma]);

        // The coordinates `set_orchestrator_model` writes into `config.json`.
        let chosen = StartupModelTarget {
            provider_id: "local".to_string(),
            model_id: "Qwen/Qwen3.5-9B".to_string(),
            quantization: "Q4_K_S".to_string(),
        };

        let coding = ModelRouter::route_with_orchestrator(
            &registry,
            "Write the example code for a linked list in cpp",
            None,
            7899 * 1024 * 1024,
            None,
            false,
            &[],
            &[],
            Some(&chosen),
        )
        .unwrap();

        assert_eq!(coding.role, ModelRole::Coding, "the prompt is coding work");
        assert_eq!(
            coding.model_id, "Qwen_Qwen3.5-9B_Q4_K_S",
            "the chosen model answers coding, not the 4B the rescue reached for"
        );
        assert!(
            coding
                .reasons
                .iter()
                .any(|reason| reason.contains("configured as the orchestrator")),
            "the trace has to say why this model answered: {:?}",
            coding.reasons
        );
    }

    /// The floor still holds when the larger model fits.
    ///
    /// Crossing it is a last resort, not a preference. On a machine with room
    /// for the 9B, the 9B answers and nothing is reported as degraded.
    #[test]
    fn the_floor_is_not_crossed_when_a_model_above_it_fits() {
        let mut small = entry(
            "nemotron-nano-4b",
            4.0,
            vec![ModelRole::Reasoning, ModelRole::Coding],
        );
        small.context_length = 8192;
        let mut large = entry("qwen-9b", 9.0, vec![ModelRole::Reasoning, ModelRole::Coding]);
        large.context_length = 8192;
        let registry = registry(vec![large, small]);

        let coding = ModelRouter::route(
            &registry,
            "Write the example code for a linked list in cpp",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(coding.model_id, "qwen-9b");
        assert!(!coding.used_fallback);
    }

    /// The problem statement's own demo: a coding request and a document summary
    /// must reach different models, each with a reason.
    #[test]
    fn the_two_demo_task_types_reach_different_models() {
        let registry = stocked();

        let coding = ModelRouter::route(
            &registry,
            "Refactor this Python function and write a unit test for the null pointer case",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        let summary = ModelRouter::route(
            &registry,
            "Summarise the key findings in this inspection report and list them by severity",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();

        assert_eq!(coding.role, ModelRole::Coding);
        assert_eq!(summary.role, ModelRole::Reasoning);
        assert_ne!(coding.model_id, summary.model_id);
    }

    #[test]
    fn an_unclear_prompt_goes_to_reasoning_rather_than_a_specialist() {
        let registry = stocked();
        let decision =
            ModelRouter::route(&registry, "hello", None, 24 * GB, None, false, &[], &[]).unwrap();
        assert_eq!(decision.role, ModelRole::Reasoning);
    }

    #[test]
    fn the_largest_model_that_fits_is_preferred() {
        let registry = stocked();
        let decision = ModelRouter::route(
            &registry,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-32b");
        assert!(!decision.used_fallback);
    }

    /// The coordinates an administrator persisted with "Set as orchestrator".
    fn chose(model_id: &str) -> StartupModelTarget {
        StartupModelTarget {
            provider_id: "huggingface".to_string(),
            model_id: model_id.to_string(),
            quantization: "Q4_K_M".to_string(),
        }
    }

    /// The orchestrator is the user-configured chat model. When it fits in
    /// VRAM, the router should pick it instead of a larger cleared model.
    /// (A manifest can also tag one by giving it an id starting with
    /// "orchestrator." — see [`ModelRegistry::orchestrator_entry`].)
    #[test]
    fn the_orchestrator_wins_over_a_larger_cleared_model() {
        let mut r = stocked();
        // Mark qwen-8b (smaller than qwen-32b) as the orchestrator.
        if let Some(qwen8b) = r.entries.iter_mut().find(|e| e.id == "qwen-8b") {
            qwen8b.id = "orchestrator.qwen-8b".to_string();
        }
        let decision = ModelRouter::route(
            &r,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        // qwen-8b is 8B and fits in 80 GB. qwen-32b is 32B and also fits.
        // The orchestrator should win, even though it is smaller.
        assert_eq!(decision.model_id, "orchestrator.qwen-8b");
        assert!(!decision.used_fallback);
    }

    /// The bug this exists to prevent: an administrator picks a chat model in
    /// Models, and the router answers from a different one because that other
    /// model happens to be bigger. The choice is coordinates in the config, not
    /// a marker in the manifest, so the router has to be told about it.
    #[test]
    fn the_administrator_choice_beats_the_largest_model_that_fits() {
        let registry = stocked();

        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
            Some(&chose("qwen-8b")),
        )
        .unwrap();

        assert_eq!(decision.model_id, "qwen-8b", "qwen-32b is larger and also fits");
        assert!(!decision.used_fallback);
        assert!(
            decision
                .reasons
                .iter()
                .any(|reason| reason.contains("configured as the orchestrator")),
            "the trace has to say why the smaller model won: {:?}",
            decision.reasons
        );
    }

    /// The reported bug, with the coordinates that actually produced it.
    ///
    /// The installed package could not read `Q4_K_M` out of its file name, so
    /// it recorded the container word "GGUF" and that is what the choice was
    /// saved as. The registry says `Q4_K_M`. The two never matched, the rank
    /// stayed at zero, and the chat answered from the largest model that fitted
    /// — with the Models screen still showing the star beside the chosen one.
    #[test]
    fn a_choice_saved_with_a_placeholder_quantisation_is_still_honoured() {
        let registry = stocked();
        let saved = StartupModelTarget {
            provider_id: "huggingface".to_string(),
            model_id: "qwen-8b".to_string(),
            // What was really in config.json on the machine this was found on.
            quantization: "GGUF".to_string(),
        };

        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
            Some(&saved),
        )
        .unwrap();

        assert_eq!(
            decision.model_id, "qwen-8b",
            "qwen-32b is larger and also fits, and used to win this"
        );
    }

    /// The tolerance above only relaxes the quantisation. A choice naming a
    /// different package is still a different package.
    #[test]
    fn a_placeholder_quantisation_does_not_match_a_different_model() {
        let registry = stocked();
        let saved = StartupModelTarget {
            provider_id: "huggingface".to_string(),
            model_id: "qwen-8b".to_string(),
            quantization: "GGUF".to_string(),
        };

        assert!(
            !is_orchestrator(
                registry.find("qwen-32b").expect("registered"),
                Some(&saved)
            ),
            "a loose quantisation must not make every model the orchestrator"
        );
    }

    /// The exact shape of the reported bug: a provisioning script had written
    /// an `orchestrator.*` entry into the live manifest, so every chat answered
    /// from that model no matter which one the administrator picked in Models.
    /// A choice made today outranks a tag written at some point in the past.
    #[test]
    fn a_stale_manifest_tag_does_not_outrank_the_administrator_choice() {
        let mut r = stocked();
        // A leftover tag on the 32B model — larger, so it also wins on size.
        if let Some(stale) = r.entries.iter_mut().find(|e| e.id == "qwen-32b") {
            stale.id = "orchestrator.qwen-32b".to_string();
        }

        let decision = ModelRouter::route_with_orchestrator(
            &r,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
            Some(&chose("qwen-8b")),
        )
        .unwrap();

        assert_eq!(
            decision.model_id, "qwen-8b",
            "the tagged entry is stale; the administrator chose qwen-8b"
        );
    }

    /// The choice is a preference, not a bypass. A chat model chosen for
    /// general work is not thereby a coding model, and a coding prompt still
    /// routes to the coding specialist.
    #[test]
    fn a_choice_that_cannot_serve_the_role_does_not_hijack_the_decision() {
        let registry = stocked();

        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Refactor this Python function and write a unit test for the null pointer case",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
            // An OCR model, which serves neither coding nor reasoning.
            Some(&chose("surya")),
        )
        .unwrap();

        assert_eq!(decision.role, ModelRole::Coding);
        assert_eq!(decision.model_id, "qwen-coder-14b");
    }

    /// A choice slightly too big for the GPU is still the administrator's
    /// choice: run it partly on the CPU rather than silently answering from a
    /// different model.
    #[test]
    fn the_choice_survives_a_gpu_too_small_to_hold_it() {
        let registry = stocked();

        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            6 * GB,
            None,
            false,
            &[],
            &[],
            Some(&chose("qwen-32b")),
        )
        .unwrap();

        assert_eq!(decision.model_id, "qwen-32b");
        assert!(decision.used_fallback, "it does not fit, and the trace must say so");
        assert!(!decision.fully_on_gpu);
    }

    /// The gap between "the orchestrator fits" and "nothing fits".
    ///
    /// Both of those already had tests and both already passed. The case
    /// between them did not: an orchestrator too large for VRAM, on a machine
    /// where a *smaller* candidate fits entirely. The full-offload search ran
    /// first, the choice failed its `if`, and the smaller model answered — so
    /// the Models screen showed the star beside one model while another did the
    /// talking, with a reason line that said "the largest cleared model that
    /// fits in VRAM" and never mentioned that a choice had been overruled.
    ///
    /// 20 GB holds the 8B (≈4.8 GB of weights) with room to spare and cannot
    /// hold the 32B (≈19.2 GB) once the KV cache for 32k tokens is charged.
    #[test]
    fn the_choice_wins_even_when_a_smaller_model_would_have_fitted_and_it_does_not() {
        let registry = stocked();

        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            20 * GB,
            None,
            false,
            &[],
            &[],
            Some(&chose("qwen-32b")),
        )
        .unwrap();

        assert_eq!(
            decision.model_id, "qwen-32b",
            "the administrator's choice lost to a model that merely fitted better"
        );
        assert!(
            !decision.fully_on_gpu,
            "this test is only meaningful while the choice does not fully fit"
        );
        assert!(
            decision.used_fallback,
            "running partly on the CPU is a degraded answer and the trace must say so"
        );
        // Confirm the smaller model really was available, or the test would
        // pass for the wrong reason on a future change to the VRAM heuristic.
        let smaller = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            20 * GB,
            None,
            false,
            &[],
            &[],
            None,
        )
        .unwrap();
        assert_eq!(smaller.model_id, "qwen-8b");
        assert!(smaller.fully_on_gpu);
    }

    /// A choice that fails a *hard* gate is still overruled. The fix above
    /// makes VRAM fit a preference; it does not make the choice a bypass.
    #[test]
    fn a_choice_that_fails_a_hard_gate_still_loses_when_it_does_not_fit() {
        let registry = stocked();

        // `surya` serves document OCR only, so it is not a reasoning candidate
        // at any VRAM size and cannot win a reasoning prompt.
        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            20 * GB,
            None,
            false,
            &[],
            &[],
            Some(&chose("surya")),
        )
        .unwrap();

        assert_eq!(decision.model_id, "qwen-8b");
    }

    /// With nobody having chosen, the router is back to judging on capability
    /// alone — no compiled-in favourite quietly winning.
    #[test]
    fn no_choice_means_the_largest_that_fits_still_wins() {
        let registry = stocked();

        let decision = ModelRouter::route_with_orchestrator(
            &registry,
            "Explain the trade-offs here",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
            None,
        )
        .unwrap();

        assert_eq!(decision.model_id, "qwen-32b");
    }

    /// If the orchestrator does not fit in VRAM, fall back to the largest
    /// cleared model that does — do not refuse to work.
    #[test]
    fn orchestrator_falls_back_when_it_does_not_fit() {
        let mut r = stocked();
        if let Some(qwen8b) = r.entries.iter_mut().find(|e| e.id == "qwen-8b") {
            qwen8b.id = "orchestrator.qwen-8b".to_string();
        }
        // 6 GB cannot hold the 8B orchestrator (we have a 4 GB-per-param
        // heuristic; 8B would not fit in 6 GB).
        let decision = ModelRouter::route(
            &r,
            "Explain the trade-offs here",
            None,
            6 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        // The 7B coding model is also unavailable for reasoning, so the
        // router falls back to the next-best reasoning model that fits.
        // We only assert used_fallback is true (the exact choice depends
        // on the vram planner's heuristic).
        assert!(decision.used_fallback);
    }

    /// A laptop is the case the problem statement explicitly allows for.
    #[test]
    fn a_small_gpu_falls_back_and_says_so() {
        let registry = stocked();
        let decision = ModelRouter::route(
            &registry,
            "Explain the trade-offs here",
            None,
            6 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-8b", "the smallest cleared candidate");
        assert!(decision.used_fallback);
    }

    #[test]
    fn routing_to_a_role_directly_skips_classification() {
        let registry = stocked();
        let decision = ModelRouter::route_for_role(
            &registry,
            ModelRole::DocumentOcr,
            None,
            8 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "surya");
    }

    #[test]
    fn an_empty_registry_explains_what_to_do() {
        let registry = registry(vec![]);
        let failure =
            ModelRouter::route(&registry, "anything", None, 24 * GB, None, false, &[], &[])
                .unwrap_err();
        assert!(failure.reason.contains("No models are registered"), "{}", failure.reason);
    }

    #[test]
    fn every_model_below_the_floor_says_so_rather_than_blaming_availability() {
        let registry = registry(vec![entry("tiny-coder", 1.5, vec![ModelRole::Coding])]);
        let failure = ModelRouter::route(
            &registry,
            "Refactor this Python function and fix the stack trace",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(failure.reason.contains("too small"), "{}", failure.reason);
    }

    #[test]
    fn a_disabled_model_produces_a_disabled_explanation() {
        let mut off = entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]);
        off.enabled = false;
        let registry = registry(vec![off]);
        let failure = ModelRouter::route(
            &registry,
            "Summarise this",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(failure.reason.contains("disabled"), "{}", failure.reason);
    }

    #[test]
    fn an_uncleared_classification_names_the_material() {
        let mut restricted = entry("qwen-8b", 8.0, vec![ModelRole::Reasoning]);
        restricted.permitted_classifications = vec![Classification::Internal];
        let registry = registry(vec![restricted]);

        let failure = ModelRouter::route(
            &registry,
            "Summarise this",
            Some(Classification::VendorNegotiation),
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        assert!(failure.reason.contains("Vendor negotiation"), "{}", failure.reason);
    }

    /// The contract from the prompt: when no model is cleared for the
    /// material, the refusal must (a) name the classification, (b) name
    /// the model that needs administrator review, and (c) not look
    /// like a generic "no model" message. Each of those is what a
    /// plant safety officer needs to act on the refusal in a hurry.
    #[test]
    fn refusal_names_the_classification_and_a_candidate_for_review() {
        let mut cleared = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        cleared.permitted_classifications = vec![Classification::Internal];
        let registry = registry(vec![cleared]);

        let failure = ModelRouter::route(
            &registry,
            "Summarise the vendor's offer",
            Some(Classification::VendorNegotiation),
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap_err();
        let reason = &failure.reason;
        assert!(
            reason.contains("Vendor negotiation"),
            "refusal must name the classification, got: {reason}"
        );
        assert!(
            reason.contains("qwen-7b"),
            "refusal must name a model an administrator can review, got: {reason}"
        );
    }

    /// Whatever else changes, a decision must always be explainable.
    #[test]
    fn every_successful_decision_carries_its_reasons() {
        let registry = stocked();
        for prompt in [
            "Refactor this Python function",
            "Summarise the inspection findings",
            "hello",
        ] {
            let decision =
                ModelRouter::route(&registry, prompt, None, 24 * GB, None, false, &[], &[])
                    .unwrap();
            assert!(
                !decision.reasons.is_empty(),
                "no reasons recorded for {prompt:?}"
            );
        }
    }
    // ── Routing preferences: a deterministic tie-breaker ────────────────

    /// "Always use the best model" is implemented as `routing.preferred`
    /// on a model entry. Two same-size peers, one preferred, the
    /// preferred one wins. The hard gates still run first — preferred
    /// is a tie-break, not a bypass.
    #[test]
    fn the_preferred_model_wins_a_size_tie() {
        let mut preferred = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        preferred.routing.preferred = true;
        let other = entry("mistral-7b", 7.0, vec![ModelRole::Reasoning]);
        let registry = registry(vec![preferred, other]);

        let decision = ModelRouter::route(
            &registry,
            "Explain the trade-offs here",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-7b");
    }

    /// `rank_within_band` is the telemetry-driven tie-breaker. Lower
    /// rank wins. The point is to let the per-model performance sink
    /// influence routing without changing the hard gates.
    #[test]
    fn a_lower_rank_within_band_wins_among_same_size_peers() {
        let mut faster = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        faster.routing.rank_within_band = 0;
        let mut slower = entry("mistral-7b", 7.0, vec![ModelRole::Reasoning]);
        slower.routing.rank_within_band = 5;
        let registry = registry(vec![slower, faster]);

        let decision = ModelRouter::route(
            &registry,
            "Summarise the inspection findings",
            None,
            24 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(decision.model_id, "qwen-7b");
    }

    /// Preferred is *only* a tie-breaker. A larger model that is not
    /// preferred still wins over a smaller preferred one, because size
    /// is the dominant sort key.
    #[test]
    fn preferred_does_not_bypass_the_size_band() {
        let mut small_preferred = entry("qwen-7b", 7.0, vec![ModelRole::Reasoning]);
        small_preferred.routing.preferred = true;
        let big = entry("qwen-14b", 14.0, vec![ModelRole::Reasoning]);
        let registry = registry(vec![small_preferred, big]);

        let decision = ModelRouter::route(
            &registry,
            "Summarise the inspection findings",
            None,
            80 * GB,
            None,
            false,
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(
            decision.model_id, "qwen-14b",
            "size dominates; preferred only breaks ties"
        );
    }}
