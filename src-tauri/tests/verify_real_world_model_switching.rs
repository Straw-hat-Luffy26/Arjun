//! A fact learned once is still known after the model changes, and after the
//! app restarts.
//!
//! This is the end-to-end form of the thing this product is asked for: the
//! conversation belongs to the person, not to whichever model happened to be
//! resident when they said something. Real weights, real inference, no stubs —
//! a mock memory provider would answer from the fixture rather than from
//! anything the model was told.
//!
//! ## Three things were wrong with the version this replaces
//!
//! It named `huggingface` models nobody has, so it failed everywhere and read
//! like the loader was broken (see `common::installed_models`).
//!
//! It wrote into `%APPDATA%/com.sarathi.app` — the **live** profile database
//! of whoever ran it. A test that seeds a fact into a person's real memory
//! store and leaves it there is not a test, it is a side effect. The memory
//! manager takes its own directory, so it now gets a temporary one; only the
//! weights come from the real app data.
//!
//! And it asserted on a real person's name as the remembered fact, which meant
//! a model could pass by having read that name somewhere rather than by being
//! told it here. The fact is now one no model can know.

mod common;

use std::sync::Arc;

use sarathi_lib::ai_engine::manager::InferenceManager;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams};
use sarathi_lib::memory_engine::MemoryManager;

/// Nothing in any training set says this. A model that produces it was told it
/// by the memory engine on this turn; a model that guesses a plausible answer
/// fails, which is the point.
const SECRET_KEY: &str = "commissioning_tag";
const SECRET_VALUE: &str = "VX-7741-QRT";
const QUESTION: &str = "What is the commissioning tag? Answer with the tag only.";

fn asked() -> Vec<ChatMessage> {
    vec![ChatMessage {
        role: "user".to_string(),
        content: QUESTION.to_string(),
        timestamp: None,
    }]
}

/// Whether the model said the thing it was told, however it wrapped it.
fn recalled(answer: &str) -> bool {
    let normalised: String = answer
        .to_ascii_uppercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    normalised.contains(SECRET_VALUE)
}

/// Why this is `#[test]` with a hand-built runtime rather than `#[tokio::test]`
///
/// `MemoryManager` owns a Tokio runtime of its own, through the Python sidecar
/// provider. Dropping one runtime from inside another panics:
///
/// ```text
/// Cannot drop a runtime in a context where blocking is not allowed.
/// This happens when a runtime is dropped from within an asynchronous context.
/// ```
///
/// So the managers are built and dropped out here, where blocking is allowed,
/// and only the async work happens inside `block_on`. The restart is a scope
/// ending rather than a `drop()` call, which is also a truer model of it.
#[test]
fn a_remembered_fact_survives_a_model_switch_and_a_restart() {
    let Some(models) = common::need_models(2, "the model-switch memory test") else {
        return;
    };
    let app_data = common::app_data_dir();
    let rt = tokio::runtime::Runtime::new().expect("a runtime for the async calls");

    // The memory store lives here and is deleted with it. The weights still
    // come from the real app data directory, which is only ever read.
    let store = tempfile::tempdir().expect("a temporary directory for the memory store");
    let store_path = store.path().to_path_buf();

    let params = GenerationParams {
        // Low, because this is a recall question with one right answer.
        temperature: 0.2,
        max_tokens: 100,
        ..Default::default()
    };

    // --- the first session -------------------------------------------------
    let answered = {
        let inference = Arc::new(InferenceManager::new());
        let memory = Arc::new(MemoryManager::new(&store_path));

        memory
            .set_user_profile_fact(SECRET_KEY, SECRET_VALUE, "user_fact")
            .expect("the fact is written before anything is asked");

        // Ask the same question of two different models in turn. Whatever the
        // first one is told, the second must be told too.
        let mut answered: Vec<String> = Vec::new();
        for model in models.iter().take(2) {
            inference.unload_active_model_direct().ok();

            let info = inference
                .load_installed_model_direct(
                    &app_data,
                    &model.provider,
                    &model.id,
                    &model.quantization,
                )
                .unwrap_or_else(|e| panic!("{} is installed and did not load: {e:?}", model.id));
            println!(
                "\n--- {} ({}, template '{}') ---",
                info.model_name, info.quantization, info.chat_template
            );

            let injected = rt.block_on(async {
                memory.process_user_turn(QUESTION, None).await.ok();
                memory
                    .prepare_injected_messages(&asked(), QUESTION)
                    .await
                    .unwrap_or_else(|e| panic!("memory injection failed: {e:?}"))
            });

            // The fact must be in what the model is handed. If it is not, the
            // model cannot possibly answer, and the failure belongs to the
            // memory engine rather than to the model — worth separating here,
            // because the assertion below cannot tell them apart.
            let prompt = injected
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join("\n");
            assert!(
                prompt.contains(SECRET_VALUE),
                "the memory engine did not put the fact in front of {}; \
                 nothing about the model is being tested here",
                model.id
            );

            let mut answer = String::new();
            inference
                .generate_direct(&injected, &params, |chunk| answer.push_str(&chunk.text))
                .unwrap_or_else(|e| panic!("{} failed to generate: {e:?}", model.id));

            println!("  answered: {}", answer.trim());
            assert!(
                recalled(&answer),
                "{} was given the fact and did not repeat it back: {answer:?}",
                model.id
            );
            answered.push(model.id.clone());
        }
        inference.unload_active_model_direct().ok();
        answered
        // Both managers drop here, outside `block_on`.
    };

    assert_eq!(answered.len(), 2, "both models answered");
    assert_ne!(
        answered[0], answered[1],
        "the switch must be between two different models, or nothing was switched"
    );

    // --- after a restart ---------------------------------------------------
    // New managers, same store on disk. The fact went to SQLite and must
    // still be there.
    println!("\n--- after a restart ---");
    {
        let inference = Arc::new(InferenceManager::new());
        let memory = Arc::new(MemoryManager::new(&store_path));

        let model = &models[0];
        let info = inference
            .load_installed_model_direct(&app_data, &model.provider, &model.id, &model.quantization)
            .unwrap_or_else(|e| panic!("{} did not load after the restart: {e:?}", model.id));
        println!("  reloaded {}", info.model_name);

        let injected = rt.block_on(async {
            memory
                .prepare_injected_messages(&asked(), QUESTION)
                .await
                .expect("memory injection after the restart")
        });
        assert!(
            injected.iter().any(|m| m.content.contains(SECRET_VALUE)),
            "the fact did not survive the restart: it is not in the prompt at all"
        );

        let mut answer = String::new();
        inference
            .generate_direct(&injected, &params, |chunk| answer.push_str(&chunk.text))
            .expect("generation after the restart");

        println!("  answered: {}", answer.trim());
        assert!(
            recalled(&answer),
            "the fact did not survive the restart: {answer:?}"
        );

        inference.unload_active_model_direct().ok();
    }
}
