//! The injected memory survives the chat template.
//!
//! `prepare_injected_messages` puts what the person is known to have said into
//! a system message. What actually reaches llama.cpp is that list rendered
//! through *the model's own* chat template, and templates differ: some render
//! a system turn, some fold it into the first user turn, and a template that
//! does neither drops it silently. The model then answers as though it was
//! never told, and every layer above reports success.
//!
//! So this checks the last step — the rendered string — for every installed
//! model's template, and then runs one real generation end to end so the claim
//! is not only about a string.
//!
//! ## What this file used to do
//!
//! It computed a SHA-256 of each rendered prompt and printed it. Four hashes,
//! four `println!`s, no assertion — a trace, not a test. The hashes are now
//! used for something: templating the same messages twice must produce the
//! same bytes, because a prompt that differs run to run cannot be reproduced
//! from a task record, and reproducing one is what the record is for.
//!
//! It also loaded models that are on nobody's machine, wrote its fixture into
//! the live profile database, and asserted on a real person's name. Those are
//! dealt with the same way as in `verify_real_world_model_switching`.

mod common;

use std::sync::Arc;

use sha2::{Digest, Sha256};

use sarathi_lib::ai_engine::manager::InferenceManager;
use sarathi_lib::ai_engine::traits::{ChatMessage, GenerationParams};
use sarathi_lib::memory_engine::MemoryManager;

const SECRET_KEY: &str = "commissioning_tag";
const SECRET_VALUE: &str = "VX-7741-QRT";
const QUESTION: &str = "What is the commissioning tag? Answer with the tag only.";

fn sha256(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// `#[test]` with a hand-built runtime, not `#[tokio::test]`: `MemoryManager`
/// owns a Tokio runtime through the Python sidecar provider, and dropping one
/// runtime inside another panics with "Cannot drop a runtime in a context
/// where blocking is not allowed". The manager is built and dropped out here
/// where blocking is allowed; only the async calls go through `block_on`.
#[test]
fn injected_memory_reaches_the_prompt_the_runtime_is_handed() {
    let Some(models) = common::need_models(1, "the prompt-injection trace") else {
        return;
    };
    let app_data = common::app_data_dir();
    let rt = tokio::runtime::Runtime::new().expect("a runtime for the async calls");

    // A temporary store. The live profile database is not a test fixture.
    let store = tempfile::tempdir().expect("a temporary directory for the memory store");
    let inference = Arc::new(InferenceManager::new());
    let memory = Arc::new(MemoryManager::new(&store.path().to_path_buf()));

    memory
        .set_user_profile_fact(SECRET_KEY, SECRET_VALUE, "user_fact")
        .expect("the fact is written before anything is asked");

    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: QUESTION.to_string(),
        timestamp: None,
    }];

    let injected = rt.block_on(async {
        let retrieved = memory.search_memories(QUESTION, None).await.unwrap_or_default();
        println!("retrieval returned {} memory/ies", retrieved.len());
        memory
            .prepare_injected_messages(&messages, QUESTION)
            .await
            .expect("memory injection")
    });
    assert!(
        injected.iter().any(|m| m.content.contains(SECRET_VALUE)),
        "the fact never entered the message list, so nothing below is about templating"
    );

    // Reading a template means loading the model that carries it, so this is
    // the expensive part and it is bounded. Two different templates is what
    // proves the assertion is about templating rather than about one model.
    // `ARJUN_LOAD_ALL_MODELS=1` checks every installed model.
    let take = if std::env::var("ARJUN_LOAD_ALL_MODELS").is_ok() {
        models.len()
    } else {
        2.min(models.len())
    };

    for model in models.iter().take(take) {
        let template = template_of(&inference, &app_data, model);
        let rendered = sarathi_lib::ai_engine::runtime::format_chat_prompt_with_template(
            &injected, &template,
        );

        assert!(
            rendered.contains(SECRET_VALUE),
            "{}'s chat template dropped the injected memory. The message list \
             carried it and the rendered prompt does not, so the model would \
             answer as though it had never been told.\n--- rendered ---\n{rendered}",
            model.id
        );

        // Deterministic. A task record that names a prompt hash is only
        // evidence if the same messages render to the same bytes.
        let again = sarathi_lib::ai_engine::runtime::format_chat_prompt_with_template(
            &injected, &template,
        );
        assert_eq!(
            sha256(&rendered),
            sha256(&again),
            "{} renders the same messages to different bytes on two calls",
            model.id
        );

        println!("  {} -> {} ({} chars)", model.id, sha256(&rendered), rendered.len());
    }

    // And once for real, so this is not only a claim about a string.
    let model = &models[0];
    inference
        .load_installed_model_direct(&app_data, &model.provider, &model.id, &model.quantization)
        .unwrap_or_else(|e| panic!("{} is installed and did not load: {e:?}", model.id));

    let mut answer = String::new();
    inference
        .generate_direct(
            &injected,
            &GenerationParams {
                temperature: 0.1,
                max_tokens: 60,
                ..Default::default()
            },
            |chunk| answer.push_str(&chunk.text),
        )
        .unwrap_or_else(|e| panic!("{} failed to generate: {e:?}", model.id));

    println!("\n{} answered: {}", model.id, answer.trim());
    let normalised: String = answer
        .to_ascii_uppercase()
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '-')
        .collect();
    assert!(
        normalised.contains(SECRET_VALUE),
        "{} was handed the fact in its prompt and did not repeat it: {answer:?}",
        model.id
    );

    inference.unload_active_model_direct().ok();
}

/// A model's chat template, which only the loader knows.
///
/// Loading is the only way to read it, so this loads and immediately unloads.
/// It is the expensive part of this test and the reason the render checks come
/// first: a template bug is found without waiting for the generation.
fn template_of(
    inference: &InferenceManager,
    app_data: &std::path::Path,
    model: &common::InstalledModel,
) -> String {
    let info = inference
        .load_installed_model_direct(app_data, &model.provider, &model.id, &model.quantization)
        .unwrap_or_else(|e| panic!("{} is installed and did not load: {e:?}", model.id));
    let template = info.chat_template.clone();
    inference.unload_active_model_direct().ok();
    template
}
