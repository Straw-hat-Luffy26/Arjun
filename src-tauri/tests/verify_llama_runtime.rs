//! Load, unload and reload every model installed on this machine.
//!
//! The property: the runtime can bring a model up through the whole stack, put
//! it down cleanly, and bring it up again. A leak or a half-released handle
//! shows up on the reload, which is why the reload is here and not in a unit
//! test with a stub.
//!
//! This used to name four models under a `huggingface` provider that this
//! build no longer has, whose weights are on nobody's machine. It asked for
//! what somebody once had rather than what is there, so it failed everywhere
//! and read like the loader was broken. It now asks the disk — see
//! `common::installed_models`.

mod common;

use sarathi_lib::ai_engine::manager::InferenceManager;

#[test]
fn every_installed_model_loads_unloads_and_reloads() {
    let Some(models) = common::need_models(1, "the runtime load/unload/reload audit") else {
        return;
    };
    let app_data = common::app_data_dir();
    let mgr = InferenceManager::new();

    // Two is enough to prove the property, and the property is about the
    // runtime rather than about any particular file. Every installed model is
    // a different test: this machine holds five, the largest a 35B at roughly
    // twenty gigabytes, and loading each of them twice is tens of minutes of
    // disk. `ARJUN_LOAD_ALL_MODELS=1` does the full sweep when that is what
    // somebody actually wants.
    let sweep = std::env::var("ARJUN_LOAD_ALL_MODELS").is_ok();
    let take = if sweep { models.len() } else { 2.min(models.len()) };
    println!(
        "{} model(s) installed under {}; testing {take}{}",
        models.len(),
        app_data.display(),
        if sweep { "" } else { " (set ARJUN_LOAD_ALL_MODELS=1 for all)" }
    );

    for model in models.iter().take(take) {
        println!(
            "\n--- {}/{} ({}) ---\n    {}",
            model.provider,
            model.id,
            model.quantization,
            model.weights.display()
        );

        let info = mgr
            .load_installed_model_direct(&app_data, &model.provider, &model.id, &model.quantization)
            .unwrap_or_else(|e| {
                panic!(
                    "{} is installed at {} and did not load: {e:?}",
                    model.id,
                    model.weights.display()
                )
            });
        println!(
            "  loaded: {} ({}) via {}, {} ctx, {} gpu layers",
            info.model_name, info.quantization, info.backend_used, info.context_length, info.gpu_layers
        );

        // The loader must have opened the file this test named, not another
        // one in the same package. A silently different quantisation would
        // pass every other assertion here.
        assert_eq!(
            std::path::Path::new(&info.file_path),
            model.weights.as_path(),
            "{} loaded a different file from the one that was asked for",
            model.id
        );
        assert!(
            info.context_length > 0,
            "{} reported a zero context length, which the VRAM planner divides by",
            model.id
        );

        mgr.unload_active_model_direct()
            .unwrap_or_else(|e| panic!("{} did not unload: {e:?}", model.id));

        // The reload is the actual test. The first load proves the path
        // resolves; this proves the unload released what it claimed to.
        mgr.load_installed_model_direct(&app_data, &model.provider, &model.id, &model.quantization)
            .unwrap_or_else(|e| panic!("{} did not reload after unloading: {e:?}", model.id));

        mgr.unload_active_model_direct()
            .unwrap_or_else(|e| panic!("{} did not unload the second time: {e:?}", model.id));

        println!("  load / unload / reload / unload: all clean");
    }
}
