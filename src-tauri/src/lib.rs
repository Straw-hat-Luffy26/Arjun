//! Sarathi Main Library
//! Wires together all modules and sets up the Tauri application.

pub mod core;
pub mod database;
pub mod config;
pub mod deployment;
pub mod logging;
pub mod commands;
pub mod sovereignty;
pub mod artifacts;
pub mod audit;
pub mod documents;
pub mod health;
pub mod hooks;
pub mod identity;
pub mod knowledge;
pub mod agent_runtime;
pub mod orchestrator;
pub mod package;
pub mod policy;
pub mod registry;
pub mod serving;
pub mod skills;
pub mod subagents;

// Phase modules
pub mod system_analyzer;
pub mod model_recommendation;
pub mod model_manager;
pub mod model_providers;
pub mod download_manager;
pub mod ai_engine;
pub mod capability;
pub mod gateway;
pub mod model_intelligence;
pub mod model_package;
pub mod installer;
pub mod plugins;
pub mod memory_engine;
// `media_adapter` was removed. It was a second media abstraction alongside the
// attachment/OCR pipeline in `commands::ocr` and `ai_engine::ocr_*`, which is
// the one that is wired and which every attachment actually goes through.
//
// Nothing outside the module ever referenced it, and its two extraction
// functions returned typed *successes* carrying empty findings and the note
// "not yet implemented in this phase". A caller reading `Ok(MultimodalResult)`
// with `needs_human_review: true` and no findings cannot tell a document with
// nothing in it from a reader that does nothing — which is the shape of silent
// failure this repository's own rules forbid.
pub mod sih_workflow;
pub mod voice;
pub mod benchmarks;

use std::sync::Arc;
use tauri::Manager;
use tauri_plugin_log::{Target, TargetKind};
use tauri_plugin_sql::Builder as SqlBuilder;
use log::info;

use download_manager::DownloadManager;
use ai_engine::InferenceManager;
use memory_engine::MemoryManager;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // Set up crash handler
    logging::setup_panic_handler();

    // Initialize core and managers
    let sarathi_core = core::init();
    let download_manager = Arc::new(DownloadManager::new());
    let inference_manager = Arc::new(InferenceManager::new());

    // Configure SQL plugin with migrations
    let migrations = database::get_migrations();
    let sql_plugin = SqlBuilder::default()
        .add_migrations("sqlite:sarathi.db", migrations)
        .build();

    // Configure Log plugin
    //
    // Three things here are about how fast a chat turn feels, not about
    // diagnostics.
    //
    // **The default level is `Trace`.** Left unset, every `hyper`, `reqwest`,
    // `sqlx` and `h2` trace line was recorded: in a captured session log, 232
    // of 291 lines were third-party noise, including a `connecting to
    // 127.0.0.1` triple for every single HTTP request the app made.
    //
    // **`TargetKind::Webview` sends each of those lines to the front-end over
    // IPC** — the same channel the token stream uses. During generation the two
    // compete, and the tokens are the ones a person is watching. It is kept in
    // debug builds, where a developer wants the console, and dropped from
    // release.
    //
    // **The chatty crates are pinned to `Warn` regardless**, because raising
    // the global level to `Info` still leaves `reqwest` free to log every
    // connection at `Debug`.
    let mut log_targets = vec![
        Target::new(TargetKind::Stdout),
        Target::new(TargetKind::LogDir { file_name: Some("sarathi".into()) }),
    ];
    if cfg!(debug_assertions) {
        log_targets.push(Target::new(TargetKind::Webview));
    }

    let log_plugin = tauri_plugin_log::Builder::new()
        .targets(log_targets)
        .level(if cfg!(debug_assertions) {
            log::LevelFilter::Debug
        } else {
            log::LevelFilter::Info
        })
        // ARJUN's own modules keep their level; these are the ones whose
        // per-request chatter drowned it.
        .level_for("hyper", log::LevelFilter::Warn)
        .level_for("hyper_util", log::LevelFilter::Warn)
        .level_for("reqwest", log::LevelFilter::Warn)
        .level_for("h2", log::LevelFilter::Warn)
        .level_for("rustls", log::LevelFilter::Warn)
        .level_for("sqlx", log::LevelFilter::Warn)
        .level_for("tokio_util", log::LevelFilter::Warn)
        .level_for("tower", log::LevelFilter::Warn)
        .build();

    // Build and run the app
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .plugin(sql_plugin)
        .plugin(log_plugin)
        .manage(sarathi_core)
        .manage(download_manager)
        // Cloned rather than moved: the activator built in setup below needs
        // the same manager the commands see, not a second one.
        .manage({
            // Named for `serving::admission`, which has to be able to release
            // the in-process model to make room for a served one. See
            // `ai_engine::manager::register_global`.
            ai_engine::manager::register_global(inference_manager.clone());
            inference_manager.clone()
        })
        .setup(move |app| {
            info!("Sarathi application starting...");

            // Where the installer put the files this build ships. Recorded
            // before anything can want one, because `deployment` is the only
            // module that knows how to find a sidecar and it is deliberately
            // Tauri-free — this is the one place the two meet.
            //
            // A checkout has no resource directory and does not get one; the
            // module's checkout fallback covers that case and the preflight
            // reports it as development-only rather than as working.
            match app.path().resource_dir() {
                Ok(dir) => deployment::set_resource_dir(dir),
                Err(error) => log::warn!(
                    "[DEPLOYMENT] no packaged resource directory ({error}); \
                     bundled scripts will be looked for in the checkout"
                ),
            }
            for status in deployment::preflight() {
                if let Some(remedy) = &status.remedy {
                    log::warn!(
                        "[DEPLOYMENT] {} ({}) is not properly installed: {:?}. {remedy}",
                        status.label,
                        status.needed_for,
                        status.resolution,
                    );
                }
            }

            // Resolve app_data_dir dynamically from Tauri app handle
            let app_data_dir = app
                .path()
                .app_data_dir()
                .unwrap_or_else(|_| std::path::PathBuf::from("./app_data"));
            let memory_manager = Arc::new(MemoryManager::new(&app_data_dir));
            // The single outbound chokepoint. Managed before anything that could
            // want the network, so no module can come up with its own client.
            // Governance state. The audit log is opened first so that everything
            // after it — including the broker's own decisions — is on the record.
            let data_dir = app
                .path()
                .app_data_dir()
                .expect("the application data directory must resolve");
            // Expose the resolved data dir to commands that need to read or
            // write files outside the audit DB (e.g. the provenance HMAC key
            // and tag files live here).
            app.manage(data_dir.clone());
            match audit::AuditService::open(&data_dir) {
                Ok(service) => {
                    let service = Arc::new(service);
                    sovereignty::global_broker().attach_audit(service.clone());
                    app.manage(service.clone());
                    // The zero-trust gate is opened next to the audit log
                    // because every gate decision is recorded in the log.
                    match sovereignty::zero_trust::ZeroTrustGate::open(&data_dir, service) {
                        Ok(gate) => {
                            app.manage(Arc::new(gate));
                        }
                        Err(e) => {
                            log::error!("[ZERO-TRUST] could not open the gate: {e}");
                        }
                    }
                }
                Err(e) => {
                    // Running without a durable record is a real degradation, so
                    // it is logged at error level rather than passed over.
                    log::error!("[AUDIT] could not open the audit log: {e}");
                }
            }
            match identity::CredentialStore::open(&data_dir) {
                Ok(store) => {
                    app.manage(Arc::new(store));
                }
                Err(e) => log::error!("[IDENTITY] could not open the credential store: {e}"),
            }
            app.manage(Arc::new(identity::UserDirectory::seeded()));

            // The model registry is a manifest beside the models on disk. A
            // missing one is an empty registry, not a failure — a fresh install
            // legitimately has nothing to route to until somebody provisions it.
            // The activator owns model swapping: routing chooses, this loads.
            app.manage(std::sync::Arc::new(ai_engine::activation::ModelActivator::new(
                ai_engine::activation::InferenceLoader::new(
                    inference_manager.clone(),
                    data_dir.clone(),
                ),
            )));

            match registry::ModelRegistry::load_with_discovery(&data_dir) {
                Ok(loaded) => {
                    // Logged on every start, including — especially — when it is
                    // zero. An empty registry turns into "no models are
                    // registered yet" on the workbench, which sends somebody off
                    // to import models they may already have; this line says
                    // where it looked, so the next person can tell an empty
                    // directory from an unreadable manifest in one glance.
                    info!(
                        "[REGISTRY] {} model(s) available from {}",
                        loaded.all().len(),
                        data_dir.join("models").display()
                    );
                    app.manage(Arc::new(loaded));
                }
                Err(e) => {
                    log::error!("[REGISTRY] the model manifest could not be read: {e}");
                    app.manage(Arc::new(
                        registry::ModelRegistry::load(std::path::Path::new("./__absent__"))
                            .expect("an absent manifest always loads as empty"),
                    ));
                }
            }
            // The knowledge index is the same SQLite file the rest of the app
            // uses. It is managed here so the health panel can count documents
            // without opening a second connection per request.
            match knowledge::index::KnowledgeIndex::open(&data_dir) {
                Ok(index) => {
                    app.manage(Arc::new(index));
                }
                Err(e) => log::error!("[KNOWLEDGE] the index could not be opened: {e}"),
            }

            // The multimodal half of the same index: page regions, table
            // chunks, and the passages that resolve a citation into a drawing
            // rather than into prose.
            //
            // It was never constructed outside tests, so
            // `knowledge.multimodal_retrieve` was a tool in the catalogue with
            // nothing behind it. Managed here, beside the text index, because
            // it is the same SQLite file and a second connection per request
            // would be a second lock on it.
            match knowledge::MultimodalIndex::open(&data_dir) {
                Ok(index) => {
                    app.manage(Arc::new(index) as commands::agent::Multimodal);
                }
                Err(e) => {
                    log::error!("[KNOWLEDGE] the multimodal index could not be opened: {e}")
                }
            }

            let approval_queue = Arc::new(orchestrator::approvals::ApprovalQueue::new());
            app.manage(Arc::clone(&approval_queue));
            app.manage(commands::governance::CurrentSession::default());
            // The telemetry sink: per-model call records, written to the
            // audit log on each call so the Model Health page reads from
            // a single signed chain rather than a separate database.
            let telemetry_sink =
                Arc::new(model_intelligence::telemetry::TelemetrySink::new());
            // Diagnostic: prove the sink is registered and the lookup
            // from the inference path will resolve. If the page is
            // empty after a chat, this log line is the first thing to
            // check; the path that records calls goes through
            // `app_handle.try_state::<Arc<TelemetrySink>>` and would
            // hit the same lookup mechanism.
            // `eprintln!` is used deliberately: the Tauri log plugin's
            // flush is lazy, and `log::info!` from the `log` crate was
            // observed to not reach `sarathi.log` for these new lines.
            // `eprintln!` writes to stderr, which `Start-Process` on
            // Windows exposes via the parent's handle when the parent
            // is a console host. The PowerShell wrapper captures it.
            eprintln!(
                "[telemetry] sink registered; default seq = 0; \
                 snapshot at startup = {} rows",
                telemetry_sink.snapshot().len()
            );

            // A synthetic `<startup>` row used to be written here — a
            // `ModelCallRecord` with `exit: Ok`, inserted so the Model Health
            // page would be non-empty after launch and the IPC chain could be
            // seen working.
            //
            // It was a model call that never happened, in the history of model
            // calls. A fresh installation reported one successful inference
            // before anything had been asked of it; every average latency,
            // every success rate and every fallback ratio on that page was
            // computed over a row describing nothing. This repository ships
            // evidence to judges, and a diagnostic that fabricates a
            // measurement is the one kind of diagnostic it cannot have.
            //
            // What it was actually for — proving the sink, the IPC and the
            // page are wired — is answered by `agent_telemetry_health`, which
            // reports the wiring without adding to what it is reporting on.
            app.manage(telemetry_sink);
            // Started on first run rather than here: the workbench must open for an
            // auditor, and on a machine where the runtime bundle was never built.
            app.manage(commands::agent::AgentRuntimeHandle::default());
            // Model servers ARJUN starts, so a llama-server is loaded once and
            // reused across runs rather than per prompt.
            app.manage(Arc::new(serving::ModelServers::new()));
            app.manage(commands::ocr::ScanCancel::default());
            // One working directory per run, shared with the agent runtime so a
            // tool call can be resolved against the run that made it.
            app.manage(commands::agent::RunWorkspaces::default());
            // The rest of a run's working state, held here for the same reason:
            // the command that starts a run has to read all of it back when the
            // run ends, to write the task's record.
            app.manage(commands::agent::RunPlans::default());
            app.manage(commands::agent::RunCalculations::default());
            app.manage(commands::agent::RunToolCalls::default());
            // Scoped memory: what this machine remembers, and for whom. Opened
            // under the same data directory as the index and the task records,
            // and lazily per scope — a deployment with two hundred projects
            // should not pay for two hundred file reads to start.
            app.manage(std::sync::Arc::new(agent_runtime::memory::MemoryStore::open(
                &data_dir,
            )) as commands::agent::AgentMemory);
            // The fixed half of each live run's checkpoint. Dies with the
            // process on purpose: a seed describes a world observed at start,
            // and after a restart that world has to be observed again.
            app.manage(commands::agent::RunCheckpoints::default());
            app.manage(agent_runtime::retrieval::RunPassages::default());
            app.manage(agent_runtime::artifacts::RunArtifacts::default());

            // Chat conversations: persistent ordered transcripts that own
            // one or more runs. Sits beside the task record; the two are
            // complementary (the task record is the audit-grade proof a
            // run happened; the conversation is the user-visible chat).
            //
            // ## When the store cannot be opened
            //
            // The fallback used to be a *fixed* temp directory,
            // `arjun-conversations-fallback`. Three things were wrong with it,
            // and the third is the serious one:
            //
            //  - it is shared, so two sessions — or two users on one machine —
            //    wrote into each other's conversations;
            //  - it is stale, so a session that recovered found the last
            //    degraded session's threads sitting there looking like history;
            //  - nothing said so. The chat behaved exactly as normal, and the
            //    person's real conversations were not in it.
            //
            // So a failure now opens a session-unique directory, and the state
            // records that this is what happened. `ConversationHealth::refusal`
            // is what the commands consult before creating anything new: the
            // application stays usable and readable, and a person is not
            // quietly given a transcript that will be gone tomorrow.
            let (conversation_store, conversation_health) =
                match agent_runtime::conversations::ConversationStore::open(&data_dir) {
                    Ok(store) => (
                        std::sync::Arc::new(store),
                        agent_runtime::conversations::ConversationHealth::durable(),
                    ),
                    Err(error) => {
                        log::error!(
                            "[CONVERSATIONS] the store could not be opened, so this session's \
                             chats are ephemeral: {error}"
                        );
                        // Unique per session, and per process, so nothing
                        // written here can be mistaken for history or read by
                        // the next session.
                        let scratch = std::env::temp_dir().join(format!(
                            "arjun-conversations-ephemeral-{}-{}",
                            std::process::id(),
                            chrono::Utc::now().timestamp_nanos_opt().unwrap_or_default(),
                        ));
                        let store =
                            agent_runtime::conversations::ConversationStore::open(&scratch)
                                .expect("a scratch conversation store must open");
                        (
                            std::sync::Arc::new(store),
                            agent_runtime::conversations::ConversationHealth::ephemeral(
                                format!("The conversation store could not be opened: {error}"),
                                scratch,
                            ),
                        )
                    }
                };
            app.manage(commands::conversations::ConversationHealthState(
                std::sync::Arc::new(conversation_health),
            ));
            let run_to_conversation =
                std::sync::Arc::new(agent_runtime::conversations::RunToConversation::new());
            app.manage(commands::conversations::ConversationsState(conversation_store));
            app.manage(commands::conversations::RunToConversationState(run_to_conversation));

            // The durable half of all of the above. Everything managed just
            // now dies with this process; this is what a run leaves behind
            // while it is still going, and it is opened before any command can
            // run so that no run starts unrecorded.
            //
            // A log that cannot be opened does not stop the window opening —
            // a person needs to be able to read what is already there, change
            // settings, and find out what is wrong. It does stop work: the
            // in-memory substitute keeps the application usable read-only, and
            // `AuditHealth` refuses every run and every side-effecting tool
            // for as long as it is in use. Before that split existed, a failed
            // open produced an application that looked entirely normal and
            // kept a history that evaporated when the process exited.
            let audit_health: std::sync::Arc<agent_runtime::audit_health::AuditHealth>;
            let task_events: std::sync::Arc<agent_runtime::events::TaskEventLog> =
                match agent_runtime::events::TaskEventLog::open(&data_dir) {
                    Ok(log) => {
                        audit_health = std::sync::Arc::new(
                            agent_runtime::audit_health::AuditHealth::durable(),
                        );
                        std::sync::Arc::new(log)
                    }
                    Err(error) => {
                        log::error!(
                            "[TASKS] the task event log could not be opened, so this session is                              read-only: {error}"
                        );
                        audit_health = std::sync::Arc::new(
                            agent_runtime::audit_health::AuditHealth::degraded_at_startup(format!(
                                "The task event log could not be opened: {error}"
                            )),
                        );
                        std::sync::Arc::new(
                            agent_runtime::events::TaskEventLog::in_memory()
                                .expect("an in-memory task event log"),
                        )
                    }
                };
            app.manage(commands::agent::AuditHealthState(std::sync::Arc::clone(
                &audit_health,
            )));

            // Runs that were going when the process last went away. Closed off
            // here, before anything else writes, so the Tasks screen never
            // shows a run that has been "running" since last Tuesday next to
            // one that is running now.
            // Approvals raised before this process started, put back in front
            // of somebody. Until they were durable, a crash while a person was
            // deciding lost both the question and the answer, and the operator
            // came back to a run that had stopped for a reason nothing recorded.
            match task_events.pending_approvals() {
                Ok(waiting) if !waiting.is_empty() => {
                    let requests = waiting
                        .into_iter()
                        .map(|approval| orchestrator::approvals::ApprovalRequest {
                            requested_by: task_events.snapshot(&approval.run_id).ok().flatten()
                                .map(|snapshot| snapshot.actor).unwrap_or_default(),
                            arguments: approval.display_arguments(),
                            id: approval.approval_id,
                            task_id: approval.run_id,
                            tool: approval.tool,
                            target: approval.target,
                            // Not stored: evidence is gathered by the run that
                            // asked, and that run is gone. Empty is honest.
                            evidence: Vec::new(),
                            expected_output: approval.reason,
                            consequences: String::new(),
                            requested_at: chrono::DateTime::parse_from_rfc3339(
                                &approval.created_at,
                            )
                            .map(|at| at.with_timezone(&chrono::Utc))
                            .unwrap_or_else(|_| chrono::Utc::now()),
                        })
                        .collect::<Vec<_>>();
                    let count = approval_queue.restore(requests);
                    info!(
                        "[TASKS] {count} approval(s) raised before this start are waiting for a                          decision"
                    );
                }
                Ok(_) => {}
                Err(error) => {
                    log::warn!("[TASKS] pending approvals could not be read back: {error}")
                }
            }

            match task_events.recover_interrupted(agent_runtime::events::SYSTEM_ACTOR) {
                Ok(recovered) if !recovered.is_empty() => {
                    info!(
                        "[TASKS] {} run(s) were interrupted by the last shutdown and have been \
                         closed off: {}",
                        recovered.len(),
                        recovered.join(", ")
                    );
                }
                Ok(_) => {}
                Err(error) => log::error!("[TASKS] interrupted runs could not be closed off: {error}"),
            }
            let subagent_events = std::sync::Arc::clone(&task_events);
            app.manage(task_events as commands::agent::TaskEvents);

            // Skills: reusable instructions an operator installs. Discovered
            // once at start, metadata only — reading every SKILL.md into memory
            // here would be the thing that puts every skill in front of every
            // model. See `skills::registry`.
            //
            // Resolved as a bundled resource in a packaged build and as the
            // sibling directory in a checkout, the same way the agent runtime
            // bundle is.
            let skills_dir = app
                .path()
                .resolve("skills", tauri::path::BaseDirectory::Resource)
                .ok()
                .filter(|path| path.is_dir())
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .map(|root| root.join("skills"))
                        .unwrap_or_default()
                });
            let skills = std::sync::Arc::new(skills::SkillRegistry::open(&skills_dir));
            let found = skills.snapshot();
            info!(
                "[SKILLS] {} of {} skill(s) available from {}",
                found.available(),
                found.count(),
                skills_dir.display()
            );
            for card in found.cards().iter().filter(|card| !card.is_available()) {
                // Quarantined skills are named at start rather than only when
                // somebody goes looking. An operator who installed one and
                // cannot find it should not have to open a screen to find out
                // why.
                if let Some(reason) = &card.quarantined {
                    log::warn!("[SKILLS] {} is quarantined: {}", card.name, reason.explain());
                }
            }
            app.manage(skills as commands::agent::Skills);

            // Subagent profiles: bounded workers a run may delegate to. Loaded
            // beside the skills and for the same reasons — an operator reads
            // and reviews a file, and Rust enforces what it declares.
            let agents_dir = app
                .path()
                .resolve("agents", tauri::path::BaseDirectory::Resource)
                .ok()
                .filter(|path| path.is_dir())
                .unwrap_or_else(|| {
                    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                        .parent()
                        .map(|root| root.join("agents"))
                        .unwrap_or_default()
                });
            let loaded_profiles = subagents::load_profiles(&agents_dir);
            info!(
                "[SUBAGENTS] {} profile(s) from {}",
                loaded_profiles.profiles.len(),
                agents_dir.display()
            );
            for rejected in &loaded_profiles.rejected {
                // Named at start rather than only when somebody goes looking.
                log::warn!(
                    "[SUBAGENTS] {} was not loaded: {}",
                    rejected.file,
                    rejected.error.explain()
                );
            }
            let subagent_manager =
                subagents::SubagentManager::new(loaded_profiles.profiles, subagent_events);

            let subagent_manager = if let Some(index) = app.try_state::<Arc<knowledge::KnowledgeIndex>>() {
                subagents::workers::register(subagent_manager, subagents::workers::Resources {
                    index: index.inner().clone(),
                    events: app.state::<commands::agent::TaskEvents>().inner().clone(),
                    session: app.state::<commands::governance::CurrentSession>().inner().clone(),
                    workspaces: app.state::<commands::agent::RunWorkspaces>().inner().clone(),
                    passages: app.state::<agent_runtime::retrieval::RunPassages>().inner().clone(),
                    calculations: app.state::<commands::agent::RunCalculations>().inner().clone(),
                    produced: app.state::<agent_runtime::artifacts::RunArtifacts>().inner().clone(),
                })
            } else { subagent_manager };

            let performable = subagent_manager
                .profiles()
                .filter(|profile| subagent_manager.has_worker(&profile.name))
                .count();
            let declared = subagent_manager.profiles().count();
            if performable < declared {
                log::warn!(
                    "[SUBAGENTS] {} of {} declared role(s) have no worker in this build; those profiles are unavailable: {}",
                    declared - performable,
                    declared,
                    subagent_manager
                        .profiles()
                        .filter(|profile| !subagent_manager.has_worker(&profile.name))
                        .map(|profile| profile.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                );
            }

            app.manage(std::sync::Arc::new(subagent_manager) as commands::agent::Subagents);

            app.manage(sovereignty::global_broker().clone());

            app.manage(memory_manager);

            // Load the saved HuggingFace token into the process before anything
            // reaches the Hub. Catalog browsing and artifact resolution run
            // from plain commands with no app handle, so they read it from there
            // rather than opening config.json themselves.
            {
                let config_path = crate::config::ConfigManager::get_config_path(app.handle());
                match crate::config::ConfigManager::load(&config_path) {
                    Ok(cfg) if !cfg.hf_token.trim().is_empty() => {
                        crate::config::hf_token::set(Some(cfg.hf_token));
                        info!("HuggingFace token loaded from settings");
                    }
                    Ok(_) => {
                        info!(
                            "No HuggingFace token in settings (environment: {})",
                            crate::config::hf_token::source()
                        );
                    }
                    Err(e) => log::warn!("Could not read config for HuggingFace token: {e:#}"),
                }
            }

            let pack_manager = Arc::new(crate::model_recommendation::pack_manager::PackManager::new(&app_data_dir).expect("Failed to initialize PackManager"));
            app.manage(pack_manager);

            // Serialize all model access behind one worker, then start the local
            // gateway so external tools (Claude Code, opencode, openclaw) can use
            // whichever model this app has loaded.
            let inference_for_gateway = app.state::<Arc<InferenceManager>>().inner().clone();
            let scheduler = Arc::new(ai_engine::scheduler::GenerationScheduler::start(
                inference_for_gateway.clone(),
            ));
            app.manage(scheduler.clone());

            let gateway_state = Arc::new(gateway::GatewayState::new(
                scheduler,
                inference_for_gateway,
                gateway::GatewayConfig::default(),
            ));
            app.manage(gateway_state.clone());

            // Tracks tools the Launch screen started, so cards can show Running.

            // Started only when it is turned on.
            //
            // `enabled` was never read before the socket was bound, so every
            // installation served the model over HTTP to anything that could
            // reach loopback — and turning it off in the UI changed nothing.
            // It now defaults to off; an operator turns it on and the gateway
            // is started then.
            let app_for_gateway = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                if !gateway_state.enabled() {
                    info!(
                        "Sarathi gateway is turned off, so nothing is listening. Turn it on in                          settings to let other tools on this machine use this model."
                    );
                    return;
                }
                match gateway::start_gateway(gateway_state).await {
                    Ok(handle) => {
                        info!(
                            "Sarathi gateway ready on http://127.0.0.1:{} — point Claude Code at /v1/messages, opencode at /v1/chat/completions",
                            handle.port
                        );
                        // Hand the handle to Tauri so it lives as long as the
                        // app. Letting it drop here would drop the shutdown
                        // sender, resolving the graceful-shutdown future and
                        // closing the server immediately after it announced
                        // itself — the logs would claim it was listening while
                        // every connection was refused.
                        app_for_gateway.manage(handle);
                    }
                    // A busy port must not stop the desktop app from opening;
                    // the user needs the UI to change the port.
                    Err(e) => log::error!("Gateway failed to start: {e:#}"),
                }
            });

            // Bring the configured orchestrator onto the GPU during startup so
            // the agent loop and local gateway are ready without a manual model
            // selection. The orchestrator is whatever an administrator chose in
            // Models — no model is compiled in as a default — and a saved
            // session is the fallback when nothing is chosen or the chosen
            // package is not installed.
            let ai_settings = config::ConfigManager::load(
                &config::ConfigManager::get_config_path(app.handle()),
            )
            .map(|config| config.ai_settings)
            .unwrap_or_default();

            if !ai_settings.auto_load_on_startup {
                info!(
                    "Startup auto-load is off — no model will be loaded until one is \
                     chosen in the app. Set ai_settings.auto_load_on_startup to change this."
                );
            }

            if ai_settings.auto_load_on_startup {
                let inference = app.state::<Arc<InferenceManager>>().inner().clone();
                let app_data = app.path().app_data_dir().ok();
                let preferred = ai_engine::startup::StartupModelTarget::configured(&ai_settings);
                let require_gpu = ai_settings.use_gpu;

                tauri::async_runtime::spawn(async move {
                    let Some(dir) = app_data else { return };

                    if require_gpu && !ai_engine::startup::gpu_backend_compiled() {
                        log::error!(
                            "Cannot auto-load the orchestrator on the GPU: this ARJUN binary was \
                             built without CUDA or Vulkan. Start it with `npm run dev:auto`, or \
                             build it with `npm run build:auto`."
                        );
                        return;
                    }

                    let restore = ai_engine::session::SessionManager::load_session(&dir)
                        .ok()
                        .flatten()
                        .filter(|s| s.auto_restore_enabled)
                        .map(|session| ai_engine::startup::StartupModelTarget {
                            provider_id: session.provider_id,
                            model_id: session.model_id,
                            quantization: session.quantization,
                        });

                    let installed = model_manager::ModelManager::list_installed_models(&dir);
                    let target = ai_engine::startup::select_startup_model(
                        &installed,
                        preferred.as_ref(),
                        restore.as_ref(),
                    );

                    let Some(target) = target else {
                        // Two different situations, and telling them apart is
                        // the whole value of the message: a choice that points
                        // at something not installed is a broken setting, while
                        // no choice at all is simply a step nobody has taken.
                        match preferred.as_ref() {
                            Some(preferred) => log::error!(
                                "The configured orchestrator '{}' ({}) is not installed and no \
                                 unambiguous fallback is available. Install it from Discover, or \
                                 choose another installed model in Models.",
                                preferred.model_id,
                                preferred.quantization
                            ),
                            None => info!(
                                "No orchestrator has been chosen and there is no single obvious \
                                 model to load, so nothing was pre-loaded. Pick one in Models \
                                 with 'Set as orchestrator'; until then each prompt is routed to \
                                 the best installed model for it."
                            ),
                        }
                        return;
                    };
                    let provider = target.provider_id;
                    let model = target.model_id;
                    let quant = target.quantization;
                    info!(
                        "Auto-loading orchestrator '{model}' ({quant}){}",
                        if require_gpu { " with GPU residency required" } else { "" }
                    );

                    let inference_for_load = inference.clone();
                    let res = tokio::task::spawn_blocking(move || {
                        inference_for_load.load_installed_model_direct(
                            &dir, &provider, &model, &quant,
                        )
                    })
                    .await;

                    match res {
                        Ok(Ok(info)) => {
                            if require_gpu {
                                if let Err(reason) =
                                    ai_engine::startup::validate_gpu_residency(info.gpu_layers)
                                {
                                    let _ = inference.unload_active_model_direct();
                                    log::error!("Orchestrator auto-load rejected: {reason}");
                                    return;
                                }
                            }
                            info!(
                                "Orchestrator ready: {} via {} with {} GPU layer(s) — gateway can now serve requests",
                                info.model_name, info.backend_used, info.gpu_layers
                            );
                        }
                        // A load failure must not take the app down; the UI still
                        // needs to open so the user can pick a different model.
                        Ok(Err(e)) => log::error!("Auto-load failed: {e:#}"),
                        Err(e) => log::error!("Auto-load task panicked: {e}"),
                    }
                });
            }

            // Initial event publication
            let event_bus = core::event_bus::get_event_bus();
            event_bus.publish(core::event_bus::SarathiEvent::ApplicationStarted, None);

            // Run initial system analysis task on a blocking thread (not a tokio async worker)
            // so it doesn't occupy the async runtime while running PowerShell/DXGI detection
            std::thread::spawn(move || {
                let analyzer = system_analyzer::get_system_analyzer_manager();
                if let Err(e) = analyzer.analyze_system() {
                    log::error!("Initial system analysis failed: {}", e);
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            // Config commands
            commands::ocr::get_ocr_detents,
            commands::ocr::preview_attachment_routing,
            commands::ocr::get_page_image,
            commands::ocr::scan_page,
            commands::ocr::cancel_scan,
            commands::config::get_config,
            commands::config::set_config,
            commands::config::get_hf_token_status,
            commands::config::set_hf_token,
            commands::config::get_config_value,
            commands::config::set_config_value,
            commands::config::get_default_config,
            commands::config::reset_config,
            commands::config::get_app_paths,

            // System commands
            commands::system::get_app_info,
            commands::system::get_app_state_info,
            commands::system::log_activity,
            commands::system::get_hardware_profile,
            commands::system::analyze_system,
            commands::system::override_hardware_value,
            commands::system::revert_hardware_override,
            commands::system::validate_system,

            // Recommendation & Certification commands (Phase 3 & Ecosystem)
            commands::recommendation::get_model_recommendations,
            commands::recommendation::get_package_certification,
            commands::recommendation::get_all_package_certifications,
            commands::recommendation::get_recommended_packages,
            commands::recommendation::get_compatible_packages,
            commands::recommendation::get_experimental_packages,
            commands::recommendation::get_runtime_profile,
            commands::recommendation::reload_certification_packs,

            // Download & Storage Management commands (Phase 4)
            commands::download::start_model_download,
            commands::download::pause_model_download,
            commands::download::resume_model_download,
            commands::download::cancel_model_download,
            commands::download::get_active_downloads,
            commands::download::get_installed_models,
            commands::download::delete_installed_model,
            commands::download::get_storage_summary,

            // Phase 5 Inference Commands
            commands::inference::load_installed_model,
            commands::inference::unload_active_model,
            commands::inference::get_inference_status,
            commands::inference::send_chat_message,
            commands::inference::stop_chat_generation,
            commands::inference::restore_last_session,

            // Model Intelligence Layer Commands
            commands::intelligence::get_model_profile,
            commands::intelligence::update_model_profile,
            commands::intelligence::refresh_model_profile,
            commands::intelligence::model_health_snapshot,

            // Launch section — start coding tools already connected

            // Model browsing by category
            // Agent runs. The loop lives in the Node runtime; these start,
            // stop and observe it.
            commands::agent::agent_start_run,
            commands::agent::agent_abort_run,
            commands::agent::agent_acknowledge_milestone,
            commands::knowledge::knowledge_documents,
            commands::knowledge::knowledge_search,
            commands::knowledge::knowledge_health,
            commands::agent::agent_steer_run,
            commands::agent::agent_runtime_health,
            // What those runs left behind: the plan, the evidence, the working
            // and the files.
            commands::agent::agent_task_history,
            commands::agent::agent_task,
            commands::agent::agent_task_artifacts,
            commands::agent::agent_reveal_artifact,
            commands::agent::artifact_preview,

            // Chat conversations: persistent ordered transcripts that own
            // one or more runs. The chat surface calls these to create a
            // conversation, list previous ones, append a user turn (which
            // also reserves the assistant cell and binds the run id), update
            // the streaming content as tokens arrive, and mark the message
            // complete when the run ends.
            commands::conversations::agent_create_conversation,
            commands::conversations::agent_conversation_health,
            commands::intelligence::agent_telemetry_health,
            commands::conversations::agent_get_conversation,
            commands::conversations::agent_list_conversations,
            commands::conversations::agent_delete_conversation,
            commands::conversations::agent_append_turn,
            commands::conversations::agent_update_streaming_content,
            commands::conversations::agent_complete_message,
            commands::conversations::agent_run_conversation,
            commands::conversations::agent_get_message,
            // Recovering a run: the state a window reattaches to, the events
            // since that state, and which runs are still going at all.
            commands::agent::agent_run_resumability,
            commands::agent::agent_resume_run,
            commands::agent::agent_pause_run,
            commands::agent::agent_task_snapshot,
            commands::agent::agent_task_events,
            commands::agent::agent_active_tasks,
            // Side effects that were in flight when the process went away, and
            // the person saying what actually happened to them.
            commands::agent::agent_unknown_effects,
            commands::agent::agent_reconcile_effect,
            // Skills: what is installed, and re-reading the directory.
            commands::agent::skill_search,
            commands::agent::skill_reload,
            commands::agent::subagent_profiles,

            commands::sovereignty::get_operating_mode,
            commands::sovereignty::set_operating_mode,
            commands::sovereignty::recent_egress_events,
            commands::sovereignty::run_egress_canary,
            commands::sovereignty::observe_process_connections,
            commands::sovereignty::assert_confidential_allowed,
            commands::governance::list_accounts,
            commands::governance::sign_in,
            commands::governance::sign_out,
            commands::governance::current_session,
            commands::governance::current_permissions,
            commands::governance::recent_audit_entries,
            commands::governance::verify_audit_chain,
            commands::governance::verify_audit_merkle,
            commands::governance::sign_provenance,
            commands::governance::verify_provenance,
            commands::governance::read_zero_trust_config,
            commands::governance::set_zero_trust_mode,
            commands::governance::zero_trust_check_tool_call,
            commands::governance::zero_trust_confirm_approval,
            // Voice bridge (push-to-talk STT/TTS)
            commands::voice::voice_transcribe,
            commands::voice::voice_speak,
            commands::voice::voice_status,
            // Performance benchmarks for the System Health page
            commands::benchmarks::run_benchmark,
            commands::benchmarks::synthetic_benchmark,
            commands::benchmarks::recent_benchmarks,
            commands::governance::authentication_status,
            commands::governance::set_initial_administrator_password,
            commands::governance::set_account_password,
            commands::registry::list_registered_models,
            commands::registry::list_library_models,
            commands::registry::detect_system_models,
            commands::registry::model_manifest_path,
            commands::registry::get_orchestrator_model,
            commands::registry::set_orchestrator_model,
            commands::registry::preview_routing,
            commands::registry::prepare_model_for,
            commands::registry::model_residency,
            commands::health::health_snapshot,
            commands::approvals::list_approvals,
            commands::approvals::decide_approval,
            commands::catalog::browse_model_cards,
            commands::catalog::list_model_categories,

            // The ten `memory_engine::api::*` commands were removed. See
            // `memory_engine::api` for the reasoning; in short, every one of
            // them proved that *somebody* was signed in and then read, wrote or
            // deleted the memory of *everybody*, and none of them had a
            // consumer.
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|handle, event| {
            // The polite half of model-server cleanup.
            //
            // `ModelServers::stop_all` has always documented itself as "Called
            // on shutdown so no server outlives the app" and was never called
            // by anything — the comment described an intention nobody had
            // wired. This is that wire.
            //
            // It is not the guarantee: an exit handler only runs on exits the
            // process gets to participate in, and the orphans that prompted
            // this were made by the ones it does not — force-kills, panics,
            // and a developer rebuilding over a running binary. That case is
            // covered by the job object in `serving::reaper`. This exists so
            // that on a normal quit the servers are asked to stop rather than
            // killed, which lets them release VRAM and flush their own logs.
            if let tauri::RunEvent::ExitRequested { .. } = event {
                if let Some(servers) = handle.try_state::<std::sync::Arc<serving::ModelServers>>() {
                    let servers = std::sync::Arc::clone(&servers);
                    tauri::async_runtime::block_on(async move {
                        servers.stop_all().await;
                    });
                }
            }
        });
}
