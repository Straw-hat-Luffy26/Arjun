//! Integration test: run the library scanner over a real model directory on
//! this machine and check it against an independent walk of the same tree.
//!
//! ## Why this is not a unit test with a fixture
//!
//! `scan_library` is the only thing standing between the weights an operator
//! copied onto the machine and the registry. It walks a real directory tree,
//! and real trees have the things fixtures do not: nested folders, sharded
//! files, projectors sitting beside their models, and whatever else somebody
//! put there. A fixture only ever contains what its author thought of.
//!
//! ## What it used to assert, and why that was wrong
//!
//! `assert!(found >= 25)` — the size of one developer's library, on a drive
//! (`F:\models`) that no other machine has. The file's own doc comment said it
//! was "skipped when the F:\models root is missing", and it was not: the
//! assertion ran whatever was found, so the test failed on every machine but
//! that one, and the comment describing it as skipped is what let that sit.
//!
//! A count nobody else can reproduce proves nothing about the scanner. What is
//! worth asserting is the property, and it holds for one file or a hundred:
//! **the scanner finds exactly the GGUFs that are there, splits the projectors
//! out correctly, and reports each file's real size.** That is checked here
//! against a second, independent walk.
//!
//! With no model directory on the machine there is nothing to compare against,
//! and the test says so and returns rather than asserting on an empty tree.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Every `.gguf` under `root`, found without using any of the code under test.
fn walk(root: &Path) -> Vec<PathBuf> {
    let mut found = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path
                .extension()
                .and_then(|e| e.to_str())
                .is_some_and(|e| e.eq_ignore_ascii_case("gguf"))
            {
                found.push(path);
            }
        }
    }
    found
}

fn is_projector(path: &Path) -> bool {
    path.file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.to_ascii_lowercase().starts_with("mmproj-"))
}

/// Where to look. `ARJUN_MODEL_ROOT` first, so this is runnable on a machine
/// that keeps its weights somewhere else without editing the test.
fn roots() -> Vec<PathBuf> {
    if let Ok(configured) = std::env::var("ARJUN_MODEL_ROOT") {
        return vec![PathBuf::from(configured)];
    }
    [r"F:\models", r"C:\Users\lenovo\models"]
        .iter()
        .map(PathBuf::from)
        .collect()
}

#[test]
fn the_scanner_agrees_with_the_directory_it_scanned() {
    let present: Vec<PathBuf> = roots().into_iter().filter(|r| r.is_dir()).collect();
    if present.is_empty() {
        // Not a silent pass: a reader of the output needs to know that the
        // strongest check in this file did not run.
        println!(
            "no model directory on this machine (looked in {:?}); \
             set ARJUN_MODEL_ROOT to run this against a real library",
            roots()
        );
        return;
    }

    let mut checked = 0usize;
    for root in &present {
        let scanned = sarathi_lib::registry::scan::scan_library(root);
        let actual = walk(root);

        let expected_models: BTreeSet<&PathBuf> =
            actual.iter().filter(|p| !is_projector(p)).collect();
        let expected_projectors: BTreeSet<&PathBuf> =
            actual.iter().filter(|p| is_projector(p)).collect();

        let got_models: BTreeSet<&PathBuf> = scanned.ggufs.iter().map(|g| &g.path).collect();
        let got_projectors: BTreeSet<&PathBuf> = scanned.mmprojs.iter().collect();

        println!(
            "{}: {} models, {} projectors",
            root.display(),
            got_models.len(),
            got_projectors.len()
        );

        assert_eq!(
            got_models, expected_models,
            "the scanner and an independent walk of {} disagree about which files are models",
            root.display()
        );
        assert_eq!(
            got_projectors, expected_projectors,
            "a projector was counted as a model, or the other way round, under {}",
            root.display()
        );

        for gguf in &scanned.ggufs {
            // The size is what the VRAM planner divides up. A zero here reads
            // as "this model costs nothing" and it would be admitted anywhere.
            let real = std::fs::metadata(&gguf.path)
                .unwrap_or_else(|e| panic!("{} was scanned but cannot be read: {e}", gguf.path.display()))
                .len();
            assert_eq!(
                gguf.bytes,
                real,
                "{} was recorded as {} bytes and is {real}",
                gguf.path.display(),
                gguf.bytes
            );

            // Pairing is by sibling directory, which is what the llama.cpp
            // loader expects. A projector paired from another folder would
            // load, and produce wrong vision output rather than an error.
            if let Some(mmproj) = &gguf.mmproj_path {
                assert_eq!(
                    mmproj.parent(),
                    gguf.path.parent(),
                    "{} was paired with a projector from a different directory",
                    gguf.path.display()
                );
                assert!(
                    is_projector(mmproj),
                    "{} was paired with {}, which is not a projector",
                    gguf.path.display(),
                    mmproj.display()
                );
            }
        }
        checked += scanned.ggufs.len() + scanned.mmprojs.len();
    }

    println!("checked {checked} files across {} root(s)", present.len());
}
