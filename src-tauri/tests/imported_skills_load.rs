//! Every skill in `skills/` loads, and nothing is quarantined.
//!
//! Eighteen skills were imported from an outside collection written for a
//! different agent. In that collection's own format **none** of them would
//! load here: `skills::manifest::validate` requires `author`, `compatibility`,
//! `network`, `classification` and `allowed-tools`, and all five were absent
//! from all ninety-three. The ones brought across had that frontmatter added
//! and their sha256 recorded in `trusted.json`.
//!
//! A quarantined skill fails quietly — it is simply never offered to a model —
//! so this is asserted rather than left to be noticed when a run does not use
//! a skill somebody installed.

use sarathi_lib::skills::SkillRegistry;

#[test]
fn every_shipped_skill_loads_and_is_trusted() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the repository root")
        .join("skills");
    assert!(root.is_dir(), "skills/ is missing at {}", root.display());

    let registry = SkillRegistry::open(&root);
    let snapshot = registry.snapshot();
    let cards = snapshot.cards();
    // Ten shipped with ARJUN, eighteen imported. Asserted as a floor rather
    // than an exact count so adding a skill does not fail this test — what it
    // must catch is a skill silently disappearing, which is what a rename or a
    // bad hash produces.
    assert!(
        cards.len() >= 28,
        "expected at least 28 skills, found {}: {:?}",
        cards.len(),
        cards.iter().map(|c| c.name.as_str()).collect::<Vec<_>>()
    );

    let quarantined: Vec<String> = cards
        .iter()
        .filter(|card| card.quarantined.is_some())
        .map(|card| {
            let reason = card
                .quarantined
                .as_ref()
                .map(|q| q.explain())
                .unwrap_or_else(|| "unexplained".to_string());
            format!("{}: {reason}", card.name)
        })
        .collect();

    assert!(
        quarantined.is_empty(),
        "{} of {} skills were quarantined:\n  {}",
        quarantined.len(),
        cards.len(),
        quarantined.join("\n  ")
    );
}
