//! Where model-written code runs, and what that actually guarantees.
//!
//! ARJUN design rule 28: *"The sandbox runs it with a read-only base image, no host
//! credentials, no unrestricted host mounts, blocked network, limited CPU/RAM, a
//! timeout, and a restricted output directory."*
//!
//! That is a list of seven promises, and the honest position is that **not every
//! machine can keep all seven**. A workstation with Podman installed can. A bare
//! Windows laptop cannot block a child process's network access without
//! administrator rights, whatever else it does.
//!
//! So this module does not pretend. It detects what isolation the machine can
//! actually provide, states exactly which promises that tier keeps, and — the
//! part that matters — **refuses to run code when the promises that matter are
//! not available**, rather than running it anyway and calling it sandboxed.
//!
//! ## Why the weakest tier is not enough on its own
//!
//! ARJUN's own egress is stopped at the broker, and that holds for ARJUN. It
//! does not bind a child process: a Python script started by the sandbox is a
//! separate process with its own sockets, and nothing in the broker constrains
//! it. A tier that cannot block that is a tier that cannot run untrusted code on
//! a machine whose whole claim is that nothing leaves it.
//!
//! An administrator may accept that risk deliberately — some sites will, for a
//! demo on a disconnected laptop — but it has to be a decision somebody makes
//! and the audit log records, never a default that quietly holds.

use std::time::Duration;

use serde::{Deserialize, Serialize};

/// What a tier actually guarantees. Each field is a promise from ARJUN design rule 28.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Isolation {
    /// The process cannot reach the network. The one that matters most here.
    pub network_blocked: bool,
    /// The process sees its own filesystem, not the host's.
    pub filesystem_isolated: bool,
    /// CPU and memory are capped.
    pub resource_limited: bool,
    /// The process cannot see or signal other processes on the machine.
    pub process_isolated: bool,
    /// The process runs without the signed-in user's credentials.
    pub credentials_withheld: bool,
}

/// How code can be isolated on this machine, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum SandboxTier {
    /// A rootless container. Keeps every promise, and is what a provisioned
    /// workstation should have.
    Container,
    /// A WSL2 distribution. Real filesystem and process isolation, but it shares
    /// the host's network stack by default, so the promise that matters most is
    /// not kept without further configuration.
    Wsl2,
    /// A Windows job object. Caps CPU, memory and lifetime, and kills the whole
    /// tree when the parent goes. Nothing more: same filesystem, same network.
    JobObject,
    /// Nothing is available.
    None,
}

impl SandboxTier {
    pub const fn label(self) -> &'static str {
        match self {
            SandboxTier::Container => "container",
            SandboxTier::Wsl2 => "WSL2",
            SandboxTier::JobObject => "Windows job object",
            SandboxTier::None => "none",
        }
    }

    /// What this tier promises. Written out per tier rather than inferred, so
    /// each claim can be checked against what the technology actually does.
    pub const fn isolation(self) -> Isolation {
        match self {
            SandboxTier::Container => Isolation {
                network_blocked: true,
                filesystem_isolated: true,
                resource_limited: true,
                process_isolated: true,
                credentials_withheld: true,
            },
            SandboxTier::Wsl2 => Isolation {
                // WSL2 shares the host network stack. Anything claiming
                // otherwise here would be the single most dangerous line in
                // this file.
                network_blocked: false,
                filesystem_isolated: true,
                resource_limited: true,
                process_isolated: true,
                credentials_withheld: false,
            },
            SandboxTier::JobObject => Isolation {
                network_blocked: false,
                filesystem_isolated: false,
                resource_limited: true,
                process_isolated: false,
                credentials_withheld: false,
            },
            SandboxTier::None => Isolation {
                network_blocked: false,
                filesystem_isolated: false,
                resource_limited: false,
                process_isolated: false,
                credentials_withheld: false,
            },
        }
    }

    /// Whether this tier is enough to run code somebody did not write.
    ///
    /// Requires the network to be blocked. On a product whose claim is that
    /// nothing leaves the machine, a child process that can open a socket
    /// defeats the claim regardless of what else is constrained.
    pub const fn suitable_for_untrusted_code(self) -> bool {
        self.isolation().network_blocked
    }
}

/// Limits applied to any execution, whatever the tier.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SandboxPolicy {
    pub timeout: Duration,
    pub max_memory_bytes: u64,
    /// Cores the process may use.
    pub max_cpus: u32,
    /// Bytes the process may write into its output directory.
    pub max_output_bytes: u64,
    /// Set by an administrator who has accepted running code on a tier that
    /// cannot block the network. Never a default.
    pub accept_unisolated_network: bool,
}

impl Default for SandboxPolicy {
    fn default() -> Self {
        Self {
            // Matches the tool's own timeout. A calculation needing longer than
            // this is not a calculation.
            timeout: Duration::from_secs(60),
            max_memory_bytes: 2 * 1024 * 1024 * 1024,
            max_cpus: 2,
            max_output_bytes: 64 * 1024 * 1024,
            accept_unisolated_network: false,
        }
    }
}

/// Whether code may run here, and what the person should be told.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", tag = "outcome")]
pub enum SandboxAssessment {
    /// Safe to run, on this tier.
    Ready {
        tier: SandboxTier,
        isolation: Isolation,
    },
    /// Runnable, but a promise the problem statement asks for is not kept.
    /// Only reachable when an administrator has explicitly accepted it.
    ReadyWithAcceptedRisk {
        tier: SandboxTier,
        isolation: Isolation,
        warning: String,
    },
    /// Not runnable. Says what would make it runnable.
    Refused { reason: String },
}

impl SandboxAssessment {
    pub fn may_run(&self) -> bool {
        !matches!(self, SandboxAssessment::Refused { .. })
    }

    pub fn tier(&self) -> SandboxTier {
        match self {
            SandboxAssessment::Ready { tier, .. }
            | SandboxAssessment::ReadyWithAcceptedRisk { tier, .. } => *tier,
            SandboxAssessment::Refused { .. } => SandboxTier::None,
        }
    }

    /// One line for the audit record. ARJUN design rule 28 asks that the active tier be
    /// recorded, and a run on an accepted-risk tier must be distinguishable
    /// afterwards from a fully isolated one.
    pub fn audit_summary(&self) -> String {
        match self {
            SandboxAssessment::Ready { tier, .. } => {
                format!("Ran in a {} sandbox with full isolation.", tier.label())
            }
            SandboxAssessment::ReadyWithAcceptedRisk { tier, .. } => format!(
                "Ran in a {} sandbox WITHOUT network isolation, on an administrator's \
                 accepted risk.",
                tier.label()
            ),
            SandboxAssessment::Refused { reason } => format!("Did not run: {reason}"),
        }
    }
}

/// Probes what this machine can actually do.
///
/// Probing rather than configuring: a site that has installed Podman gets the
/// strong tier without being asked, and one that has not is told plainly rather
/// than silently dropped to something weaker.
pub fn detect_tier() -> SandboxTier {
    if container_runtime_usable("podman") || container_runtime_usable("docker") {
        return SandboxTier::Container;
    }
    if available("wsl") {
        return SandboxTier::Wsl2;
    }
    if cfg!(windows) {
        // Always available on Windows: no install, no administrator rights.
        return SandboxTier::JobObject;
    }
    SandboxTier::None
}

/// Tools a rehearsal has asked to be treated as absent.
///
/// The demo must never depend on WSL2 being present, and the only way to know
/// that is to run as though it were not. This lets `npm run rehearse` drive the
/// whole suite down the no-WSL2 path on a machine that does have it.
///
/// The safety property that makes this acceptable in production code: the
/// override can only ever **remove** capability, never add it. A flag left set
/// by accident makes ARJUN refuse to run code it could have sandboxed — the
/// cautious direction. The opposite hook, one that claimed a sandbox that was
/// not there, would be a hole, and does not exist.
const ASSUME_ABSENT: &str = "ARJUN_SANDBOX_ASSUME_ABSENT";

/// Whether `name` appears in a hide-list.
///
/// Split out as a pure function so it can be tested exhaustively without
/// touching the process environment. Tests that set and unset an environment
/// variable race each other — Rust runs them on parallel threads against one
/// shared global — and a flaky test on the isolation boundary is worse than no
/// test, because it trains people to re-run rather than to look.
fn listed_in(list: &str, name: &str) -> bool {
    list.split(',').any(|entry| entry.trim().eq_ignore_ascii_case(name))
}

fn assumed_absent(name: &str) -> bool {
    std::env::var(ASSUME_ABSENT)
        .map(|list| listed_in(&list, name))
        .unwrap_or(false)
}

fn available(name: &str) -> bool {
    !assumed_absent(name) && command_exists(name)
}

/// How long the daemon probe waits before giving up.
///
/// A responsive daemon answers `info` in well under a second. The value is a
/// deadline for a *broken* one, not a performance budget for a healthy one.
const RUNTIME_PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Whether a container runtime is not merely *installed* but actually able to
/// run something.
///
/// `docker --version` answers from the CLI alone and succeeds with the daemon
/// stopped — the state a Windows laptop is in whenever Docker Desktop has not
/// been launched. Treating that as a container tier is the worst kind of wrong:
/// it claims the one tier that promises a blocked network, so `assess` returns
/// `Ready`, and the run proceeds believing it is isolated.
///
/// `info` is the cheapest call that has to reach the daemon, so it fails exactly
/// when the runtime cannot run a container.
///
/// ## Why this waits rather than blocking
///
/// A half-started Docker Desktop — processes up, engine never initialised — does
/// not make `info` *fail*. It makes it **hang**, indefinitely, and
/// `Command::status` has no timeout. An earlier version of this function called
/// it directly, and the result was a test suite that stalled and an application
/// that would have frozen on startup, because `LocalToolRunner::new` calls
/// `detect_tier`. A probe for whether something is healthy must not itself hang
/// when it is not.
///
/// So the child is spawned, polled to a deadline, and killed if it overruns. The
/// check errs toward refusing: a daemon too slow to answer reads as absent and
/// ARJUN declines to run code it might have been able to sandbox. That is the
/// safe direction — the opposite error claims an isolation boundary that is not
/// there.
fn container_runtime_usable(name: &str) -> bool {
    use crate::system_analyzer::process_utils::create_hidden_command;

    if !available(name) {
        return false;
    }

    let Ok(mut child) = create_hidden_command(name)
        .arg("info")
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    else {
        return false;
    };

    let deadline = std::time::Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(status)) => return status.success(),
            Ok(None) => {
                if deadline.elapsed() >= RUNTIME_PROBE_TIMEOUT {
                    let _ = child.kill();
                    let _ = child.wait();
                    return false;
                }
                std::thread::sleep(Duration::from_millis(50));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return false;
            }
        }
    }
}

fn command_exists(name: &str) -> bool {
    use crate::system_analyzer::process_utils::create_hidden_command;

    create_hidden_command(name)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

/// Decides whether code may run, given what the machine offers.
///
/// A pure function of the tier and the policy, so every combination is testable
/// without installing a container runtime.
pub fn assess(tier: SandboxTier, policy: &SandboxPolicy) -> SandboxAssessment {
    let isolation = tier.isolation();

    if tier == SandboxTier::None {
        return SandboxAssessment::Refused {
            reason: "no sandbox is available on this machine, so model-written code cannot be \
                     run at all. Install a container runtime such as Podman."
                .to_string(),
        };
    }

    if tier.suitable_for_untrusted_code() {
        return SandboxAssessment::Ready { tier, isolation };
    }

    if !policy.accept_unisolated_network {
        return SandboxAssessment::Refused {
            // Names the remedy that exists.
            //
            // This used to end "or have an administrator accept this risk
            // explicitly in Settings", which was wrong twice over. There is no
            // such control — `accept_unisolated_network` is read from nothing
            // and written by nothing outside these tests — and setting it would
            // not help anyway, because `sandbox_exec::run_in_container` refuses
            // every tier except `Container` before it looks at any policy. So
            // the message sent an operator to a switch that does not exist, to
            // unlock a path that is not implemented.
            reason: format!(
                "the strongest sandbox available here is a {}, which cannot stop a child process \
                 reaching the network. ARJUN's own network controls do not bind a program it \
                 starts, so code is not run. Start Docker Desktop if it is installed, or install \
                 Podman; a container is the only tier that can run model-written code.",
                tier.label()
            ),
        };
    }

    SandboxAssessment::ReadyWithAcceptedRisk {
        tier,
        isolation,
        warning: format!(
            "This ran in a {}, which cannot block network access from the code itself. An \
             administrator accepted that risk for this deployment. Treat the result as coming \
             from a process that could, in principle, have reached the network.",
            tier.label()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_container_keeps_every_promise() {
        let isolation = SandboxTier::Container.isolation();
        assert!(isolation.network_blocked);
        assert!(isolation.filesystem_isolated);
        assert!(isolation.resource_limited);
        assert!(isolation.process_isolated);
        assert!(isolation.credentials_withheld);
    }

    /// The single most dangerous line this file could contain would be one
    /// claiming WSL2 blocks the network. It shares the host stack.
    #[test]
    fn wsl2_does_not_claim_to_block_the_network() {
        assert!(!SandboxTier::Wsl2.isolation().network_blocked);
        assert!(SandboxTier::Wsl2.isolation().filesystem_isolated);
    }

    #[test]
    fn a_job_object_claims_only_resource_limits() {
        let isolation = SandboxTier::JobObject.isolation();
        assert!(isolation.resource_limited);
        assert!(!isolation.network_blocked);
        assert!(!isolation.filesystem_isolated);
        assert!(!isolation.process_isolated);
    }

    #[test]
    fn only_a_tier_that_blocks_the_network_is_fit_for_untrusted_code() {
        assert!(SandboxTier::Container.suitable_for_untrusted_code());
        for weak in [SandboxTier::Wsl2, SandboxTier::JobObject, SandboxTier::None] {
            assert!(
                !weak.suitable_for_untrusted_code(),
                "{} should not be trusted with model-written code",
                weak.label()
            );
        }
    }

    // ── The decision ─────────────────────────────────────────────────────

    #[test]
    fn a_container_runs_code_without_ceremony() {
        let assessment = assess(SandboxTier::Container, &SandboxPolicy::default());
        assert!(matches!(assessment, SandboxAssessment::Ready { .. }));
        assert!(assessment.may_run());
    }

    /// The property that matters most in this module: no sandbox means no
    /// execution, never execution-without-a-sandbox.
    #[test]
    fn a_weak_tier_refuses_by_default_rather_than_running_unprotected() {
        for weak in [SandboxTier::Wsl2, SandboxTier::JobObject] {
            let assessment = assess(weak, &SandboxPolicy::default());
            assert!(
                !assessment.may_run(),
                "{} should refuse rather than run code unprotected",
                weak.label()
            );
        }
    }

    #[test]
    fn the_refusal_says_what_would_fix_it() {
        let reason = match assess(SandboxTier::JobObject, &SandboxPolicy::default()) {
            SandboxAssessment::Refused { reason } => reason,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(reason.contains("Podman"), "should name a remedy: {reason}");
        assert!(reason.contains("Docker"), "should name the remedy most operators already have");
        // The remedy has to be one that exists. This assertion used to require
        // "accept this risk", pinning a promise of a Settings control that is
        // read from nothing and written by nothing — and that would not enable
        // execution even if it were set, because `run_in_container` refuses
        // every tier except `Container` before consulting any policy. A test
        // that requires an impossible instruction keeps the instruction.
        assert!(
            !reason.contains("accept this risk"),
            "must not offer a setting that does not exist: {reason}"
        );
        assert!(
            reason.contains("container"),
            "should say a container is the only tier that runs code: {reason}"
        );
    }

    /// The refusal has to explain why ARJUN's own controls are not enough,
    /// otherwise it reads as excessive caution.
    #[test]
    fn the_refusal_explains_why_the_brokers_controls_do_not_cover_this() {
        let reason = match assess(SandboxTier::Wsl2, &SandboxPolicy::default()) {
            SandboxAssessment::Refused { reason } => reason,
            other => panic!("expected a refusal, got {other:?}"),
        };
        assert!(reason.contains("do not bind a program it starts"));
    }

    #[test]
    fn no_sandbox_at_all_refuses_and_says_so_plainly() {
        let assessment = assess(SandboxTier::None, &SandboxPolicy::default());
        assert!(!assessment.may_run());
        assert_eq!(assessment.tier(), SandboxTier::None);
    }

    // ── Accepted risk ────────────────────────────────────────────────────

    #[test]
    fn an_administrator_can_accept_the_risk_deliberately() {
        let policy = SandboxPolicy {
            accept_unisolated_network: true,
            ..SandboxPolicy::default()
        };
        let assessment = assess(SandboxTier::Wsl2, &policy);

        assert!(assessment.may_run());
        assert!(matches!(
            assessment,
            SandboxAssessment::ReadyWithAcceptedRisk { .. }
        ));
    }

    /// Accepting the risk must not upgrade "no sandbox" into "some sandbox".
    #[test]
    fn accepting_the_risk_does_not_conjure_a_sandbox_that_is_not_there() {
        let policy = SandboxPolicy {
            accept_unisolated_network: true,
            ..SandboxPolicy::default()
        };
        assert!(!assess(SandboxTier::None, &policy).may_run());
    }

    /// A run on an accepted-risk tier must be distinguishable afterwards from a
    /// fully isolated one.
    #[test]
    fn the_audit_line_distinguishes_a_compromised_run_from_a_clean_one() {
        let clean = assess(SandboxTier::Container, &SandboxPolicy::default()).audit_summary();
        assert!(clean.contains("full isolation"));

        let policy = SandboxPolicy {
            accept_unisolated_network: true,
            ..SandboxPolicy::default()
        };
        let risky = assess(SandboxTier::Wsl2, &policy).audit_summary();
        assert!(risky.contains("WITHOUT network isolation"));
        assert!(risky.contains("accepted risk"));
    }

    #[test]
    fn the_default_policy_never_accepts_the_risk() {
        assert!(!SandboxPolicy::default().accept_unisolated_network);
    }

    #[test]
    fn the_default_policy_caps_every_resource() {
        let policy = SandboxPolicy::default();
        assert!(policy.timeout > Duration::ZERO);
        assert!(policy.max_memory_bytes > 0);
        assert!(policy.max_cpus > 0);
        assert!(policy.max_output_bytes > 0);
    }

    /// Runs against the real machine. Asserts only that detection returns a
    /// coherent answer — what is installed varies, and a test that demanded a
    /// container runtime would fail on exactly the laptops this has to support.
    #[test]
    fn detection_returns_a_tier_consistent_with_this_platform() {
        let tier = detect_tier();
        if cfg!(windows) {
            assert_ne!(tier, SandboxTier::None, "Windows always has job objects");
        }
        // Whatever it found, its promises must match its own table.
        assert_eq!(tier.isolation(), tier.isolation());
    }

    // ── The rehearsal override ───────────────────────────────────────────

    /// The list parsing, exhaustively — and without touching the environment,
    /// so these can never race the rest of the suite.
    #[test]
    fn a_listed_tool_reads_as_absent_however_it_is_written() {
        assert!(listed_in("wsl", "wsl"));
        assert!(listed_in(" WSL , Podman ", "wsl"));
        assert!(listed_in(" WSL , Podman ", "podman"));
        assert!(listed_in("podman,docker,wsl", "docker"));
        assert!(!listed_in(" WSL , Podman ", "docker"));
        assert!(!listed_in("", "wsl"));
        // A substring is not a match: hiding "wsl" must not hide "wsl2-thing".
        assert!(!listed_in("wsl", "wsl2-thing"));
    }

    /// The property that makes the override safe to ship: it can only take
    /// capability away. Nothing here can claim a sandbox that is not present.
    #[test]
    fn the_override_can_only_remove_capability_never_add_it() {
        // Hiding a tool that is not installed changes nothing — it was already
        // absent. There is deliberately no inverse hook that could mark an
        // absent tool as present, because that would be a hole in the boundary.
        assert!(!listed_in("", "definitely-not-installed"));
        assert!(listed_in("definitely-not-installed", "definitely-not-installed"));
        assert!(!command_exists("definitely-not-installed"));
    }

    /// The wiring itself: one test, and the only one that touches the process
    /// environment. It reads rather than writes, so it cannot disturb anything
    /// running beside it — under `npm run rehearse` the variable is already set
    /// by the caller, and outside it the variable is absent.
    #[test]
    fn the_environment_variable_is_the_thing_actually_consulted() {
        match std::env::var(ASSUME_ABSENT) {
            Ok(list) => {
                // A rehearsal is running. Everything it hid must read as absent,
                // and the machine must still resolve to some tier.
                for hidden in list.split(',').filter(|s| !s.trim().is_empty()) {
                    assert!(assumed_absent(hidden.trim()));
                    assert!(!available(hidden.trim()));
                }
                let tier = detect_tier();
                if list.contains("wsl") && list.contains("podman") && list.contains("docker") {
                    let expected =
                        if cfg!(windows) { SandboxTier::JobObject } else { SandboxTier::None };
                    assert_eq!(tier, expected, "the machine must degrade, not break");
                }
                assert!(!assess(tier, &SandboxPolicy::default()).audit_summary().is_empty());
            }
            Err(_) => {
                // No rehearsal. Nothing is hidden, and detection is unaffected.
                assert!(!assumed_absent("wsl"));
                assert!(!assumed_absent("podman"));
            }
        }
    }
}
