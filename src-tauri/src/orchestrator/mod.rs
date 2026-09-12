//! Planning work, calling tools, and stopping when it should.
//!
//! PS 26117 asks the assistant to "actually act like an agent" — plan multi-step
//! work, call local tools, and iterate rather than answering once and stopping.
//! The risk in that sentence is the whole design problem: an agent that can call
//! tools is an agent that can do damage, and a probabilistic system will
//! eventually emit a call nobody intended.
//!
//! So the model here can only ever *request*. It emits a [`tools::ToolCall`] and
//! [`gateway::ToolGateway`] decides — against the user's permissions, the task's
//! workspace, and the sovereignty invariant, none of which the model can reach
//! or influence.
//!
//! - [`tools`]: the catalogue of what may be asked for, and each one's limits.
//! - [`gateway`]: what decides whether a particular request happens.
//! - [`calculation`]: arithmetic done properly, so the numbers do not come from
//!   a model that is usually about right.
//! - [`plan`]: budgets the model cannot widen, and a run that knows to stop.
//! - [`grammar`]: constrains the emitting turn so a malformed call cannot occur.
//! - [`sandbox`]: whether model-written code may run, and what that guarantees.
//! - [`sandbox_exec`]: the container it actually runs in, when it may.
//! - [`executor`]: the loop, one step at a time, pausing when a person is needed.
//! - [`runner`]: the tools themselves, running only what the gateway permitted.

pub mod approvals;
pub mod calculation;
pub mod executor;
pub mod gateway;
pub mod grammar;
pub mod plan;
pub mod progress;
pub mod runner;
pub mod sandbox;
pub mod sandbox_exec;
pub mod tools;

pub use calculation::{evaluate, CalculationRecord};
pub use executor::{Executor, StepOutcome, TaskState, ToolRunner};
pub use gateway::{GatewayVerdict, TaskContext, ToolGateway};
pub use grammar::{build as build_grammar, ToolGrammar};
pub use plan::{Budget, Continuation, PlanRun, StopReason};
pub use runner::LocalToolRunner;
pub use sandbox::{assess, detect_tier, SandboxAssessment, SandboxPolicy, SandboxTier};
pub use tools::{spec_for, ToolCall, ToolName, ToolSpec};
