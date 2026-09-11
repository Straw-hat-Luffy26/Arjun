/**
 * Whether a generation that never returns fails, or takes the run with it.
 *
 * `ToolSpec::timeout` existed on the Rust side from the start and reached this
 * side as `timeoutSeconds`, where nothing read it. Every generator was
 * unbounded: a wedged subprocess, a headless render that never exited, or a
 * handler that blocked produced a spinner rather than an error, and the run
 * could not end.
 *
 * The spec asks for the same three checks on every generator - resolves inside
 * the ceiling, a forced hang becomes an error rather than a hang, and the
 * failure is one the surface can show. They are asserted here once, against the
 * single call site every tool passes through, rather than six times against six
 * generators that would each need their own harness.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { GrantLedger, buildTools, type EligibleTool } from "./tools.js";

/** A peer whose `tool.execute` never settles, which is the failure being tested. */
function wedgedPeer() {
  return {
    request: (method: string) =>
      method === "tool.authorize"
        ? Promise.resolve({ outcome: "allow", grant: "g-1" })
        : new Promise(() => {}),
    notify: () => {},
  } as never;
}

function answeringPeer(afterMs: number) {
  return {
    request: (method: string) =>
      method === "tool.authorize"
        ? Promise.resolve({ outcome: "allow", grant: "g-1" })
        : new Promise((resolve) =>
            setTimeout(() => resolve({ text: "done", details: {} }), afterMs),
          ),
    notify: () => {},
  } as never;
}

function eligible(name: string, timeoutSeconds: number): EligibleTool[] {
  return [
    {
      name,
      summary: "",
      readOnly: false,
      approvalClass: "Automatic",
      network: "None",
      maxResponseBytes: 4096,
      timeoutSeconds,
    },
  ];
}

/** Every generator the spec names, so none is bounded by accident. */
const GENERATORS = [
  "artifact.create_pdf",
  "artifact.create_table",
  "artifact.create_chart",
  "artifact.create_diagram",
  "artifact.create_approval_note",
  "artifact.create_briefing_deck",
  "artifact.create_calculation_workbook",
];

describe("a generation that does not come back", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it.each(GENERATORS)("%s fails rather than hanging", async (name) => {
    const ledger = new GrantLedger();
    const [tool] = buildTools(wedgedPeer(), ledger, "run-1", "m", undefined, eligible(name, 10));
    // Thrown rather than asserted: `expect(...).toBeDefined()` does not narrow
    // `Tool | undefined` for the compiler, so every later `tool.execute` was a
    // type error. The throw carries the same diagnostic and narrows.
    if (!tool) throw new Error(`${name} is not in the catalogue`);

    ledger.put("call-1", "g-1");
    const call = tool.execute("call-1", {});
    const settled = call.then(
      () => "resolved",
      (error: Error) => error.message,
    );

    await vi.advanceTimersByTimeAsync(10_000);
    const outcome = await settled;

    expect(outcome).toContain("did not finish within 10s");
    expect(outcome).toContain("stopped");
  });

  it("a call that answers in time is untouched", async () => {
    const ledger = new GrantLedger();
    const [tool] = buildTools(
      answeringPeer(500),
      ledger,
      "run-1",
      "m",
      undefined,
      eligible("artifact.create_chart", 10),
    );
    if (!tool) throw new Error("artifact.create_chart is not in the catalogue");
    ledger.put("call-1", "g-1");

    const call = tool.execute("call-1", {});
    await vi.advanceTimersByTimeAsync(600);
    await expect(call).resolves.toBeDefined();
  });

  /** A catalogue entry with no usable ceiling must not become an unbounded call. */
  it("a missing or nonsensical ceiling falls back rather than never returning", async () => {
    for (const seconds of [0, -1, Number.NaN]) {
      const ledger = new GrantLedger();
      const [tool] = buildTools(
        wedgedPeer(),
        ledger,
        "run-1",
        "m",
        undefined,
        eligible("artifact.create_pdf", seconds),
      );
      if (!tool) throw new Error("artifact.create_pdf is not in the catalogue");
      ledger.put("call-1", "g-1");
      const settled = tool.execute("call-1", {}).then(
        () => "resolved",
        (error: Error) => error.message,
      );

      await vi.advanceTimersByTimeAsync(20_000);
      expect(await settled, `ceiling ${seconds}`).toContain("did not finish within 20s");
    }
  });
});

/**
 * Stop has to reach a call that is already running.
 *
 * `execute` took `(toolCallId, params)` and dropped the third argument
 * agent-core has always passed it: the run's `AbortSignal`. So `run.abort` —
 * the operator's Stop button — could not touch an in-flight tool. The loop sat
 * in `Promise.all` over the launched calls until each one's own ceiling
 * expired, which for a document is two minutes of a turn the person had already
 * cancelled.
 */
describe("stopping a run that is inside a tool call", () => {
  beforeEach(() => vi.useFakeTimers());
  afterEach(() => vi.useRealTimers());

  it("gives up on a call that is still running", async () => {
    const ledger = new GrantLedger();
    const [tool] = buildTools(
      wedgedPeer(),
      ledger,
      "run-1",
      "m",
      undefined,
      eligible("artifact.create_approval_note", 120),
    );
    if (!tool) throw new Error("the tool is not in the catalogue");

    const controller = new AbortController();
    ledger.put("call-1", "g-1");
    const settled = tool.execute("call-1", {}, controller.signal).then(
      () => "resolved",
      (error: Error) => error.message,
    );

    controller.abort();
    const outcome = await settled;

    expect(outcome).toContain("stopped");
    // And it does not claim the effect did not happen. `Promise.race` abandons
    // the loser; it does not reach into Rust and undo a file that was written.
    expect(outcome).toContain("may already have happened");
  });

  it("does not start a call on a run that is already stopped", async () => {
    const ledger = new GrantLedger();
    const [tool] = buildTools(
      wedgedPeer(),
      ledger,
      "run-1",
      "m",
      undefined,
      eligible("artifact.create_approval_note", 120),
    );
    if (!tool) throw new Error("the tool is not in the catalogue");

    const controller = new AbortController();
    controller.abort();
    ledger.put("call-1", "g-1");

    const outcome = await tool.execute("call-1", {}, controller.signal).then(
      () => "resolved",
      (error: Error) => error.message,
    );

    // Refused before the request goes out, so this one *cannot* have happened
    // and says so plainly.
    expect(outcome).toContain("did not run");
  });
});
