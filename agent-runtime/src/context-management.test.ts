/**
 * The properties that make a long run survivable.
 *
 * Each test here names a way a run breaks in front of somebody, not a function
 * signature. The distinction matters because every one of these failures is
 * silent: the loop keeps running, the model keeps answering, and what is lost
 * is the approval that was granted, the effect that already happened, or the
 * evidence a citation resolves against.
 */

import { describe, expect, it, vi } from "vitest";
import { convertToLlm, estimateTokens, type AgentMessage } from "@openclaw/agent-core";
import type { Model } from "@openclaw/ai";
import {
  RunCompactor,
  alignCutToPairs,
  isEvidenceMessage,
  pairingIsIntact,
  pruneStaleToolResults,
} from "./compaction.js";
import { ContextLedger } from "./context-ledger.js";
import { NOTE_LIMITS, WorkingNotes } from "./working-notes.js";

function model(contextWindow: number): Model {
  return {
    id: "qwen2.5-coder-7b",
    name: "Qwen2.5 Coder 7B",
    api: "openai-completions",
    provider: "llama-cpp",
    baseUrl: "http://127.0.0.1:8080/v1",
    reasoning: false,
    input: ["text"],
    cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    contextWindow,
    maxTokens: 2048,
  } as Model;
}

function user(text: string): AgentMessage {
  return { role: "user", content: [{ type: "text", text }], timestamp: 1 } as AgentMessage;
}

function assistant(text: string): AgentMessage {
  return {
    role: "assistant",
    content: [{ type: "text", text }],
    api: "openai-completions",
    provider: "llama-cpp",
    model: "qwen2.5-coder-7b",
    stopReason: "stop",
    timestamp: 1,
  } as unknown as AgentMessage;
}

function toolCall(id: string, name = "search_documents"): AgentMessage {
  return {
    role: "assistant",
    content: [
      { type: "text", text: "searching" },
      { type: "toolCall", id, name, arguments: { query: "seal wear" } },
    ],
    api: "openai-completions",
    provider: "llama-cpp",
    model: "qwen2.5-coder-7b",
    stopReason: "toolUse",
    timestamp: 1,
  } as unknown as AgentMessage;
}

function toolResult(id: string, text: string, name = "search_documents"): AgentMessage {
  return {
    role: "toolResult",
    toolCallId: id,
    toolName: name,
    content: [{ type: "text", text }],
    isError: false,
    timestamp: 1,
  } as unknown as AgentMessage;
}

/** A summariser that answers without a model server. */
function summariser(text = "Earlier: the operator asked about pump seals.") {
  return {
    completeSimple: vi.fn(async () => ({
      role: "assistant",
      content: [{ type: "text", text }],
      api: "openai-completions",
      provider: "llama-cpp",
      model: "qwen2.5-coder-7b",
      stopReason: "stop",
      timestamp: 1,
      usage: {
        input: 0,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens: 0,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    })),
  } as never;
}

/** A long run of search-heavy turns, which is the shape that overflows first. */
function documentRun(turns: number, chars: number): AgentMessage[] {
  const messages: AgentMessage[] = [];
  for (let i = 0; i < turns; i++) {
    messages.push(user(`question ${i} ${"x".repeat(chars)}`));
    messages.push(toolCall(`call_${i}`));
    messages.push(
      toolResult(`call_${i}`, `[E${i + 1}] Maintenance SOP, page 4\n${"y".repeat(chars)}`),
    );
    messages.push(assistant(`answer ${i}`));
  }
  return messages;
}

describe("working notes stay bounded", () => {
  it("does not grow into a second transcript however much is written to it", () => {
    const notes = new WorkingNotes();
    // Ten times the cap on every list, which is what an eager model produces.
    for (let i = 0; i < NOTE_LIMITS.decisions * 10; i++) {
      notes.decided(`decision ${i}`, `reason ${i}`);
    }
    for (let i = 0; i < NOTE_LIMITS.evidenceIds * 10; i++) notes.sawEvidence(`E${i}`);
    for (let i = 0; i < NOTE_LIMITS.openQuestions * 10; i++) notes.asked(`question ${i}`);
    for (let i = 0; i < NOTE_LIMITS.completed * 10; i++) {
      notes.didEffect("write_scoped_file", `file-${i}.txt`);
    }

    const state = notes.state;
    expect(state.decisions).toHaveLength(NOTE_LIMITS.decisions);
    expect(state.evidenceIds).toHaveLength(NOTE_LIMITS.evidenceIds);
    expect(state.openQuestions).toHaveLength(NOTE_LIMITS.openQuestions);
    expect(state.completed).toHaveLength(NOTE_LIMITS.completed);
    // The newest survive: an old decision superseded by a newer one is the one
    // to lose, and the reverse would pin the notes to the run's first minute.
    expect(state.decisions.at(-1)?.what).toBe(`decision ${NOTE_LIMITS.decisions * 10 - 1}`);
  });

  it("says when it dropped something rather than forgetting quietly", () => {
    const notes = new WorkingNotes();
    for (let i = 0; i < NOTE_LIMITS.decisions + 5; i++) notes.decided(`d${i}`, "because");

    expect(notes.state.dropped.decisions).toBe(5);
    expect(notes.render()).toContain("Older entries were dropped");
  });

  it("caps a single line so one long note cannot fill the window on its own", () => {
    const notes = new WorkingNotes();
    notes.setGoal("g".repeat(10_000));

    expect(notes.state.goal.length).toBeLessThanOrEqual(NOTE_LIMITS.lineChars);
  });

  it("counts one marker once however often it is recorded", () => {
    const notes = new WorkingNotes();
    for (let i = 0; i < 50; i++) notes.sawEvidence("E3");

    expect(notes.state.evidenceIds).toEqual(["E3"]);
  });

  it("re-applies the caps when loaded, rather than trusting the file", () => {
    // A record written before a cap was tightened must not reintroduce the
    // unbounded state it was written with.
    const oversized = {
      decisions: Array.from({ length: 100 }, (_, i) => ({
        what: `d${i}`,
        because: "b",
        at: "2026-08-28T09:15:00.000Z",
      })),
    };

    expect(WorkingNotes.from(oversized).state.decisions).toHaveLength(NOTE_LIMITS.decisions);
  });
});

describe("a recovered run continues from its notes", () => {
  it("knows which side effects already happened and does not repeat them", () => {
    // The failure: the process died after `create_docx` wrote the approval note
    // but before the run recorded an answer. A resumption that only reads the
    // prompt writes the document a second time.
    const before = new WorkingNotes();
    before.setGoal("Draft the approval note for the seal replacement.");
    before.atStage(3, "produce the document");
    before.sawEvidence("E1");
    before.didEffect("create_docx", "approval-note.docx", "2026-08-28T09:15:00.000Z");
    before.setNextAction("check the produced document opens and cites E1");

    const resumed = WorkingNotes.from(before.state);

    expect(resumed.hasDone("create_docx", "approval-note.docx")).toBe(true);
    // And the model is told, in the text it actually reads.
    const rendered = resumed.render();
    expect(rendered).toContain("Already done — do not repeat");
    expect(rendered).toContain("approval-note.docx");
    expect(rendered).toContain("Next: check the produced document");
  });

  it("carries the goal and stage across, so the resumption is not a fresh start", () => {
    const before = new WorkingNotes();
    before.setGoal("Draft the approval note.");
    before.atStage(2, "search the manuals");

    const resumed = WorkingNotes.from(before.state);

    expect(resumed.state.goal).toBe("Draft the approval note.");
    expect(resumed.state.stage).toEqual({ ordinal: 2, intent: "search the manuals" });
  });

  it("does not double-count an effect replayed by the recovery path", () => {
    const notes = new WorkingNotes();
    notes.didEffect("create_docx", "approval-note.docx");
    notes.didEffect("create_docx", "approval-note.docx");

    expect(notes.state.completed).toHaveLength(1);
  });
});

describe("tool calls and their results are never separated", () => {
  it("recognises an orphaned tool result as the malformed transcript it is", () => {
    const orphaned = [toolResult("call_1", "result"), assistant("done")];
    expect(pairingIsIntact(orphaned)).toBe(false);

    const paired = [toolCall("call_1"), toolResult("call_1", "result")];
    expect(pairingIsIntact(paired)).toBe(true);
  });

  it("moves a cut earlier rather than keeping a result without its call", () => {
    const messages = [
      user("ask"),
      toolCall("call_1"),
      toolResult("call_1", "result"),
      assistant("done"),
    ];

    // A cut landing on the tool result would orphan it.
    const aligned = alignCutToPairs(messages, 2);

    expect(aligned).toBeLessThanOrEqual(1);
    expect(pairingIsIntact(messages.slice(aligned))).toBe(true);
  });

  it("never moves a cut later, which would silently drop history", () => {
    const messages = [user("a"), assistant("b"), user("c")];
    for (let cut = 0; cut <= messages.length; cut++) {
      expect(alignCutToPairs(messages, cut)).toBeLessThanOrEqual(cut);
    }
  });

  it("keeps the pairing intact through a real compaction", async () => {
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
    });

    const projected = await compactor.transform(documentRun(30, 600));

    expect(compactor.compactions).toBeGreaterThan(0);
    expect(pairingIsIntact(projected)).toBe(true);
  });
});

describe("approval and policy state survives compaction", () => {
  it("carries a refusal across, so the run does not retry what was refused", async () => {
    // The failure that matters. A granted approval lost to a summary is
    // re-requested; a *refusal* lost to a summary is retried, and the second
    // attempt is the one nobody authorised.
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser("Earlier: the operator asked about pump seals."),
      apiKey: "local",
      preserved: () => ({
        activePlan: "Step 3 of 5: produce the approval note.",
        policyDecisions: [
          "execute_code was refused: this run holds no ExecuteCode permission.",
          "create_docx was approved by kiran at 2026-08-28T09:15:00Z.",
        ],
        evidenceRefs: ["E1", "E2"],
        unresolvedIssues: ["The 2019 revision of the SOP was not found."],
        recentFiles: ["approval-note.docx"],
      }),
    });

    const projected = await compactor.transform(documentRun(30, 600));
    const sent = JSON.stringify(convertToLlm(projected));

    // Converted, not merely present: a message the converter drops is a message
    // the model never sees, which is indistinguishable from never carrying it.
    expect(sent).toContain("execute_code was refused");
    expect(sent).toContain("create_docx was approved");
    expect(sent).toContain("Step 3 of 5");
    expect(sent).toContain("2019 revision");
    expect(sent).toContain("approval-note.docx");
  });

  it("carries evidence markers rather than the passages they refer to", async () => {
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
      preserved: () => ({ evidenceRefs: ["E1", "E2", "E3"] }),
    });

    const projected = await compactor.transform(documentRun(30, 600));
    const carried = JSON.stringify(convertToLlm(projected.slice(0, 2)));

    expect(carried).toContain("E1, E2, E3");
    // The passage text itself is not re-pasted into the preserved block.
    expect(carried).not.toContain("y".repeat(600));
  });

  it("keeps the notes in front of the model before any compaction happens", async () => {
    const notes = new WorkingNotes();
    notes.setGoal("Draft the approval note.");
    const compactor = new RunCompactor({
      model: model(32_768),
      runtime: summariser(),
      apiKey: "local",
      notes,
    });

    const projected = await compactor.transform([user("hello"), assistant("hi")]);

    expect(compactor.compactions).toBe(0);
    expect(JSON.stringify(convertToLlm(projected))).toContain("Draft the approval note.");
  });
});

describe("a second compaction refines the summary rather than duplicating it", () => {
  it("holds exactly one summary message however many times it compacts", async () => {
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
    });

    await compactor.transform(documentRun(30, 600));
    const projected = await compactor.transform(documentRun(70, 600));

    expect(compactor.compactions).toBe(2);
    const summaries = projected.filter((message) =>
      JSON.stringify(message).includes("compactionSummary"),
    );
    // Two summary messages would mean the second compaction appended rather
    // than refined, and the older half of the run would be described twice.
    expect(summaries.length).toBeLessThanOrEqual(1);
  });

  it("reports that it refined an existing summary rather than starting one", async () => {
    const events: { ordinal: number; refinedExistingSummary: boolean }[] = [];
    const compactor = new RunCompactor({
      model: model(8_192),
      runtime: summariser(),
      apiKey: "local",
      onCompacted: (event) => { events.push(event); },
    });

    await compactor.transform(documentRun(30, 600));
    await compactor.transform(documentRun(70, 600));

    expect(events).toHaveLength(2);
    expect(events[0]).toMatchObject({ ordinal: 1, refinedExistingSummary: false });
    expect(events[1]).toMatchObject({ ordinal: 2, refinedExistingSummary: true });
  });
});

describe("stale raw tool results are cleared once their evidence is durable", () => {
  it("replaces the passage text with the marker it can be looked up by", () => {
    const messages = [
      toolCall("call_1"),
      toolResult("call_1", `[E1] Maintenance SOP, page 4\n${"y".repeat(2_000)}`),
      ...Array.from({ length: 8 }, (_, i) => assistant(`turn ${i}`)),
    ];

    const { messages: pruned, cleared } = pruneStaleToolResults(messages, ["E1"]);

    expect(cleared).toBe(1);
    const text = JSON.stringify(pruned);
    expect(text).not.toContain("y".repeat(2_000));
    expect(text).toContain("[E1]");
    expect(text).toContain("load_more_evidence");
    // The message is rewritten, never removed — removing it would orphan the
    // call that produced it.
    expect(pruned).toHaveLength(messages.length);
    expect(pairingIsIntact(pruned)).toBe(true);
  });

  it("leaves a result alone when a marker it carries is not yet durable", () => {
    const messages = [
      toolCall("call_1"),
      toolResult("call_1", "[E1] one\n[E2] two"),
      ...Array.from({ length: 8 }, (_, i) => assistant(`turn ${i}`)),
    ];

    // Only E1 is recorded. Clearing would drop E2 beyond recovery.
    const { cleared } = pruneStaleToolResults(messages, ["E1"]);

    expect(cleared).toBe(0);
  });

  it("leaves the most recent results alone, which the model is still using", () => {
    const messages = [toolCall("call_1"), toolResult("call_1", "[E1] just read this")];

    const { cleared } = pruneStaleToolResults(messages, ["E1"]);

    expect(cleared).toBe(0);
  });

  it("clears nothing when no evidence is durable yet", () => {
    const messages = documentRun(10, 100);
    expect(pruneStaleToolResults(messages, []).cleared).toBe(0);
  });
});

describe("the context ledger says where the window went", () => {
  it("accounts for every section and reports the total the next turn must fit", () => {
    const ledger = new ContextLedger(8_192);
    ledger.setText("system", "s".repeat(400));
    ledger.setText("skill", "k".repeat(400));
    ledger.setText("toolSchema", "t".repeat(4_000));
    ledger.setText("notes", "n".repeat(200));
    ledger.set("reserve", 1_600);

    const snapshot = ledger.snapshot();

    expect(Object.keys(snapshot.sections).sort()).toEqual(
      [
        "compaction",
        "evidence",
        "notes",
        "reserve",
        "skill",
        "system",
        "toolSchema",
        "transcript",
      ].sort(),
    );
    expect(snapshot.committed).toBe(snapshot.occupied + snapshot.sections.reserve);
    expect(snapshot.headroom).toBe(8_192 - snapshot.committed);
    // The diagnosis an operator actually reads: the tool schemas are the
    // largest thing here, and that is a catalogue problem, not a run problem.
    expect(ledger.largest(1)[0]?.section).toBe("toolSchema");
  });

  it("declines to claim headroom on a window it was never told", () => {
    const ledger = new ContextLedger(0);
    ledger.setText("system", "s".repeat(1_000));

    expect(ledger.snapshot().headroom).toBe(0);
    // Not "it fits": an unknown window is not evidence of room.
    expect(ledger.fits()).toBe(false);
  });
});

describe("the ledger tells retrieved evidence apart from conversation", () => {
  /**
   * Short questions, long passages — the shape a document run actually has,
   * and the one where the two lines give opposite advice.
   */
  function passageHeavyRun(turns: number): AgentMessage[] {
    const messages: AgentMessage[] = [];
    for (let i = 0; i < turns; i++) {
      messages.push(user(`question ${i}`));
      messages.push(toolCall(`call_${i}`));
      messages.push(
        toolResult(`call_${i}`, `[E${i + 1}] Maintenance SOP, page 4
${"y".repeat(600)}`),
      );
      messages.push(assistant(`answer ${i}`));
    }
    return messages;
  }

  it("recognises a passage by the marker the model was told to cite", () => {
    expect(
      isEvidenceMessage(toolResult("c1", "[E1] Maintenance SOP, page 4 — the passage")),
    ).toBe(true);
    // A tool that returned no evidence is conversation, however long it is.
    expect(isEvidenceMessage(toolResult("c1", "Wrote approval-note.docx.", "create_docx"))).toBe(
      false,
    );
    // Nor is the model's own prose about evidence, whatever it quotes.
    expect(isEvidenceMessage(assistant("As [E1] says, the seal is worn."))).toBe(false);
  });

  it("still counts a reference stub as evidence once the passage text is cleared", () => {
    // What pruning leaves behind. It is small, but it is retrieval rather than
    // conversation, and booking it as conversation would make the evidence line
    // fall to zero on exactly the runs that retrieved the most.
    const { messages } = pruneStaleToolResults(passageHeavyRun(4), ["E1", "E2", "E3", "E4"]);

    expect(messages.filter(isEvidenceMessage).length).toBeGreaterThan(0);
  });

  it("recognises a stub for a result that carried several passages", () => {
    // A search returns six passages at a time, so the multi-marker stub is the
    // common case, not the exotic one. A pattern that only matched `[E1]`
    // would report the evidence line as near-empty on precisely the runs that
    // retrieved the most — the opposite of what an operator needs to read.
    const twoMarkers = toolResult("call_1", "[E1] first passage. [E2] second passage.");
    const { messages, cleared } = pruneStaleToolResults(
      [...Array.from({ length: 8 }, (_, i) => user(`filler ${i}`)), twoMarkers].reverse(),
      ["E1", "E2"],
    );

    expect(cleared).toBe(1);
    const stub = messages.find((message) => (message as { role?: string }).role === "toolResult");
    expect(stub).toBeDefined();
    expect(isEvidenceMessage(stub as AgentMessage)).toBe(true);
  });

  it("names the passages, not the conversation, when the passages filled the window", async () => {
    const window = 8_192;
    const ledger = new ContextLedger(window);
    const compactor = new RunCompactor({
      model: model(window),
      runtime: summariser(),
      apiKey: "local",
      ledger,
    });

    await compactor.transform(passageHeavyRun(6));

    // The whole reason the line exists. An operator told "transcript: 6,000"
    // reaches for the only lever that phrasing offers — compact sooner — which
    // shortens the conversation and degrades the run. Told "evidence: 5,200"
    // they ask for three pages instead of thirty, and lose nothing: the rest is
    // still retrievable by marker.
    expect(ledger.get("evidence")).toBeGreaterThan(ledger.get("transcript"));
    expect(ledger.largest(1)[0]?.section).toBe("evidence");
  });

  /**
   * An assistant turn as a real provider returns one: with the usage it
   * reported for the whole request.
   *
   * The plain `assistant` helper above carries none, which is why the ledger
   * over-reporting went unnoticed — every message a live run receives has this.
   */
  function assistantWithUsage(text: string, totalTokens: number): AgentMessage {
    return {
      ...(assistant(text) as object),
      usage: {
        input: totalTokens,
        output: 0,
        cacheRead: 0,
        cacheWrite: 0,
        totalTokens,
        cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
      },
    } as unknown as AgentMessage;
  }

  it("does not report the whole conversation as the size of one section", async () => {
    // The bug this replaced. The harness estimator answers with the provider's
    // reported usage for the entire request the moment the list it is handed
    // contains an assistant message carrying one — right for a whole context,
    // wrong for a part of it. Measuring a section that way booked the whole run
    // under that section, and then added the system prompt, the tool schemas
    // and the notes on top: an occupied total larger than the context, and an
    // operator told to shed load that was never there.
    const window = 8_192;
    const ledger = new ContextLedger(window);
    const compactor = new RunCompactor({
      model: model(window),
      runtime: summariser(),
      apiKey: "local",
      ledger,
    });

    const reported = 6_000;
    const projected = await compactor.transform([
      user("what does the SOP say about seal wear?"),
      toolCall("call_1"),
      toolResult("call_1", `[E1] Maintenance SOP, page 4 — ${"y".repeat(400)}`),
      assistantWithUsage("The seal is worn past tolerance.", reported),
    ]);

    const snapshot = ledger.snapshot();
    const whole = projected.reduce((total, message) => total + estimateTokens(message), 0);

    expect(snapshot.occupied).toBe(whole);
    // Concretely: nowhere near the 6,000 the provider reported for the request
    // as a whole, because these four messages are not that request.
    expect(snapshot.occupied).toBeLessThan(reported);
  });

  it("does not count the same tokens under two sections", async () => {
    // A ledger whose parts exceed its whole cannot be used to decide anything.
    // The failure it guards against is real: the harness estimator reports the
    // provider's usage for the entire conversation as soon as the list it is
    // given contains an assistant message carrying one, so measuring two halves
    // that way counts the run once per half.
    const window = 4_096;
    const ledger = new ContextLedger(window);
    const compactor = new RunCompactor({
      model: model(window),
      runtime: summariser(),
      apiKey: "local",
      ledger,
    });

    const projected = await compactor.transform(passageHeavyRun(12));
    const whole = projected.reduce((total, message) => total + estimateTokens(message), 0);

    // Nothing set `system`, `skill` or `toolSchema` here, so everything
    // occupied is something this projection actually contains.
    expect(ledger.snapshot().occupied).toBe(whole);
  });
});

describe("a long synthetic run stays inside the model window", () => {
  it("never projects a context at or over the window, across many turns", async () => {
    // The regression in one test: the inference server refuses a prompt at or
    // over its window, so a run that exceeds it once does not degrade — it
    // stops, mid-task, in front of whoever was watching.
    const window = 8_192;
    const notes = new WorkingNotes();
    const ledger = new ContextLedger(window);
    const compactor = new RunCompactor({
      model: model(window),
      runtime: summariser(),
      apiKey: "local",
      notes,
      ledger,
      preserved: () => ({
        activePlan: "Step 4 of 6: draft the note.",
        policyDecisions: ["create_docx was approved by kiran."],
      }),
    });

    const transcript: AgentMessage[] = [];
    for (let turn = 0; turn < 60; turn++) {
      transcript.push(user(`question ${turn} ${"x".repeat(500)}`));
      transcript.push(toolCall(`call_${turn}`));
      transcript.push(
        toolResult(
          `call_${turn}`,
          `[E${turn + 1}] Maintenance SOP, page ${turn}\n${"y".repeat(1_200)}`,
        ),
      );
      transcript.push(assistant(`answer ${turn}`));
      notes.sawEvidence(`E${turn + 1}`);

      const projected = await compactor.transform(transcript);

      expect(pairingIsIntact(projected)).toBe(true);
      // Measured the same way the runtime measures it before deciding whether
      // a request may be sent.
      const measured = new ContextLedger(window);
      measured.setMessages("transcript", projected);
      expect(measured.get("transcript")).toBeLessThan(window);
    }

    // And it did so by compacting, not by luck of a short transcript.
    expect(compactor.compactions).toBeGreaterThan(0);
    expect(transcript).toHaveLength(240);
  });
});
