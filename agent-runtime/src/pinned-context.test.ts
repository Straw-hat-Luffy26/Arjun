/**
 * A pin has to keep something. That is the whole of the control.
 *
 * ## The defect
 *
 * The context meter draws a pin beside every evictable row, labelled "Keep this
 * when the window fills". Pressing it toggled a `Set` in the component's own
 * React state and stopped there. The icon darkened, the "what goes first" line
 * moved on to the next row, and the compactor — in another process, and having
 * never heard of any of this — cleared the pinned result on its next pass
 * exactly as if nothing had been pressed.
 *
 * That is the worst shape a control can have. Somebody who pinned the drawing
 * they were working from, watched the meter fill, and carried on had every
 * reason to believe the drawing was safe. A control that did nothing at all
 * would at least have left them looking for another way.
 *
 * These tests are about the far end of the wire: given that a pin arrives in
 * `PreservedState`, the compactor keeps what it names.
 */
import { describe, expect, it } from "vitest";
import type { AgentMessage } from "@openclaw/agent-core";
import { pruneStaleToolResults } from "./compaction.js";
import { ContextLedger } from "./context-ledger.js";

/** A tool result carrying retrieved passages, the largest thing in a run's context. */
function toolResult(callId: string, text: string): AgentMessage {
  return {
    role: "toolResult",
    toolCallId: callId,
    toolName: "knowledge.search_authorized",
    content: [{ type: "text", text }],
    isError: false,
    timestamp: 1,
  } as unknown as AgentMessage;
}

/** Six trailing messages are never pruned, so pad past that window. */
function padded(target: AgentMessage): AgentMessage[] {
  const filler: AgentMessage[] = [];
  for (let i = 0; i < 8; i++) {
    filler.push({
      role: "user",
      content: [{ type: "text", text: `later turn ${i}` }],
      timestamp: 2 + i,
    } as AgentMessage);
  }
  return [target, ...filler];
}

/** Whether the passage text survived, rather than being replaced by a stub. */
function stillHasText(messages: AgentMessage[], needle: string): boolean {
  return messages.some((message) => {
    const content = (message as { content?: unknown }).content;
    if (!Array.isArray(content)) return false;
    return content.some(
      (block) =>
        typeof (block as { text?: unknown }).text === "string" &&
        (block as { text: string }).text.includes(needle),
    );
  });
}

describe("pinned context survives pruning", () => {
  /**
   * The baseline. Without a pin, a result whose marker is durable is cleared —
   * which is correct, and is exactly what a pin has to be able to override.
   */
  it("clears a durable result when nothing is pinned", () => {
    const messages = padded(toolResult("call-1", "[E3] the seal is 9.0 mm"));
    const { messages: pruned, cleared } = pruneStaleToolResults(messages, ["E3"]);

    expect(cleared).toBe(1);
    expect(stillHasText(pruned, "the seal is 9.0 mm")).toBe(false);
  });

  it("keeps that same result when its marker is pinned", () => {
    const messages = padded(toolResult("call-1", "[E3] the seal is 9.0 mm"));
    const { messages: pruned, cleared } = pruneStaleToolResults(messages, ["E3"], ["E3"]);

    expect(cleared).toBe(0);
    expect(stillHasText(pruned, "the seal is 9.0 mm")).toBe(true);
  });

  /**
   * The meter's rows are keyed by content hash for documents and by marker for
   * evidence, and a pin has to mean the same thing whichever row it was pressed
   * on — otherwise pinning a drawing works and pinning a passage does not, with
   * nothing on screen to say which is which.
   */
  it("keeps a result pinned by document id rather than by marker", () => {
    const sha = "ab".repeat(32);
    const messages = padded(toolResult("call-1", `[E4] from ${sha}: clause 7.2`));
    const { messages: pruned, cleared } = pruneStaleToolResults(messages, ["E4"], [sha]);

    expect(cleared).toBe(0);
    expect(stillHasText(pruned, "clause 7.2")).toBe(true);
  });

  it("is case-insensitive, so a pin is not lost to how an id was spelled", () => {
    const messages = padded(toolResult("call-1", "[E3] the seal is 9.0 mm"));
    const { cleared } = pruneStaleToolResults(messages, ["E3"], ["e3"]);
    expect(cleared).toBe(0);
  });

  it("still clears the results that were not pinned", () => {
    const messages = [
      toolResult("call-1", "[E1] pinned passage"),
      toolResult("call-2", "[E2] ordinary passage"),
      ...padded(toolResult("call-3", "[E3] another ordinary passage")).slice(1),
    ];
    const { messages: pruned, cleared } = pruneStaleToolResults(
      messages,
      ["E1", "E2", "E3"],
      ["E1"],
    );

    // A pin protects what it names and nothing else. Protecting everything
    // would make the window fill and the run fail rather than degrade, which is
    // not what the person asked for.
    expect(cleared).toBe(1);
    expect(stillHasText(pruned, "pinned passage")).toBe(true);
    expect(stillHasText(pruned, "ordinary passage")).toBe(false);
  });

  it("changes nothing when the pin list is empty", () => {
    const messages = padded(toolResult("call-1", "[E3] the seal is 9.0 mm"));
    const withNone = pruneStaleToolResults(messages, ["E3"], []);
    const withDefault = pruneStaleToolResults(messages, ["E3"]);
    expect(withNone.cleared).toBe(withDefault.cleared);
  });

  /**
   * An empty string matches every message under a naive `includes`, so a stray
   * one in the list would silently disable pruning for the whole run — a window
   * that fills and a turn that fails, from one blank id.
   */
  it("ignores an empty id rather than matching everything", () => {
    const messages = padded(toolResult("call-1", "[E3] the seal is 9.0 mm"));
    const { cleared } = pruneStaleToolResults(messages, ["E3"], ["", "  "]);
    expect(cleared).toBe(1);
  });
});

/**
 * Unpinning has to release, or the meter lies in the other direction.
 *
 * The pinned set arrives whole on every `run.note`, so an id that has left it
 * has been unpinned. A ledger that only ever set `pinned: true` would keep the
 * row drawn as protected for the rest of the run while the compactor — reading
 * the same list — correctly stopped protecting it. The panel would then be
 * promising something nothing was doing, which is the failure this control was
 * wired up to remove, pointing the other way.
 */
describe("the ledger's pinned set follows what arrives", () => {
  function ledgerWith(...ids: string[]): ContextLedger {
    const ledger = new ContextLedger(32_000);
    for (const id of ids) {
      ledger.upsertEntity({
        id,
        label: id,
        section: "evidence",
        tokens: 100,
        status: "active",
        pinned: false,
        measurement: "estimated",
      } as never);
    }
    return ledger;
  }

  function pinnedIds(ledger: ContextLedger): string[] {
    const snapshot = ledger.snapshot() as unknown as {
      entities?: { id: string; pinned: boolean }[];
    };
    return (snapshot.entities ?? []).filter((e) => e.pinned).map((e) => e.id);
  }

  it("pins what arrives", () => {
    const ledger = ledgerWith("doc-a", "doc-b");
    ledger.applyPins(["doc-a"]);
    expect(pinnedIds(ledger)).toEqual(["doc-a"]);
  });

  it("releases what stopped arriving", () => {
    const ledger = ledgerWith("doc-a", "doc-b");
    ledger.applyPins(["doc-a"]);
    // The person changed their mind: the next note carries a set without it.
    ledger.applyPins([]);
    expect(pinnedIds(ledger)).toEqual([]);
  });

  it("moves a pin from one row to another", () => {
    const ledger = ledgerWith("doc-a", "doc-b");
    ledger.applyPins(["doc-a"]);
    ledger.applyPins(["doc-b"]);
    expect(pinnedIds(ledger)).toEqual(["doc-b"]);
  });

  /**
   * The same case-insensitivity `pruneStaleToolResults` applies, so a pin
   * cannot be honoured by one and dropped by the other over how it was spelled.
   */
  it("matches ids case-insensitively, as the pruner does", () => {
    const ledger = ledgerWith("E3");
    ledger.applyPins(["e3"]);
    expect(pinnedIds(ledger)).toEqual(["E3"]);
  });
});
