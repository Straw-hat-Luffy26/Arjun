/**
 * Whether a thinking model's answer reaches the surface while it is written.
 *
 * ARJUN shows reasoning live in a Thinking panel and the answer in the chat
 * cell. On this machine the first streamed and the second did not: a census the
 * runtime writes for every turn read `thinking_delta=529 text_delta=1`, and the
 * raw SSE off the local server for the same prompt carried thirty-one separate
 * `content` frames. Thirty-one frames in, one delta out - so something between
 * the provider and the surface was gluing the answer back together.
 *
 * It was `markStrict()`. The transport calls `beginReasoning(_, true)` the
 * moment a chunk carries `reasoning_content`, and from then on the partitioner
 * re-arms its hold after every consume and releases only at `flush()`, at the
 * end of the turn. Every model that thinks lost answer streaming entirely.
 *
 * These tests exist because that hold is also what keeps chain-of-thought out
 * of `Message.content` - the text ARJUN persists, signs and resolves citations
 * against. Loosening it without pinning the leak cases first would trade a
 * visible annoyance for an invisible and much worse one. So the leak cases are
 * pinned here first, and they must keep passing.
 */
import { describe, expect, it } from "vitest";
import { processCompletionsStream } from "./openai-completions-stream.js";
import {
  type CapturedStreamEvent,
  createAssistantOutput,
  makeCompletionsChunk,
  makeCompletionsModel,
  streamChunks,
} from "./openai-completions.test-support.js";

function collector() {
  const events: CapturedStreamEvent[] = [];
  return { events, stream: { push: (event: unknown) => events.push(event as CapturedStreamEvent) } };
}

const textDeltas = (events: CapturedStreamEvent[]) =>
  events
    .filter((event) => (event as { type?: string }).type === "text_delta")
    .map((event) => (event as { delta?: string }).delta ?? "");

const visibleText = (events: CapturedStreamEvent[]) => textDeltas(events).join("");

describe("a thinking model's answer streams as it is written", () => {
  /**
   * The shape the local server actually sends: reasoning in its own field,
   * answer in `content`, several frames of each.
   */
  it("emits one delta per content frame when reasoning arrives out of band", async () => {
    const model = makeCompletionsModel({ id: "nemotron3-nano", name: "Nemotron 3 Nano" });
    const output = createAssistantOutput(model);
    const { events, stream } = collector();

    await processCompletionsStream(
      streamChunks([
        makeCompletionsChunk({ role: "assistant", reasoning_content: "The user wants " }),
        makeCompletionsChunk({ reasoning_content: "a count. Simple." }),
        makeCompletionsChunk({ content: "One. " }),
        makeCompletionsChunk({ content: "Two. " }),
        makeCompletionsChunk({ content: "Three." }, "stop"),
      ]),
      output,
      model,
      stream,
    );

    expect(visibleText(events)).toBe("One. Two. Three.");
    // The regression this file exists for. Before the fix this was 1: the
    // whole answer arrived in a single delta at the end of the turn.
    expect(textDeltas(events).length).toBeGreaterThan(1);
    // And the reasoning stayed out of the answer, which is the whole reason
    // the hold existed.
    expect(visibleText(events)).not.toContain("The user wants");
  });

  /**
   * The leak case. A model with no separate reasoning field puts its thinking
   * inline, and the answer is only the part after the closing tag.
   */
  it("still hides reasoning written inline as tags", async () => {
    const model = makeCompletionsModel({ id: "tagged-thinker", name: "Tagged Thinker" });
    const output = createAssistantOutput(model);
    const { events, stream } = collector();

    await processCompletionsStream(
      streamChunks([
        makeCompletionsChunk({ role: "assistant", content: "<think>secret plan" }),
        makeCompletionsChunk({ content: " still thinking</think>" }),
        makeCompletionsChunk({ content: "The answer is four." }, "stop"),
      ]),
      output,
      model,
      stream,
    );

    const visible = visibleText(events);
    expect(visible).not.toContain("secret plan");
    expect(visible).not.toContain("still thinking");
    expect(visible).not.toContain("<think>");
    expect(visible).toContain("The answer is four.");
  });

  /**
   * The mixed case, and the one most likely to break under a careless change:
   * a separate reasoning field *and* tags in the content stream.
   */
  it("hides inline tags even when a reasoning field is also present", async () => {
    const model = makeCompletionsModel({ id: "both-ways", name: "Both Ways" });
    const output = createAssistantOutput(model);
    const { events, stream } = collector();

    await processCompletionsStream(
      streamChunks([
        makeCompletionsChunk({ role: "assistant", reasoning_content: "field reasoning" }),
        makeCompletionsChunk({ content: "<think>inline reasoning</think>" }),
        makeCompletionsChunk({ content: "Final answer." }, "stop"),
      ]),
      output,
      model,
      stream,
    );

    const visible = visibleText(events);
    expect(visible).not.toContain("inline reasoning");
    expect(visible).not.toContain("field reasoning");
    expect(visible).not.toContain("<think>");
    expect(visible).toContain("Final answer.");
  });

  /**
   * A tag split across frame boundaries. The partitioner must not emit the
   * fragment while it is still deciding what it is.
   */
  it("does not leak a tag split across frames", async () => {
    const model = makeCompletionsModel({ id: "split-tag", name: "Split Tag" });
    const output = createAssistantOutput(model);
    const { events, stream } = collector();

    await processCompletionsStream(
      streamChunks([
        makeCompletionsChunk({ role: "assistant", reasoning_content: "warming up" }),
        makeCompletionsChunk({ content: "<thi" }),
        makeCompletionsChunk({ content: "nk>hidden</think>" }),
        makeCompletionsChunk({ content: "Visible." }, "stop"),
      ]),
      output,
      model,
      stream,
    );

    const visible = visibleText(events);
    expect(visible).not.toContain("hidden");
    expect(visible).not.toContain("<thi");
    expect(visible).toContain("Visible.");
  });

  /**
   * Nothing is dropped. Streaming earlier must not lose the tail of an answer
   * that the old code only ever emitted at `flush()`.
   */
  it("delivers the whole answer, not just the streamed part", async () => {
    const model = makeCompletionsModel({ id: "whole-answer", name: "Whole Answer" });
    const output = createAssistantOutput(model);
    const { events, stream } = collector();

    await processCompletionsStream(
      streamChunks([
        makeCompletionsChunk({ role: "assistant", reasoning_content: "thinking" }),
        makeCompletionsChunk({ content: "alpha " }),
        makeCompletionsChunk({ content: "beta " }),
        makeCompletionsChunk({ content: "gamma" }, "stop"),
      ]),
      output,
      model,
      stream,
    );

    expect(visibleText(events)).toBe("alpha beta gamma");
  });
});
