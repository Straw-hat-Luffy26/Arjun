import { describe, expect, it } from "vitest";
import {
  applyThinkingPolicy,
  DEFAULT_BASE_URL,
  hasSeparableReasoning,
  LOCAL_PROVIDERS,
  payloadPolicy,
  PROVIDER_LABEL,
} from "./providers.js";

describe("recognising models that emit a reasoning block", () => {
  it("spots the families that do", () => {
    for (const id of [
      "Qwen3-8B-Instruct",
      "qwen3-30b-a3b",
      "Qwen/Qwen3-Coder-30B",
      "QwQ-32B-Preview",
      "nemotron-3-nano",
      "Nemotron-3_Super",
    ]) {
      expect(hasSeparableReasoning(id), id).toBe(true);
    }
  });

  it("leaves everything else alone", () => {
    // Qwen2.5 does not emit a reasoning block, so patching it would send a
    // template argument its chat template does not know.
    for (const id of [
      "Qwen2.5-Coder-7B-Instruct",
      "Meta-Llama-3-8B-Instruct",
      "mistral-7b-instruct",
      "gemma-3-4b-it",
      "phi-4",
    ]) {
      expect(hasSeparableReasoning(id), id).toBe(false);
    }
  });
});

describe("the thinking policy", () => {
  it("turns thinking off explicitly rather than relying on the default", () => {
    // The whole point: a Qwen3 served without this thinks by default and the
    // block lands in the visible answer.
    const patched = applyThinkingPolicy({ model: "Qwen3-8B" }, "Qwen3-8B", false);
    expect(patched).toMatchObject({
      chat_template_kwargs: { enable_thinking: false },
    });
  });

  it("turns thinking on when the routed model is a reasoning one", () => {
    const patched = applyThinkingPolicy({ model: "Qwen3-8B" }, "Qwen3-8B", true);
    expect(patched).toMatchObject({
      chat_template_kwargs: { enable_thinking: true },
    });
  });

  it("keeps Nemotron from returning an empty turn when thinking is off", () => {
    const patched = applyThinkingPolicy({}, "nemotron-3-nano", false) as Record<string, never>;
    expect(patched.chat_template_kwargs).toMatchObject({
      enable_thinking: false,
      force_nonempty_content: true,
    });
  });

  it("does not force content when Nemotron is allowed to think", () => {
    const patched = applyThinkingPolicy({}, "nemotron-3-nano", true) as Record<string, never>;
    expect(patched.chat_template_kwargs).not.toHaveProperty("force_nonempty_content");
  });

  it("returns a model that needs nothing completely untouched", () => {
    const payload = { model: "Qwen2.5-Coder-7B", messages: [] };
    expect(applyThinkingPolicy(payload, "Qwen2.5-Coder-7B", false)).toBe(payload);
  });

  it("preserves template arguments the caller already set", () => {
    const patched = applyThinkingPolicy(
      { chat_template_kwargs: { custom: "keep me" } },
      "Qwen3-8B",
      false,
    ) as Record<string, Record<string, unknown>>;
    expect(patched.chat_template_kwargs).toMatchObject({
      custom: "keep me",
      enable_thinking: false,
    });
  });

  it("does not mutate the payload it was given", () => {
    const payload = { model: "Qwen3-8B" };
    applyThinkingPolicy(payload, "Qwen3-8B", false);
    expect(payload).toEqual({ model: "Qwen3-8B" });
  });

  it("ignores a payload that is not an object rather than throwing", () => {
    for (const payload of [null, undefined, "text", 42, []]) {
      expect(() => applyThinkingPolicy(payload, "Qwen3-8B", false)).not.toThrow();
      expect(applyThinkingPolicy(payload, "Qwen3-8B", false)).toBe(payload);
    }
  });
});

describe("payloadPolicy", () => {
  it("keys on the model the transport is actually about to call", () => {
    const hook = payloadPolicy(false);
    expect(hook({}, { id: "Qwen3-8B" })).toMatchObject({
      chat_template_kwargs: { enable_thinking: false },
    });
    expect(hook({ a: 1 }, { id: "Meta-Llama-3-8B" })).toEqual({ a: 1 });
  });
});

describe("provider identity", () => {
  it("covers both runtimes and nothing else", () => {
    // Two providers is the claim: C++ and Python. A third appearing here
    // without a matching Rust runtime would mean the two sides disagree.
    expect(Object.values(LOCAL_PROVIDERS).sort()).toEqual(["llama-cpp", "vllm"]);
  });

  it("has a label and a default endpoint for each", () => {
    for (const provider of Object.values(LOCAL_PROVIDERS)) {
      expect(PROVIDER_LABEL[provider]).toBeTruthy();
      expect(DEFAULT_BASE_URL[provider]).toMatch(/^http:\/\/127\.0\.0\.1:\d+\/v1$/);
    }
  });

  it("defaults to loopback for both, never to a public host", () => {
    for (const url of Object.values(DEFAULT_BASE_URL)) {
      expect(new URL(url).hostname).toBe("127.0.0.1");
    }
  });
});

/**
 * The four combinations Rust can report, pinned.
 *
 * `RunRequest.model.reasoning` is `emits_reasoning` and
 * `RunRequest.model.hasReasoningToggle` is `supports_toggled_reasoning`, and
 * `startRun` passes them to `payloadPolicy(reasoningWanted, supportsToggle)` in
 * that order. They used to arrive under names ambiguous enough that they were
 * read the other way round, so the gate was "does it reason" and the value was
 * "does it have a switch". Nothing failed loudly: reasoning was simply always
 * on for switchable models, and always-reasoning models were sent a kwarg their
 * template does not branch on.
 *
 * A table rather than prose, because the defect was that two booleans looked
 * interchangeable.
 */
describe("the two reasoning capabilities are not interchangeable", () => {
  const payload = { messages: [] };
  const kwargs = (emitsReasoning: boolean, hasToggle: boolean) =>
    (payloadPolicy(emitsReasoning, hasToggle)(payload, { id: "unpatterned-model-8b" }) as {
      chat_template_kwargs?: Record<string, unknown>;
    }).chat_template_kwargs;

  it("asks a switchable reasoning model to think", () => {
    // Qwen3: template branches on the variable, and reasons when it is on.
    expect(kwargs(true, true)?.enable_thinking).toBe(true);
  });

  it("sends nothing to a model that always reasons but has no switch", () => {
    // Nemotron. This is the case the old wiring got wrong: the gate was
    // `emits_reasoning`, which is true here, so it sent `enable_thinking:
    // false` — a variable the template does not read, and one vLLM can refuse
    // the whole request over.
    expect(kwargs(true, false)).toBeUndefined();
  });

  it("sends nothing to a model that does not reason at all", () => {
    expect(kwargs(false, false)).toBeUndefined();
  });

  it("never lets the two booleans be swapped without a test noticing", () => {
    // The swap, expressed directly: reading the pair in the wrong order turns
    // the "always reasons, no switch" case into a kwarg being sent.
    const correct = kwargs(true, false);
    const swapped = kwargs(false, true);
    expect(correct).toBeUndefined();
    expect(swapped?.enable_thinking).toBe(false);
    expect(correct).not.toEqual(swapped);
  });
});

describe("applyThinkingPolicy: the capability comes from the model, not its name", () => {
  const payload = { messages: [] };

  it("honours an explicit capability for a model no pattern recognises", () => {
    // The case the name match cannot answer: a model nobody has added to the
    // regex, whose own chat template branches on `enable_thinking`.
    const out = applyThinkingPolicy(payload, "some-new-reasoner-8b", true, true) as {
      chat_template_kwargs?: Record<string, unknown>;
    };
    expect(out.chat_template_kwargs?.enable_thinking).toBe(true);
  });

  it("leaves a model alone when it says it has no reasoning switch", () => {
    // The other direction, and the one that matters for correctness: a
    // fine-tune whose id still matches the pattern but whose template no
    // longer has the variable must not be sent the kwarg.
    const out = applyThinkingPolicy(payload, "qwen3-something-distilled", true, false);
    expect(out).toEqual(payload);
  });

  it("falls back to the name match when the capability is not supplied", () => {
    const known = applyThinkingPolicy(payload, "qwen3-9b", true) as {
      chat_template_kwargs?: Record<string, unknown>;
    };
    expect(known.chat_template_kwargs?.enable_thinking).toBe(true);
    expect(applyThinkingPolicy(payload, "gemma-3-12b-it", true)).toEqual(payload);
  });
});
