import { describe, expect, it, vi } from "vitest";

/** A literal newline, kept out of the template strings below. */
const NL = String.fromCharCode(10);
import type { BeforeToolCallContext } from "@openclaw/agent-core";
import { RpcPeer, type PeerTransport } from "./peer.js";
import { isImageWithMediaPayload } from "../vendor/openclaw/packages/ai/src/providers/tool-result-text.js";
import { GrantLedger, authorizeToolCall, buildTools, type Verdict } from "./tools.js";

/** A peer whose `request` is scripted, so the gateway's answer is the variable. */
function scriptedPeer(script: (method: string, params: unknown) => Promise<unknown>): RpcPeer {
  const silent: PeerTransport = { write: () => {}, onData: () => {}, onClose: () => {} };
  const peer = new RpcPeer(silent);
  vi.spyOn(peer, "request").mockImplementation((method, params) => script(method, params));
  return peer;
}

function callContext(toolCallId = "tc-1", name = "knowledge.search_authorized", args: unknown = { query: "seal spec" }) {
  return {
    toolCall: { type: "toolCall", id: toolCallId, name, arguments: args },
    args,
    assistantMessage: {},
    context: {},
  } as unknown as BeforeToolCallContext;
}

describe("authorizeToolCall", () => {
  it("records the grant and lets the call through when the gateway allows it", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => ({ outcome: "allow", tool: "knowledge.search_authorized", grant: "g-1" } satisfies Verdict));

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext());

    expect(result).toBeUndefined();
    expect(ledger.size).toBe(1);
  });

  it("blocks with the gateway's own words when refused", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => ({ outcome: "refuse", reason: "SearchKnowledge not held" } satisfies Verdict));

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext());

    expect(result).toEqual({ block: true, reason: "SearchKnowledge not held" });
    expect(ledger.size).toBe(0);
  });

  it("blocks rather than assuming consent when a person is required", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () =>
      ({ outcome: "needsApproval", tool: "workspace.write_text", summary: "Write 5 bytes to note.txt" } satisfies Verdict),
    );

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext("tc-2", "workspace.write_text"));

    expect(result?.block).toBe(true);
    expect(result?.reason).toContain("Write 5 bytes to note.txt");
    expect(ledger.size).toBe(0);
  });

  it("fails closed when the gateway cannot be reached", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => {
      throw new Error("core closed the channel");
    });

    const result = await authorizeToolCall(peer, ledger, "run-1", callContext());

    expect(result?.block).toBe(true);
    expect(result?.reason).toContain("authorisation is unavailable");
    expect(ledger.size).toBe(0);
  });

  it("sends the run, call, tool and arguments the gateway needs to decide", async () => {
    const ledger = new GrantLedger();
    const seen: unknown[] = [];
    const peer = scriptedPeer(async (method, params) => {
      seen.push({ method, params });
      return { outcome: "allow", tool: "knowledge.search_authorized", grant: "g" } satisfies Verdict;
    });

    await authorizeToolCall(peer, ledger, "run-42", callContext("tc-9"));

    expect(seen[0]).toEqual({
      method: "tool.authorize",
      params: { runId: "run-42", toolCallId: "tc-9", tool: "knowledge.search_authorized", args: { query: "seal spec" } },
    });
  });
});

describe("GrantLedger", () => {
  it("hands a grant out exactly once", () => {
    const ledger = new GrantLedger();
    ledger.put("tc-1", "g-1");

    expect(ledger.take("tc-1")).toBe("g-1");
    expect(ledger.take("tc-1")).toBeUndefined();
  });

  it("clears everything, so a grant cannot outlive its run", () => {
    const ledger = new GrantLedger();
    ledger.put("tc-1", "g-1");
    ledger.put("tc-2", "g-2");
    ledger.clear();
    expect(ledger.size).toBe(0);
  });
});

describe("host tools", () => {
  it("exposes only tools the Rust catalogue also knows", () => {
    const peer = scriptedPeer(async () => ({}));
    const names = buildTools(peer, new GrantLedger(), "run-1", "qwen2.5-coder-7b").map((tool) => tool.name);
    // Rust's `ToolName` enum is the authority; a name here that is absent
    // there is refused by the gateway regardless of what this declares.
    expect(names.slice().sort()).toEqual([
      "agent.delegate_readonly",
      "artifact.create_approval_note",
      "artifact.create_briefing_deck",
      "artifact.create_calculation_workbook",
      "artifact.create_chart",
      "artifact.create_diagram",
      "artifact.create_pdf",
      "artifact.create_table",
      "artifact.verify_docx",
      "calculation.evaluate_with_units",
      "capability.search",
      "document.read_pages",
      "document.search",
      "knowledge.build_graph",
      "knowledge.load_evidence_region",
      "knowledge.multimodal_retrieve",
      "knowledge.search_authorized",
      "media.extract_findings",
      "memory.promote_approved",
      "memory.recall_authorized",
      "notebook.add_source",
      "notebook.create",
      "notebook.delete",
      "notebook.list",
      "notebook.list_sources",
      "notebook.remove_source",
      "notebook.rename",
      "sandbox.run_code",
      "sovereignty.get_evidence",
      "workspace.read_text",
      "workspace.write_text",
    ]);
  });

  it("refuses to execute without a grant, so the gateway cannot be skipped", async () => {
    const peer = scriptedPeer(async () => ({ text: "should never be reached" }));
    const tool = buildTools(peer, new GrantLedger(), "run-1", "qwen2.5-coder-7b").find((t) => t.name === "knowledge.search_authorized")!;

    await expect(tool.execute("tc-unauthorised", { query: "x" })).rejects.toMatchObject({
      code: "refused",
    });
  });

  it("spends the grant on execution and cannot reuse it", async () => {
    const ledger = new GrantLedger();
    const calls: unknown[] = [];
    const peer = scriptedPeer(async (method, params) => {
      calls.push({ method, params });
      return { text: "3 passages", details: { hits: 3 } };
    });
    const tool = buildTools(peer, ledger, "run-1", "qwen2.5-coder-7b").find((t) => t.name === "knowledge.search_authorized")!;
    ledger.put("tc-1", "g-1");

    const result = await tool.execute("tc-1", { query: "seal spec" });

    expect(result.content).toEqual([{ type: "text", text: "3 passages" }]);
    expect(result.details).toEqual({ hits: 3 });
    expect(calls[0]).toEqual({
      method: "tool.execute",
      params: {
        runId: "run-1",
        toolCallId: "tc-1",
        tool: "knowledge.search_authorized",
        args: { query: "seal spec" },
        grant: "g-1",
        model: "qwen2.5-coder-7b",
      },
    });

    // A second execution of the same call has no grant left to spend.
    await expect(tool.execute("tc-1", { query: "seal spec" })).rejects.toMatchObject({ code: "refused" });
  });

  it("describes the tool well enough for a model to know when to reach for it", () => {
    const peer = scriptedPeer(async () => ({}));
    const tool = buildTools(peer, new GrantLedger(), "run-1", "qwen2.5-coder-7b").find((t) => t.name === "knowledge.search_authorized")!;
    expect(tool.description).toMatch(/permitted to read/i);
    expect(tool.description).toMatch(/do not answer such questions from memory/i);
  });

  it("forwards multimodal tool results as structured content parts", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => ({
      text: "1 passage, 1 region, 1 table.",
      images: [{
        mime: "image/png",
        data: "BASE64",
        caption: "PT-2201 on a P&ID",
        bbox: { left: 0.1, top: 0.2, right: 0.3, bottom: 0.4 },
      }],
      tables: [{
        headers: ["K", "V"],
        rows: [["design pressure", "14 bar"]],
        citation: "Pump Manual, page 4 (table)",
      }],
    }));
    const tool = buildTools(peer, ledger, "run-1", "qwen2.5-vl-3b")
      .find((t) => t.name === "knowledge.multimodal_retrieve")!;
    ledger.put("tc-mm", "g-mm");

    const result = await tool.execute("tc-mm", { query: "PT-2201" });

    // Text first, then images. Order matters because some agent-core versions
    // read the first part as the answer and the rest as supplementary.
    //
    // This used to assert `{ source: { type: "base64", media_type, data } }`
    // for the image and a `{ type: "table", headers, rows }` block beside it.
    // Both shapes were pinned here and understood nowhere: `llm-core`'s
    // `ImageContent` wants `data` and `mimeType` at the top level, and there is
    // no table block type at all. The test passed while the rows and the
    // picture reached neither the model nor the person.
    expect(result.content).toEqual([
      {
        type: "text",
        text:
          "1 passage, 1 region, 1 table." + NL + NL +
          "Pump Manual, page 4 (table)" + NL +
          "| K | V |" + NL + "| --- | --- |" + NL + "| design pressure | 14 bar |",
      },
      {
        type: "image",
        data: "BASE64",
        mimeType: "image/png",
        caption: "PT-2201 on a P&ID",
      },
    ]);
  });

  /**
   * The property the shape above exists for.
   *
   * `isImageWithMediaPayload` is what every provider emitter gates image output
   * on, and it reads `block.data` — not `block.source.data`. An image that
   * fails it is dropped silently, and `describeToolResultMediaPlaceholder`
   * fails the same check, so the model is not even told one existed. Asserting
   * the predicate rather than the field names keeps this true if the
   * transport's shape ever moves.
   */
  it("emits images the transport will actually forward", async () => {
    const ledger = new GrantLedger();
    const peer = scriptedPeer(async () => ({
      text: "one region",
      images: [{ mime: "image/png", data: "BASE64", caption: "a valve" }],
    }));
    const tool = buildTools(peer, ledger, "run-1", "qwen2.5-vl-3b")
      .find((t) => t.name === "knowledge.multimodal_retrieve")!;
    ledger.put("tc-img", "g-img");

    const result = await tool.execute("tc-img", { query: "valve" });
    const image = result.content.find(
      (block: { type: string }) => block.type === "image",
    );

    expect(isImageWithMediaPayload(image)).toBe(true);
  });
});
