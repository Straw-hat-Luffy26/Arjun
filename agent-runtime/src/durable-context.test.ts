import { readFileSync } from "node:fs";
import { describe, expect, it, vi } from "vitest";
import type { AgentMessage } from "@openclaw/agent-core";
import { DurableContext, scopedPeer, type ContextCommit, type RequestPeer } from "./durable-context.js";
import { WorkingNotes } from "./working-notes.js";

const identity = { protocolVersion: 1 as const, attemptId: "attempt-1", fenceToken: 1 };
const notes = new WorkingNotes().state;
const user = { role: "user", content: [{ type: "text", text: "source A-17 https://example.invalid/spec#A-17" }], timestamp: 1 } as AgentMessage;
function receipt(request: ContextCommit) {
  return { protocolVersion: 1, runId: request.runId, revision: request.expectedRevision + 1,
    checkpointId: request.commitId, rawSeq: request.expectedRevision + request.entries.length,
    projectionSeq: 0, phase: request.phase, messages: request.projection ?? [], notes: request.notes,
    ledger: null, pendingApprovals: [], unsettledEffects: [] };
}

describe("durable context handshake", () => {
  it("uses the same serialized request shape as the Rust storage test", async () => {
    const fixture = JSON.parse(readFileSync(new URL("../../contracts/runtime-context-v1.json", import.meta.url), "utf8")) as ContextCommit;
    const sent: ContextCommit[] = [];
    const peer = { request: vi.fn(async (_method: string, value: unknown) => { sent.push(value as ContextCommit); return receipt(value as ContextCommit); }) };
    const client = new DurableContext(scopedPeer(peer, fixture.runId, identity), fixture.runId, identity);
    await client.commit(fixture.phase, fixture.notes, fixture.ledger, { message: fixture.entries[0]!.message, projection: fixture.projection! });
    expect(Object.keys(sent[0]!).sort()).toEqual(Object.keys(fixture).sort());
    expect(sent[0]!.notes.nextAction).toBe("Read source A-17");
    expect(sent[0]!.projection).toEqual(fixture.projection);
  });

  it("serializes simultaneous tool observations and waits for each acknowledgment", async () => {
    const sent: ContextCommit[] = [];
    let release!: () => void;
    const hold = new Promise<void>((resolve) => { release = resolve; });
    const peer: RequestPeer = { request: async (_method, value) => {
      const request = value as ContextCommit; sent.push(request);
      if (sent.length === 1) await hold;
      return receipt(request);
    } };
    const client = new DurableContext(peer, "run", identity);
    const first = client.commit("observed", notes, null, { message: user });
    const second = client.commit("observed", notes, null, { message: user });
    await Promise.resolve(); expect(sent).toHaveLength(1);
    release(); await Promise.all([first, second]);
    expect(sent.map((request) => request.expectedRevision)).toEqual([0, 1]);
  });

  it("poisons further progress after a failed durable write", async () => {
    const request = vi.fn(async () => { throw new Error("injected disk failure"); });
    const client = new DurableContext({ request }, "run", identity);
    await expect(client.commit("observed", notes, null, { message: user })).rejects.toThrow(/checkpoint/);
    await expect(client.commit("modelReady", notes, null, { projection: [user] })).rejects.toThrow(/checkpoint/);
    expect(request).toHaveBeenCalledTimes(1);
  });

  it("rejects a mismatched acknowledgment rather than advancing its revision", async () => {
    const client = new DurableContext({ request: async (_method, value) => ({ ...receipt(value as ContextCommit), revision: 90 }) }, "run", identity);
    await expect(client.commit("observed", notes, null)).rejects.toThrow(/acknowledgment/);
  });
});
