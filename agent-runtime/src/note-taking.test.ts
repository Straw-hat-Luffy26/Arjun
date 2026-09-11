/**
 * What the notes learn without the model's help.
 *
 * The failure guarded here is the one that costs real work: a run whose process
 * died after `create_docx` wrote the approval note, resuming and writing it
 * again. That only stays prevented while the effect is recorded by the code
 * that saw the tool succeed — a model asked to note its own side effects
 * records the interesting ones and forgets the completed ones.
 */

import { describe, expect, it } from "vitest";
import { markersIn, nextCalculationId, observeToolResult } from "./note-taking.js";
import { WorkingNotes } from "./working-notes.js";

describe("evidence markers", () => {
  it("takes every marker the model was shown, once each", () => {
    const text = "[E1] SOP, page 4\ntext\n\n[E2] SOP, page 5\nmore\n\n[E1] again";

    expect(markersIn(text)).toEqual(["E1", "E2"]);
  });

  it("finds none in a result that carries none", () => {
    expect(markersIn('No passages matched "unicorns".')).toEqual([]);
  });

  it("records a search's markers into the notes", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "search_documents",
      args: { query: "seal wear" },
      text: "[E1] SOP, page 4\ntext\n\n[E2] SOP, page 5\nmore",
    });

    expect(notes.state.evidenceIds).toEqual(["E1", "E2"]);
  });

  it("records markers from a loaded page range the same way", () => {
    // Both tools feed one numbered table, so both must feed one note list —
    // otherwise a marker pulled back by page would never become durable, and
    // its raw result would never be cleared.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "load_more_evidence",
      args: { documentSha256: "sop", fromPage: 11, toPage: 13 },
      text: "Read pages 11 to 13 of Maintenance SOP.\n\n[E3] SOP, page 11\ntext",
    });

    expect(notes.state.evidenceIds).toEqual(["E3"]);
  });

  it("does not read markers out of a tool that does not produce evidence", () => {
    // A drafted document quoting "[E1]" is the model citing, not the tool
    // retrieving. Treating it as evidence would make a marker durable that
    // nothing in the evidence table backs.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "read_scoped_file",
      args: { path: "draft.md" },
      text: "The seal is worn beyond the limit [E1].",
    });

    expect(notes.state.evidenceIds).toEqual([]);
  });
});

describe("side effects a resumption must not repeat", () => {
  it("records a produced document by name, not by path", () => {
    // The path contains the run's own workspace directory, which differs on
    // every attempt. A resumption comparing full paths would never recognise
    // its own earlier work and would write the document again.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "create_docx",
      args: { path: "C:\\data\\runs\\run-1\\approval-note.docx", template: "approval_note" },
      text: "Created approval-note.docx.",
    });

    // Findable by either spelling, and recorded under the current one.
    expect(notes.hasDone("create_docx", "approval-note.docx")).toBe(true);
    expect(notes.hasDone("artifact.create_approval_note", "approval-note.docx")).toBe(true);
    expect(notes.state.completed[0]?.tool).toBe("artifact.create_approval_note");
    expect(notes.state.artifactIds).toEqual(["approval-note.docx"]);
  });

  it("records a workspace write as an effect", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "write_scoped_file",
      args: { path: "draft.md", content: "…" },
      text: "Wrote draft.md.",
    });

    expect(notes.hasDone("write_scoped_file", "draft.md")).toBe(true);
    expect(notes.hasDone("workspace.write_text", "draft.md")).toBe(true);
  });

  it("records that code ran without claiming which execution it was", () => {
    // An execution has no stable name. Saying one happened warns a resumption;
    // naming it would let the resumption conclude a *different* execution was
    // already done and skip it.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "execute_code",
      args: { language: "python", source: "print(1)" },
      text: "1",
    });

    const completed = notes.state.completed;
    expect(completed).toHaveLength(1);
    expect(completed[0]?.tool).toBe("sandbox.run_code");
    // Not filed as an artifact: nothing was produced that can be re-opened.
    expect(notes.state.artifactIds).toEqual([]);
  });

  it("does not record a read as something that must not be repeated", () => {
    // Reading is free to redo. Recording it would fill the bounded list with
    // entries no resumption needs, pushing out the ones it does.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "read_scoped_file",
      args: { path: "draft.md" },
      text: "…",
    });

    expect(notes.state.completed).toEqual([]);
  });

  it("counts the same effect once however often it is observed", () => {
    const notes = new WorkingNotes();
    for (let i = 0; i < 5; i++) {
      observeToolResult(notes, {
        tool: "create_docx",
        args: { path: "approval-note.docx" },
        text: "Created approval-note.docx.",
      });
    }

    expect(notes.state.completed).toHaveLength(1);
  });
});

describe("calculations", () => {
  it("records a reference into the engine's record rather than the working", () => {
    // The steps are in the calculation record on the Rust side. Copying them
    // here would put the whole working in the context on every turn, which is
    // what the notes exist to avoid.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "run_calculation",
      args: { expression: "9.0 mm - 7.4 mm" },
      text: "1.6 mm",
    });
    observeToolResult(notes, {
      tool: "run_calculation",
      args: { expression: "1.6 mm / 9.0 mm" },
      text: "0.178",
    });

    expect(notes.state.calculationIds).toEqual(["C1", "C2"]);
    expect(notes.render()).not.toContain("9.0 mm - 7.4 mm");
  });
});

describe("what the notes then let compaction do", () => {
  it("makes a search's markers durable, which is what allows its raw text to be cleared", () => {
    // The chain requirement 12 depends on: the marker has to reach the notes
    // before the raw passage can be dropped, or the reference would point at
    // nothing a resumed run could resolve.
    const notes = new WorkingNotes();
    const text = `[E1] Maintenance SOP, page 4\n${"y".repeat(2_000)}`;
    observeToolResult(notes, { tool: "search_documents", args: { query: "seal" }, text });

    expect(notes.state.evidenceIds).toContain("E1");
    expect(markersIn(text).every((marker) => notes.state.evidenceIds.includes(marker))).toBe(true);
  });
});

/**
 * The canonical names, on the path that actually matters.
 *
 * ## The defect
 *
 * `note-taking.ts` classified tool results against three sets written in the
 * pre-namespace spelling — `create_docx`, `search_documents`,
 * `run_calculation` — while the catalogue hands it the current one. Every
 * comparison was false, and had been since the rename. So:
 *
 *   - a search returned `[E3]` and no marker was recorded;
 *   - a calculation ran and no calculation id was recorded;
 *   - a document was produced and **no completed effect** was recorded.
 *
 * The last one is the failure this whole mechanism exists to prevent. A
 * completed effect is what a resumed run reads to find out what already
 * happened; with none recorded, a run that produced an approval note, lost its
 * process and resumed would produce the note a second time.
 */
describe("canonical tool names are what the notes are keyed on", () => {
  it("records evidence markers for a search under its current name", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "knowledge.search_authorized",
      args: { query: "seal specification" },
      text: "The seal is worn beyond the limit [E1]. Replacement is specified [E2].",
    });
    expect(notes.state.evidenceIds).toEqual(["E1", "E2"]);
  });

  it("records evidence markers for every canonical retrieval tool", () => {
    for (const tool of [
      "knowledge.search_authorized",
      "knowledge.load_evidence_region",
      "knowledge.multimodal_retrieve",
      "media.extract_findings",
    ]) {
      const notes = new WorkingNotes();
      observeToolResult(notes, { tool, args: {}, text: "A passage [E7]." });
      expect(notes.state.evidenceIds, tool).toEqual(["E7"]);
    }
  });

  it("records a calculation id under the current name", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "calculation.evaluate_with_units",
      args: { expression: "40 bar * 2" },
      text: "80 bar",
    });
    expect(notes.state.calculationIds).toEqual(["C1"]);
  });

  it("records an artifact and an effect for every canonical document tool", () => {
    const cases: Array<[string, string]> = [
      ["artifact.create_approval_note", "approval-note.docx"],
      ["artifact.create_calculation_workbook", "working.xlsx"],
      ["artifact.create_briefing_deck", "briefing.pptx"],
      ["workspace.write_text", "draft.md"],
    ];
    for (const [tool, name] of cases) {
      const notes = new WorkingNotes();
      observeToolResult(notes, {
        tool,
        args: { path: `C:\\data\\runs\\run-1\\${name}` },
        text: `Created ${name}.`,
      });
      expect(notes.state.artifactIds, tool).toEqual([name]);
      expect(notes.hasDone(tool, name), tool).toBe(true);
      expect(notes.state.completed[0]?.tool, tool).toBe(tool);
    }
  });

  it("records a sandbox execution as an effect and not as an artifact", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "sandbox.run_code",
      args: { language: "python", source: "print(1)" },
      text: "1",
    });
    expect(notes.state.completed).toHaveLength(1);
    expect(notes.state.completed[0]?.tool).toBe("sandbox.run_code");
    expect(notes.state.artifactIds).toEqual([]);
  });

  it("still records nothing for a read, under either spelling", () => {
    // The control. A rename must not have turned every tool into a
    // side-effecting one either.
    for (const tool of ["workspace.read_text", "read_scoped_file"]) {
      const notes = new WorkingNotes();
      observeToolResult(notes, { tool, args: { path: "notes.md" }, text: "…" });
      expect(notes.state.completed, tool).toEqual([]);
      expect(notes.state.artifactIds, tool).toEqual([]);
    }
  });
});

/**
 * The restart, end to end.
 *
 * A run produces a document, the process goes away, and the notes are handed to
 * the next attempt. What that attempt must not do is produce the document
 * again — and the only thing standing between it and doing so is a completed
 * effect it can recognise.
 */
describe("a canonical side effect survives a restart and is not repeated", () => {
  /** What Rust persists with the task record and hands back on resumption. */
  function persist(notes: WorkingNotes): string {
    return JSON.stringify(notes.state);
  }

  it("does not repeat a document the previous attempt already produced", () => {
    // ── First attempt ────────────────────────────────────────────────
    const first = new WorkingNotes();
    observeToolResult(first, {
      tool: "artifact.create_approval_note",
      args: { path: "C:\\data\\runs\\run-1\\approval-note.docx", template: "approval_note" },
      text: "Created approval-note.docx.",
    });
    expect(first.hasDone("artifact.create_approval_note", "approval-note.docx")).toBe(true);

    // ── The process goes away, and the notes go to disk and back ─────
    const carried = WorkingNotes.from(JSON.parse(persist(first)));

    // ── Second attempt ───────────────────────────────────────────────
    // The workspace path is different on every attempt, which is exactly why
    // the effect is recorded by file name rather than by path.
    expect(
      carried.hasDone("artifact.create_approval_note", "approval-note.docx"),
      "the resumption cannot see what the first attempt did, so it will do it again",
    ).toBe(true);

    // And re-observing the same effect does not double it: the note stays one
    // entry, so a resumption reading the list is not told it happened twice.
    observeToolResult(carried, {
      tool: "artifact.create_approval_note",
      args: { path: "C:\\data\\runs\\run-2\\approval-note.docx", template: "approval_note" },
      text: "Created approval-note.docx.",
    });
    expect(carried.state.completed).toHaveLength(1);
  });

  it("recognises an effect a pre-rename attempt recorded under the old name", () => {
    // The upgrade case: the first attempt ran on a build that wrote
    // `create_docx`, and the resumption is on this one. If the two spellings do
    // not fold together, the resumption sees an empty list and writes the
    // document a second time.
    const legacyNotes = WorkingNotes.from({
      completed: [
        {
          tool: "create_docx",
          target: "approval-note.docx",
          at: "2026-08-27T10:00:42+00:00",
        },
      ],
    });
    expect(legacyNotes.hasDone("artifact.create_approval_note", "approval-note.docx")).toBe(
      true,
    );
    // And it is not recorded a second time under the new spelling.
    observeToolResult(legacyNotes, {
      tool: "artifact.create_approval_note",
      args: { path: "approval-note.docx" },
      text: "Created approval-note.docx.",
    });
    expect(legacyNotes.state.completed).toHaveLength(1);
  });

  it("does not confuse two different documents from the same tool", () => {
    // The other half of the promise: recognising work already done must not
    // become skipping work that was never done.
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "artifact.create_approval_note",
      args: { path: "approval-note.docx" },
      text: "Created approval-note.docx.",
    });
    const carried = WorkingNotes.from(JSON.parse(persist(notes)));
    expect(carried.hasDone("artifact.create_approval_note", "second-note.docx")).toBe(false);
  });

  it("carries a sandbox execution across the restart too", () => {
    const notes = new WorkingNotes();
    observeToolResult(notes, {
      tool: "sandbox.run_code",
      args: { language: "python", source: "deploy()" },
      text: "done",
    });
    const carried = WorkingNotes.from(JSON.parse(persist(notes)));
    expect(carried.state.completed[0]?.tool).toBe("sandbox.run_code");
    expect(carried.hasDone("execute_code", "python (an execution)")).toBe(true);
  });
});

describe("numbering calculations", () => {
  it("counts from one when nothing has been recorded", () => {
    expect(nextCalculationId([])).toBe("C1");
  });

  it("counts from the highest marker held, not from how many are held", () => {
    // `calculationIds` is capped at 32 and the oldest is shifted out, so once
    // the cap is reached the length stops growing. Numbering by length made
    // every calculation after the 32nd `C33` — which `#addId` then dropped as
    // a duplicate, silently and without moving the `dropped` counter.
    const atCap = Array.from({ length: 32 }, (_, i) => `C${i + 2}`); // C2..C33
    expect(atCap).toHaveLength(32);
    expect(nextCalculationId(atCap)).toBe("C34");
  });

  it("ignores anything that is not a marker", () => {
    expect(nextCalculationId(["", "E4", "not-a-marker", "C7"])).toBe("C8");
  });

  it("does not reuse a marker after notes are restored from state", () => {
    // A resumption replays the ids into a fresh instance. A counter would
    // restart at zero here and hand out markers a previous attempt already
    // used; reading the list cannot.
    const restored = ["C9", "C10", "C11"];
    expect(nextCalculationId(restored)).toBe("C12");
  });
});
