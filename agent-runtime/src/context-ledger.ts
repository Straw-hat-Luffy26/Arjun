/**
 * What is actually in the context window, section by section.
 *
 * ## The failure this prevents
 *
 * `RunCompactor` knows one number: how many tokens the whole projected context
 * came to. When that number is too large it summarises the oldest half. That
 * works, and it tells an operator nothing about *why* the window filled — which
 * is the question they actually have when a run compacts four times in twenty
 * turns. Was it the tool schemas? A skill body nobody uses? One 40-page tool
 * result pasted whole?
 *
 * Without an answer the only lever is "summarise sooner", which degrades the
 * run. With one, the lever is usually "stop pasting the document" — which does
 * not degrade anything, because the document is still retrievable by reference.
 *
 * ## Why the sections are these sections
 *
 * They are the things that can independently grow, and each has a different
 * remedy:
 *
 * - `system` — the base prompt. Fixed; growth here is a code change.
 * - `skill` — guidance loaded for this task. Progressive disclosure: a skill
 *   body belongs in context only while the step that needs it is live.
 * - `toolSchema` — the tool definitions. Grows with the *catalogue*, not the
 *   run, so a large number here means too many tools are offered at once.
 * - `evidence` — retrieved passages. Should be references plus the regions
 *   actually read, never whole documents.
 * - `notes` — the bounded working notes. If this is not small, the bound is
 *   wrong.
 * - `transcript` — the live message tail.
 * - `compaction` — the summary standing in for older history.
 * - `reserve` — held back for the model's own output and for the summarisation
 *   request. Not occupied; *committed*, which is the same thing for the purpose
 *   of deciding whether the next turn fits.
 *
 * ## What this is not
 *
 * Not a budget enforcer. It counts and reports; `RunCompactor` decides. Keeping
 * measurement apart from policy means a wrong count produces a wrong *number on
 * a screen* rather than a run that compacts when it should not.
 */

import { estimateContextTokens, estimateTokens, type AgentMessage } from "@openclaw/agent-core";

import {
  type ContextEntity,
  reconcileSections,
  withRemainders,
} from "./context-entities.js";

/** The independently-growing parts of a context window. */
export const CONTEXT_SECTIONS = [
  "system",
  "skill",
  "toolSchema",
  "evidence",
  "notes",
  "transcript",
  "compaction",
  "reserve",
] as const;

export type ContextSection = (typeof CONTEXT_SECTIONS)[number];

/**
 * What a section's un-itemised balance is called on screen.
 *
 * Phrased as the thing it is rather than as "other", because "other: 4,200
 * tokens" tells a reader nothing they can act on, and "the conversation so far"
 * tells them where to look.
 */
const SECTION_REMAINDERS: Record<ContextSection, string> = {
  system: "System prompt",
  skill: "Skill guidance",
  toolSchema: "Tool definitions",
  evidence: "Other retrieved passages",
  notes: "Working notes",
  transcript: "Conversation so far",
  compaction: "Summary of earlier turns",
  reserve: "Reserved for the reply",
};

/** Loose text as the estimator wants to see it. */
function asTextMessage(text: string): AgentMessage {
  return { role: "user", content: [{ type: "text", text }], timestamp: 0 } as AgentMessage;
}

/** One reading of the ledger. Plain data, so it crosses the wire unchanged. */
export interface ContextLedgerSnapshot {
  /** Token count per section. Every section is present, zero included, so a
   * reader never has to distinguish "absent" from "empty". */
  sections: Record<ContextSection, number>;
  /** Everything except `reserve` — what is really occupied right now. */
  occupied: number;
  /** Occupied plus reserve. What the next turn has to fit inside. */
  committed: number;
  /** The model's window, as the runtime was told it. Zero when unknown. */
  window: number;
  /** `window - committed`. Negative means the next turn does not fit. */
  headroom: number;
  /** Times this run has compacted so far. */
  compactions: number;
  /**
   * The itemisation under the sections, completed with remainder rows so that
   * it always sums to them. See `context-entities.ts`.
   */
  entities: ContextEntity[];
  /**
   * Every model call this run has made, each with what was predicted and what
   * was counted.
   *
   * Kept in full rather than reduced to a running average: the question a
   * person asks of these is "when did the estimate start drifting", and a mean
   * cannot answer it.
   */
  reconciliations: TurnReconciliation[];
  /**
   * Sections whose itemised rows disagree with their totals.
   *
   * Empty in normal operation. Non-empty means the breakdown below the bar is
   * not an explanation of the bar, and the screen says so rather than drawing
   * rows that do not add up.
   */
  itemisationErrors: { section: string; fromEntities: number; fromSection: number }[];
}

/** One model call's prediction measured against what it actually cost. */
export interface TurnReconciliation {
  /** Which call of this run, 1-based. */
  turn: number;
  at: string;
  /** What the ledger predicted the input would come to. */
  estimatedIn: number;
  /**
   * What the provider reported. `null` when it reported nothing — which is
   * left as null rather than back-filled from the estimate, so that a screen
   * can tell "we checked and it matched" from "nobody checked".
   */
  actualIn: number | null;
  actualOut: number | null;
  /** `actualIn / estimatedIn`, or `null` when there is nothing to divide. */
  driftRatio: number | null;
}

/**
 * Accumulates token counts for one run.
 *
 * Mutable and reused across turns rather than rebuilt: the fixed sections
 * (system, tool schemas) are measured once, and re-measuring them every turn
 * would spend real time counting characters that did not change.
 */
export class ContextLedger {
  readonly #sections: Record<ContextSection, number>;
  #window: number;
  #compactions = 0;
  /** Keyed by entity id, so a document re-registered on a later turn updates
   *  its row instead of adding a second one for the same file. */
  readonly #entities = new Map<string, ContextEntity>();
  readonly #reconciliations: TurnReconciliation[] = [];

  constructor(window = 0) {
    this.#window = Math.max(0, Math.floor(window));
    this.#sections = Object.fromEntries(
      CONTEXT_SECTIONS.map((section) => [section, 0]),
    ) as Record<ContextSection, number>;
  }

  /** Replaces a section's count. Used for the parts measured once. */
  set(section: ContextSection, tokens: number): void {
    this.#sections[section] = Math.max(0, Math.floor(tokens));
  }

  /**
   * Measures text and records it as a section's whole content.
   *
   * Wrapped as a message rather than counted by character, so a section made of
   * loose text and one made of messages are measured by the same estimator. Two
   * estimators would make the ledger's own total disagree with the number
   * compaction decides on, and only one of them can be right.
   */
  setText(section: ContextSection, text: string): void {
    this.set(
      section,
      text ? estimateContextTokens([asTextMessage(text)]).tokens : 0,
    );
  }

  /**
   * Measures messages and records them as a section's whole content.
   *
   * Uses the harness estimator rather than a character count, because an image
   * block counts as a flat block of tokens there. ARJUN feeds rendered PDF
   * pages to vision models, and counting a page image by its characters reads
   * it as ~0 tokens — which is how a ledger ends up claiming plenty of headroom
   * on the turn that overflows.
   *
   * Summed **per message** rather than measured with `estimateContextTokens`,
   * and the difference is not cosmetic. That function short-circuits to the
   * provider's own reported usage as soon as the list contains an assistant
   * message carrying one — a sound reading of a whole context and a wrong one
   * for a *part* of it. Two sections measured that way do not sum to the context
   * they were cut from: whichever half caught the usage-bearing message reports
   * the entire conversation, and the other reports only itself. A ledger whose
   * parts exceed its whole cannot be used to decide anything, which is the only
   * reason to keep one.
   */
  setMessages(section: ContextSection, messages: AgentMessage[]): void {
    this.set(
      section,
      messages.reduce((total, message) => total + estimateTokens(message), 0),
    );
  }

  /** Adds to a section. For things that arrive one at a time, like passages. */
  add(section: ContextSection, tokens: number): void {
    this.#sections[section] = Math.max(0, this.#sections[section] + Math.floor(tokens));
  }

  get(section: ContextSection): number {
    return this.#sections[section];
  }

  setWindow(window: number): void {
    this.#window = Math.max(0, Math.floor(window));
  }

  countCompaction(): void {
    this.#compactions += 1;
  }

  get compactions(): number {
    return this.#compactions;
  }

  /**
   * Registers or updates one itemised entity.
   *
   * Keyed by id, so the same document arriving on a second turn replaces its
   * row rather than doubling it — the reason ids are content hashes for files.
   * A pin already set by the person survives an update that does not mention
   * one, because the update comes from the runtime and the pin came from a
   * human, and the human's instruction is not the runtime's to discard.
   */
  upsertEntity(entity: ContextEntity): void {
    const existing = this.#entities.get(entity.id);
    this.#entities.set(entity.id, {
      ...entity,
      pinned: entity.pinned || (existing?.pinned ?? false),
    });
  }

  /** Protects an entity from eviction, or releases it. */
  setPinned(id: string, pinned: boolean): void {
    const existing = this.#entities.get(id);
    if (existing) this.#entities.set(id, { ...existing, pinned });
  }

  /**
   * Makes the pinned set exactly `ids`, releasing everything else.
   *
   * ## Why this replaces rather than adds
   *
   * The pinned set arrives whole on every `run.note`, because unpinning matters
   * as much as pinning. A caller that looped `setPinned(id, true)` over the
   * arriving list would therefore never release anything: an id the person had
   * just *removed* would keep its row drawn as protected for the rest of the
   * run, while the compactor — reading the same list — correctly stopped
   * protecting it. The meter would then be promising something nothing was
   * doing, which is the precise failure the pin was wired up to remove.
   *
   * Case-insensitive on the way in, matching `pruneStaleToolResults`, so a pin
   * cannot be honoured by one and dropped by the other over how an id happened
   * to be spelled.
   */
  applyPins(ids: readonly string[]): void {
    const wanted = new Set(ids.map((id) => id.toUpperCase()));
    for (const [id, entity] of this.#entities) {
      const pinned = wanted.has(id.toUpperCase());
      if (entity.pinned !== pinned) this.#entities.set(id, { ...entity, pinned });
    }
  }

  /**
   * Records what one model call was predicted to cost and what it actually did.
   *
   * Called on every model turn — see `token_reconciliation.rs` for why that
   * cadence and not a periodic one.
   */
  reconcile(input: {
    estimatedIn: number;
    actualIn: number | null;
    actualOut: number | null;
    at?: string;
  }): TurnReconciliation {
    const record: TurnReconciliation = {
      turn: this.#reconciliations.length + 1,
      at: input.at ?? new Date().toISOString(),
      estimatedIn: input.estimatedIn,
      actualIn: input.actualIn,
      actualOut: input.actualOut,
      driftRatio:
        input.actualIn !== null && input.estimatedIn > 0
          ? input.actualIn / input.estimatedIn
          : null,
    };
    this.#reconciliations.push(record);
    return record;
  }

  /**
   * Corrects the transcript section to what the provider actually charged.
   *
   * ## Why the correction lands on `transcript` and nowhere else
   *
   * The provider reports one number for the whole input. It does not say how
   * that number divides between the system prompt, the tool schemas and the
   * conversation — so spreading the correction across sections would mean
   * inventing a division nobody measured, and every section would end up
   * slightly false rather than one being right.
   *
   * The sections that do not change during a run — system, tool schemas — were
   * measured once from text this process holds, and are the ones least likely
   * to be wrong. The transcript is the part that grows, the part the estimator
   * has the least purchase on, and the only part whose size this side infers
   * rather than reads. So the difference is booked there, which is both the
   * likeliest home for it and the one place a reader can interpret.
   *
   * Applied only when the provider actually reported a figure. A `null` leaves
   * every section exactly as estimated, and the snapshot's `reconciliations`
   * still say that nothing was confirmed.
   */
  applyMeasuredInput(actualIn: number | null): void {
    if (actualIn === null) return;
    const fixed =
      this.#sections.system +
      this.#sections.skill +
      this.#sections.toolSchema +
      this.#sections.evidence +
      this.#sections.notes +
      this.#sections.compaction;
    // Never negative: a provider total below the fixed sections means the
    // sections are over-counted, and writing a negative transcript would make
    // the ledger sum to something impossible. Zero is the floor, and the
    // discrepancy stays visible as drift on the reconciliation record.
    this.#sections.transcript = Math.max(0, actualIn - fixed);
  }

  snapshot(): ContextLedgerSnapshot {
    const sections = { ...this.#sections };
    const occupied = CONTEXT_SECTIONS.filter((section) => section !== "reserve").reduce(
      (total, section) => total + sections[section],
      0,
    );
    const committed = occupied + sections.reserve;
    // Completed with remainders so the rows always sum to the sections above
    // them. `reserve` is excluded from the itemisation: it is committed by
    // policy rather than held by anything, so a row for it would invite an
    // operator to reclaim the one thing that must not be reclaimed.
    const itemisable = Object.fromEntries(
      CONTEXT_SECTIONS.filter((section) => section !== "reserve").map((section) => [
        section,
        sections[section],
      ]),
    );
    const entities = withRemainders([...this.#entities.values()], itemisable, SECTION_REMAINDERS);
    return {
      sections,
      occupied,
      committed,
      window: this.#window,
      // An unknown window has no headroom to report. Zero would read as "full",
      // which is a claim this does not have the information to make; the caller
      // checks `window` before believing it.
      headroom: this.#window > 0 ? this.#window - committed : 0,
      compactions: this.#compactions,
      entities,
      reconciliations: [...this.#reconciliations],
      itemisationErrors: reconcileSections(entities, itemisable),
    };
  }

  /**
   * Whether the next turn fits.
   *
   * False when the window is unknown is deliberate — an unknown window is not
   * evidence of room, and the caller that wants to proceed anyway can say so by
   * checking `window` itself.
   */
  fits(): boolean {
    const { window, committed } = this.snapshot();
    return window > 0 && committed <= window;
  }

  /**
   * The sections that grew most, largest first.
   *
   * The thing an operator reads. Naming the top two is usually the whole
   * diagnosis, and a table of eight numbers is not.
   *
   * `reserve` is excluded. It is set by policy from the window size rather than
   * grown by anything the run did, so ranking it here would regularly name it
   * as the cause on a small window — a diagnosis whose only remedy is to reserve
   * less, which is the one change guaranteed to make the run fail.
   */
  largest(count = 3): { section: ContextSection; tokens: number }[] {
    return CONTEXT_SECTIONS.filter((section) => section !== "reserve")
      .map((section) => ({ section, tokens: this.#sections[section] }))
      .filter((entry) => entry.tokens > 0)
      .sort((a, b) => b.tokens - a.tokens)
      .slice(0, count);
  }
}
