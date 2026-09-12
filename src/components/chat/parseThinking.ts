/**
 * Splits an assistant message into the model's reasoning and its answer.
 *
 * ## Why this exists at all
 *
 * Two kinds of local server exist, and they deliver reasoning differently.
 * One splits it into its own `reasoning_content` field, which reaches the
 * surface live as `model_thinking` events. The other leaves it inline in the
 * content, wrapped in `<think>` tags, and that is what this recovers.
 *
 * ## The bug this replaces
 *
 * The previous version matched only a *closed* pair:
 *
 * ```js
 * content.match(/<think>([\s\S]*?)<\/think>/)
 * ```
 *
 * While the model is still reasoning there is no closing tag, so the match was
 * null, `reasoning` was empty, and the caller fell back to rendering the whole
 * raw buffer — `<think>Let me consider…` — straight into Markdown, which
 * swallows it as an unclosed HTML tag and draws nothing.
 *
 * Measured on NVIDIA-Nemotron3-Nano-4B, which reasons on every turn: answering
 * "hi" cost 67 output tokens for a nine-token reply, and for the twelve seconds
 * it spent on the other fifty-eight the surface showed an empty space. No
 * thinking panel, because the panel only fills once a complete pair exists, and
 * no streaming answer, because there was no answer yet — then the whole reply
 * appeared at once the instant `</think>` arrived.
 *
 * An open block is therefore a first-class state here: everything after the
 * opener is the reasoning *so far*, and there is no answer yet. That is what
 * lets the panel fill live and the answer stream when it starts.
 */

/** Openers a local model may use. Matched case-insensitively. */
const OPEN = /<(think|thinking|thought|reasoning)>/i;

export interface ParsedThinking {
  /** The reasoning so far. Empty when the model has produced none. */
  reasoning: string;
  /**
   * What to render as the answer.
   *
   * Never contains a reasoning tag, open or closed. While a block is open this
   * is the text that preceded it, which is usually empty.
   */
  answer: string;
  /**
   * Whether the reasoning block is still open.
   *
   * The caller shows the panel as live rather than finished, and knows not to
   * treat an empty `answer` as "the model said nothing".
   */
  open: boolean;
  /** Whether a reasoning tag was seen at all, open or closed. */
  sawTag: boolean;
}

export function parseThinking(content: string): ParsedThinking {
  const opener = content.match(OPEN);
  if (!opener || opener.index === undefined) {
    return { reasoning: '', answer: content, open: false, sawTag: false };
  }

  const tag = opener[1];
  const bodyStart = opener.index + opener[0].length;
  // The matching closer, searched for only after the opener — a `</think>` that
  // somehow preceded it is not this block's.
  const closer = new RegExp(`</${tag}\s*>`, 'i');
  const rest = content.slice(bodyStart);
  const closes = rest.match(closer);

  const before = content.slice(0, opener.index);

  if (!closes || closes.index === undefined) {
    // Still reasoning. Everything after the opener is the thought so far.
    return {
      reasoning: rest.trim(),
      answer: before.trim(),
      open: true,
      sawTag: true,
    };
  }

  const reasoning = rest.slice(0, closes.index).trim();
  const after = rest.slice(closes.index + closes[0].length);
  return {
    reasoning,
    answer: `${before}${after}`.trim(),
    open: false,
    sawTag: true,
  };
}
