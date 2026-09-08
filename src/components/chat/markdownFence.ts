/**
 * Recognising the delimiters of a fenced code block.
 *
 * ## Why this is its own module
 *
 * It was three characters of a regular expression inside `Markdown.tsx`, and
 * those three characters decided whether a C++ answer was drawn as code or as
 * prose. Pulled out here so the rule can be tested without a DOM, which is the
 * same reason `mermaidParse` and `runProgress` are separate modules.
 *
 * ## The defect this fixes
 *
 * The opening fence was matched with `^` + three backticks + `(\w*)\s*$`. `\w`
 * is `[A-Za-z0-9_]` — it does not include `+`, `#` or `-`. So a fence tagged
 * `cpp` was recognised and a fence tagged `c++` was not.
 *
 * A model asked for C++ writes either, and roughly as often. When it wrote the
 * second, three things went wrong at once and none of them looked like a
 * parsing bug:
 *
 * 1. The opening fence was left as an ordinary line, so the source below it was
 *    rendered as paragraphs — indentation collapsed, no highlighting, no copy
 *    button.
 * 2. The *closing* delimiter still matched, because with no language after it
 *    the pattern was satisfied. So the close was read as an **open**, and every
 *    word after the code block was swallowed into a block that never ended.
 * 3. While the answer was still streaming, all of that was happening to a
 *    partial document, so what the reader watched arrive was prose where code
 *    should have been.
 *
 * The same hole swallowed `c#`, `f#` and `objective-c`.
 *
 * ## What a language tag may contain
 *
 * Letters and digits, plus the punctuation that appears in the names languages
 * actually have: `+` (`c++`), `#` (`f#`), `-` (`objective-c`), `.` (`asp.net`)
 * and `_`, which `\w` already allowed. Deliberately not "any non-whitespace": a
 * line of prose ending in backticks should stay prose, and the narrower set
 * keeps a fence a fence.
 */

/**
 * An opening fence's language, or `null` when the line is not an opening fence.
 *
 * Returns `undefined` for a bare fence — one with no language, which is legal
 * and common. The two absent cases are distinguished on purpose: `null` means
 * "not a fence at all" and `undefined` means "a fence that named no language".
 * A caller that conflated them would treat every closing fence as the start of
 * a new block, which is precisely the second failure described above.
 */
export function openingFenceLanguage(line: string): string | undefined | null {
  const match = line.match(/^```([A-Za-z0-9+#._-]*)\s*$/);
  if (!match) return null;
  return match[1] === '' ? undefined : match[1];
}

/**
 * Whether this line closes a fenced block.
 *
 * A closing fence carries no language. Asked separately from
 * [`openingFenceLanguage`] rather than derived from it, because the same line
 * answers both questions differently — a bare fence is an open or a close
 * depending on whether a block is already open, and only the caller knows.
 */
export function isClosingFence(line: string): boolean {
  return /^```\s*$/.test(line);
}
