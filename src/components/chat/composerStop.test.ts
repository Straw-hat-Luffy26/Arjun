/**
 * The composer's Stop button has to stop the turn, and stay reachable.
 *
 * Three regressions, all of which left the button looking like it worked:
 *
 * 1. **It sent the wrong id.** `activeRunId` was the correlation id the
 *    composer minted before the request went out, not the run id the core
 *    minted afterwards. `agent_abort_run` therefore named a run nothing had
 *    heard of and returned "not running".
 * 2. **It discarded the answer.** `await agentService.abort(...)` and nothing
 *    else, so the false above was invisible.
 * 3. **It vanished when you typed.** The button was rendered on
 *    `streaming && !hasContent`, so drafting a follow-up replaced Stop with
 *    Send — and drafting a follow-up is exactly what somebody does while
 *    waiting for a turn they are about to give up on.
 *
 * Asserted against the source rather than a rendered tree, the same way
 * `inspectorStop.test.ts` does it: this repository vendors no DOM, and what is
 * under test is the wiring rather than the pixels.
 */
import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';

/** The file with its comments removed, so prose cannot satisfy an assertion. */
function code(path: string): string {
  return readFileSync(path, 'utf8')
    .split('\n')
    .filter(line => {
      const trimmed = line.trim();
      return (
        !trimmed.startsWith('//') && !trimmed.startsWith('*') && !trimmed.startsWith('/*')
      );
    })
    .join('\n');
}

const COMPOSER = 'src/components/chat/ChatComposer.tsx';

describe('the composer Stop button', () => {
  it('stays on screen while a follow-up draft is being typed', () => {
    const source = code(COMPOSER);
    // The exact regression: Stop rendered only when there was nothing typed.
    expect(source).not.toMatch(/\{streaming\s*&&\s*!hasContent\s*\?/);
    // Stop belongs to the run, so it is shown for the whole of one.
    expect(source).toMatch(/\{streaming\s*&&\s*\(/);
  });

  it('still offers Send alongside it, so a draft can be queued', () => {
    // Both actions remain reachable mid-run: Stop ends the turn, Send queues
    // the draft. Replacing one with the other is what made Stop unreachable.
    expect(code(COMPOSER)).toMatch(/\{\(!streaming \|\| hasContent\)\s*&&\s*\(/);
  });

  it('reads the abort result instead of discarding it', () => {
    const source = code(COMPOSER);
    expect(source).toContain('await agentService.abort(activeRunId)');
    // The field, not a bare boolean: `requested` says the stop reached
    // something, which is a different claim from the turn having ended.
    expect(source).toMatch(/outcome\.requested/);
  });

  /**
   * The heart of it: `agent_abort_run` resolving means the request landed, not
   * that the turn is over. A turn stopped mid-tool finishes the tool first.
   */
  it('does not clear the stopping state merely because the request returned', () => {
    const source = code(COMPOSER);
    const stopBody = source.slice(
      source.indexOf('const stop = useCallback'),
      source.indexOf('const onKeyDown'),
    );
    expect(stopBody.length).toBeGreaterThan(0);
    // A `finally` here would clear the button the instant the call resolved,
    // which is the defect: it reported a stop while the machine still worked.
    expect(stopBody).not.toMatch(/\}\s*finally\s*\{/);
  });

  it('clears the stopping state when the run itself reports it has ended', () => {
    const source = code(COMPOSER);
    // `streaming` going false is the run reaching a terminal state, and it is
    // the only thing that can honestly clear a Stop.
    expect(source).toMatch(/if \(!streaming\) setStopping\(false\);/);
  });

  it('gives up waiting rather than sitting disabled for ever', () => {
    const source = code(COMPOSER);
    expect(source).toContain('STOP_ACKNOWLEDGEMENT_TIMEOUT_MS');
    // And says so, rather than silently re-enabling.
    expect(source).toMatch(/has not reported that it ended/);
  });
});
