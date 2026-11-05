import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { reactionOf, type MessagesBody } from '../src/routing/messages-body.js';
import { openTelemetryDb } from '../src/telemetry/db.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';
import type { NewRouterLog } from '../src/telemetry/schema.js';
import { computeStats, renderStats } from '../src/telemetry/stats.js';

const body = (...messages: object[]) => ({ model: 'm', messages }) as unknown as MessagesBody;
const assistant = { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Bash', input: {} }] };
const toolResult = (isError?: boolean) => ({
  role: 'user',
  content: [{ type: 'tool_result', tool_use_id: 't1', content: 'x', ...(isError === undefined ? {} : { is_error: isError }) }],
});

describe('reactionOf: what the latest user message says about the previous assistant turn', () => {
  it('counts tool_result blocks flagged is_error', () => {
    const r = reactionOf(body({ role: 'user', content: 'oi' }, assistant, toolResult(true)));
    assert.deepEqual(r, { toolErrors: 1, interrupted: false });
  });

  it('sums every flagged tool_result in the same message', () => {
    const two = { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'a', is_error: true }, { type: 'tool_result', tool_use_id: 'b', is_error: true }] };
    assert.equal(reactionOf(body({ role: 'user', content: 'oi' }, assistant, two)).toolErrors, 2);
  });

  it('detects the marker in plain-string content and tolerates surrounding whitespace', () => {
    assert.equal(reactionOf(body({ role: 'user', content: 'oi' }, assistant, { role: 'user', content: '  [Request interrupted by user]\n' })).interrupted, true);
  });

  it('ignores successful tool results', () => {
    assert.deepEqual(reactionOf(body({ role: 'user', content: 'oi' }, assistant, toolResult(false))), { toolErrors: 0, interrupted: false });
    assert.deepEqual(reactionOf(body({ role: 'user', content: 'oi' }, assistant, toolResult())), { toolErrors: 0, interrupted: false });
  });

  it('detects both Claude Code interruption markers', () => {
    for (const text of ['[Request interrupted by user]', '[Request interrupted by user for tool use]']) {
      const r = reactionOf(body({ role: 'user', content: 'oi' }, assistant, { role: 'user', content: [{ type: 'text', text }] }));
      assert.equal(r.interrupted, true, text);
    }
  });

  it('does not flag a human who merely quotes the marker mid-sentence', () => {
    const r = reactionOf(body({ role: 'user', content: 'o log mostra [Request interrupted by user] de novo' }));
    assert.equal(r.interrupted, false);
  });

  it('is empty on a fresh conversation or when the last message is not from the user', () => {
    assert.deepEqual(reactionOf(body({ role: 'user', content: 'oi' })), { toolErrors: 0, interrupted: false });
    assert.deepEqual(reactionOf(body({ role: 'user', content: 'oi' }, assistant)), { toolErrors: 0, interrupted: false });
  });
});

const row = (over: Partial<NewRouterLog>): NewRouterLog => ({
  createdAt: new Date('2026-10-01T12:00:00Z'),
  sessionId: 's1',
  finalProvider: 'anthropic',
  routeReason: 'classified',
  outcome: 'ok',
  httpStatus: 200,
  latencyMs: 100,
  ...over,
});

describe('stats: reaction rates by the provider that served the previous turn', () => {
  // The reaction lives on the NEXT row of the session, so the join is by LAG over the session.
  it('attributes prev_tool_errors / prev_interrupted to the provider of the preceding row', () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    // session A: gemini turn, then the user hits a tool error and interrupts on the next one
    sink.record(row({ sessionId: 'A', finalProvider: 'gemini' }));
    sink.record(row({ sessionId: 'A', finalProvider: 'anthropic', prevToolErrors: 2, prevInterrupted: false }));
    sink.record(row({ sessionId: 'A', finalProvider: 'anthropic', prevToolErrors: 0, prevInterrupted: true }));
    // session B interleaved: anthropic turn, clean follow-up
    sink.record(row({ sessionId: 'B', finalProvider: 'anthropic' }));
    sink.record(row({ sessionId: 'B', finalProvider: 'anthropic', prevToolErrors: 0, prevInterrupted: false }));
    sink.flush();

    const s = computeStats(db);
    const by = Object.fromEntries(s.reactions.map((r) => [r.provider, r]));
    // 2 failing tool_results in one follow-up are ONE turn with a tool error
    assert.deepEqual(by['gemini'], { provider: 'gemini', followups: 1, toolErrors: 1, interrupted: 0 });
    assert.deepEqual(by['anthropic'], { provider: 'anthropic', followups: 2, toolErrors: 0, interrupted: 1 });
    assert.match(renderStats(s), /Follow-up turns after a tool error or interruption/);
  });

  it('leaves rows written before the migration (NULL) out of the denominator', () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    sink.record(row({ sessionId: 'A', finalProvider: 'gemini' }));
    sink.record(row({ sessionId: 'A', finalProvider: 'anthropic' })); // no reaction recorded
    sink.flush();
    assert.deepEqual(computeStats(db).reactions, []);
  });
});
