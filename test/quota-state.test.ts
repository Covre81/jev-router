import test, { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import {
  parseQuotaError,
  parseDurationMs,
  ModelQuotaState,
  QUOTA_DEFAULT_COOLDOWN_MS,
} from '../src/providers/quota-state.js';

describe('parseDurationMs', () => {
  it('parses composite durations (136h47m52s)', () => {
    const ms = parseDurationMs('136h47m52s');
    assert.equal(ms, (136 * 3600 + 47 * 60 + 52) * 1000);
  });

  it('parses single units (2h, 45m, 30s, 1d)', () => {
    assert.equal(parseDurationMs('2h'), 2 * 3600 * 1000);
    assert.equal(parseDurationMs('45m'), 45 * 60 * 1000);
    assert.equal(parseDurationMs('30s'), 30 * 1000);
    assert.equal(parseDurationMs('1d'), 24 * 3600 * 1000);
    assert.equal(parseDurationMs('1d 2h'), (24 + 2) * 3600 * 1000);
  });

  it('returns null for unparseable strings', () => {
    assert.equal(parseDurationMs('unknown'), null);
    assert.equal(parseDurationMs(''), null);
  });
});

describe('parseQuotaError', () => {
  const baseTime = 1_700_000_000_000;

  it('parses agy live quota error with duration (Resets in 136h47m52s)', () => {
    const text = 'status ERROR: Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 136h47m52s.';
    const res = parseQuotaError(text, baseTime);
    assert.ok(res);
    assert.equal(res.resetAt, baseTime + (136 * 3600 + 47 * 60 + 52) * 1000);
  });

  it('parses short reset durations (Resets in 2h, resets in 45m)', () => {
    const res2h = parseQuotaError('quota reached. Resets in 2h', baseTime);
    assert.ok(res2h);
    assert.equal(res2h.resetAt, baseTime + 2 * 3600 * 1000);

    const res45m = parseQuotaError('Rate limit exceeded: resets in 45m', baseTime);
    assert.ok(res45m);
    assert.equal(res45m.resetAt, baseTime + 45 * 60 * 1000);
  });

  it('parses ISO date reset timestamps', () => {
    const iso = '2026-10-28T04:15:30.000Z';
    const text = `Rate limit exceeded. Resets at ${iso}`;
    const res = parseQuotaError(text, baseTime);
    assert.ok(res);
    assert.equal(res.resetAt, Date.parse(iso));
  });

  it('falls back to default 6h cooldown when no duration or date is found', () => {
    const patterns = [
      'Individual quota reached',
      'Rate limit reached',
      'Resource exhausted',
      'HTTP 429 Too Many Requests',
      'You have hit your usage limit',
      "You've hit your daily limit",
      'Please upgrade your plan to increase limits',
      'You are out of credits',
    ];
    for (const pat of patterns) {
      const res = parseQuotaError(pat, baseTime);
      assert.ok(res, `Expected match for: ${pat}`);
      assert.equal(res.resetAt, baseTime + QUOTA_DEFAULT_COOLDOWN_MS);
    }
  });

  it('returns null for non-quota errors', () => {
    assert.equal(parseQuotaError('SyntaxError: unexpected token'), null);
    assert.equal(parseQuotaError('Connection refused: ECONNREFUSED'), null);
    assert.equal(parseQuotaError('timeout waiting for process'), null);
  });
});

describe('ModelQuotaState', () => {
  it('tracks model exhaustion and auto-restores with fake clock', () => {
    let now = 1_000_000;
    const quotaState = new ModelQuotaState(() => now);

    assert.equal(quotaState.isAvailable('gemini-4-argon-low'), true);

    const resetAt = now + 10_000;
    quotaState.markExhausted('gemini-4-argon-low', resetAt, 'Individual quota reached');

    assert.equal(quotaState.isAvailable('gemini-4-argon-low'), false);
    assert.equal(quotaState.isAvailable('gemini-3.8-flash-high'), true);

    // Advance clock past resetAt
    now = resetAt + 1;
    assert.equal(quotaState.isAvailable('gemini-4-argon-low'), true);
  });

  it('logs "model quota exhausted" once per exhaustion and "model quota restored" once', () => {
    let now = 1_000_000;
    const warns: any[] = [];
    const infos: any[] = [];
    const logger = {
      warn: (obj: any, msg?: string) => warns.push({ obj, msg }),
      info: (obj: any, msg?: string) => infos.push({ obj, msg }),
    };

    const quotaState = new ModelQuotaState(() => now, logger);

    const resetAt = now + 5000;
    quotaState.markExhausted('gemini-4-argon-low', resetAt, 'quota hit');
    quotaState.markExhausted('gemini-4-argon-low', resetAt, 'quota hit again next turn');
    quotaState.markExhausted('gemini-4-argon-low', resetAt, 'quota hit 3rd turn');

    // Should only have logged once
    assert.equal(warns.length, 1);
    assert.equal(warns[0].msg, 'model quota exhausted');
    assert.equal(warns[0].obj.model, 'gemini-4-argon-low');

    // Auto-restore
    now = resetAt + 1;
    assert.equal(quotaState.isAvailable('gemini-4-argon-low'), true);
    assert.equal(infos.length, 1);
    assert.equal(infos[0].msg, 'model quota restored');

    // Querying again when already restored does not log restore again
    assert.equal(quotaState.isAvailable('gemini-4-argon-low'), true);
    assert.equal(infos.length, 1);

    // Re-exhausting logs once again
    const nextReset = now + 5000;
    quotaState.markExhausted('gemini-4-argon-low', nextReset, 'quota hit again');
    assert.equal(warns.length, 2);
  });

  it('snapshot returns status for known and dynamic models', () => {
    let now = 1_000_000;
    const quotaState = new ModelQuotaState(() => now);

    const resetAt = now + 10_000;
    quotaState.markExhausted('gemini-4-argon-low', resetAt, 'Individual quota reached');

    const snap = quotaState.snapshot(['gemini-3.8-flash-high', 'gemini-4-argon-low', 'cursor']);
    assert.deepEqual(snap['gemini-3.8-flash-high'], { state: 'ok' });
    assert.deepEqual(snap['cursor'], { state: 'ok' });
    assert.equal(snap['gemini-4-argon-low']?.state, 'exhausted');
    assert.equal(snap['gemini-4-argon-low']?.until, new Date(resetAt).toISOString());
    assert.equal(snap['gemini-4-argon-low']?.reason, 'Individual quota reached');
  });
});
