import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { featuresOf, LinUcb, replay, type ReplayRow } from '../src/routing/bandit.js';
import { loadReplayRows } from '../scripts/bandit-replay.js';
import { openTelemetryDb } from '../src/telemetry/db.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';
import type { NewRouterLog } from '../src/telemetry/schema.js';

describe('featuresOf', () => {
  it('maps a JEV decision to a fixed-size vector and tolerates fields older rows lack', () => {
    const full = featuresOf({ simple: 0.9, standard: 0.07, structural: 0.03, risk: 0.1, textOnly: 0.95 }, 10_000);
    const old = featuresOf({ simple: 0.9, standard: 0.07, structural: 0.03 }, null);
    assert.equal(full.length, old.length);
    assert.equal(full[0], 1); // bias
    assert.deepEqual(full.slice(1, 6), [0.9, 0.07, 0.03, 0.1, 0.95]);
    assert.ok(Math.abs(full[6]! - Math.log10(10_001) / 6) < 1e-9);
    assert.ok(old.every(Number.isFinite));
  });
});

describe('LinUcb', () => {
  it('learns a linear target and gets more certain with data', () => {
    const m = new LinUcb(3, { alpha: 1, lambda: 1 });
    const hi = [1, 1, 0];
    const lo = [1, 0, 0];
    const before = m.score(hi).bound;
    for (let i = 0; i < 200; i++) {
      m.update(hi, 1);
      m.update(lo, 0);
    }
    assert.ok(m.score(hi).mean > 0.9, `mean(hi)=${m.score(hi).mean}`);
    assert.ok(m.score(lo).mean < 0.1, `mean(lo)=${m.score(lo).mean}`);
    assert.ok(m.score(hi).bound < before / 5);
  });
});

describe('replay', () => {
  // structural-heavy turns always go wrong on the cheap route, simple ones never do
  const synthetic = (n: number): ReplayRow[] =>
    Array.from({ length: n }, (_, i) => {
      const hard = i % 4 === 0;
      return { x: [1, hard ? 0.1 : 0.9, hard ? 0.8 : 0.05], bad: hard };
    });

  it('never vetoes during warm-up', () => {
    const r = replay(synthetic(20), { minRows: 30, maxBad: 0.2, alpha: 0.5 });
    assert.equal(r.vetoed, 0);
  });

  it('after warm-up vetoes the turns that go wrong and keeps the ones that do not', () => {
    const r = replay(synthetic(400), { minRows: 30, maxBad: 0.2, alpha: 0.5 });
    assert.ok(r.vetoed > 50, `vetoed=${r.vetoed}`);
    assert.ok(r.badAmongVetoed / r.vetoed > 0.9);
    assert.ok(r.badAmongKept / (r.n - r.vetoed) < r.baseRate / 2);
  });
});

describe('loadReplayRows', () => {
  const jev = { simple: 0.95, standard: 0.03, structural: 0.02, pSimple: 0.95, pComplex: 0.02, classifierMs: 1 };
  const row = (over: Partial<NewRouterLog>): NewRouterLog => ({
    createdAt: new Date('2026-10-01T12:00:00Z'),
    sessionId: 'A',
    finalProvider: 'gemini',
    routeReason: 'gemini:text-only',
    outcome: 'ok',
    latencyMs: 1,
    jevDecision: jev,
    ...over,
  });

  it('keeps scored cheap turns, drops sticky/Claude ones, and labels bad from the NEXT row of the session', () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    sink.record(row({})); // 0: scored cheap; the next turn hits a tool error -> bad
    sink.record(row({ finalProvider: 'anthropic', jevDecision: null, prevToolErrors: 1, prevInterrupted: false })); // Claude row: not a replay row
    sink.record(row({ jevDecision: null })); // sticky cheap: no score, skipped
    sink.record(row({ inspectionMiss: true })); // 1: bad on its own
    sink.record(row({ sessionId: 'B' })); // 2: clean
    sink.record(row({ sessionId: 'B', jevDecision: null, prevToolErrors: 0, prevInterrupted: false }));
    sink.record(row({ sessionId: 'C' })); // 3: the user interrupted right after -> bad
    sink.record(row({ sessionId: 'C', jevDecision: null, prevToolErrors: 0, prevInterrupted: true }));
    sink.record(row({ sessionId: 'D', outcome: 'stream_error' })); // 4: did not finish ok -> bad
    sink.flush();

    const rows = loadReplayRows(db);
    assert.deepEqual(rows.map((r) => r.bad), [true, true, false, true, true]);
  });
});
