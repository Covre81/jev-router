import assert from 'node:assert/strict';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabaseSync } from 'node:sqlite';
import { describe, it } from 'node:test';
import { inTransaction, openTelemetryDb } from '../src/telemetry/db.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';
import { MIGRATIONS, recordQuota, type NewRouterLog } from '../src/telemetry/schema.js';
import { pricingFromEnv } from '../src/telemetry/pricing.js';
import { computeProfileStats, computeStats, parseSince, renderProfileStats, renderStats } from '../src/telemetry/stats.js';

const row = (over: Partial<NewRouterLog>): NewRouterLog => ({
  createdAt: new Date('2026-10-01T12:00:00Z'),
  sessionId: 's:main',
  finalProvider: 'anthropic',
  routeReason: 'classified',
  outcome: 'ok',
  httpStatus: 200,
  latencyMs: 100,
  ...over,
});

describe('telemetry → stats', () => {
  it('aggregates diversion, fallbacks, repeats and estimated savings', async () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    const jev = { simple: 0.9, standard: 0.07, structural: 0.03, pSimple: 0.9, pComplex: 0.03, classifierMs: 80, tokensIn: 453, tokensOut: 20 };

    sink.record(row({ finalProvider: 'openai', jevDecision: jev, humanPromptHash: 'h1', tokensIn: 1000, tokensOut: 200 }));
    sink.record(row({ finalProvider: 'openai', humanPromptHash: 'h1', tokensIn: 500, tokensOut: 50 }));
    sink.record(row({ finalProvider: 'openai', outcome: 'stream_error', tokensIn: 999, tokensOut: 9 }));
    sink.record(row({ routeReason: 'failover:cheap-unavailable', fallbackTriggered: true, tokensIn: 3000, tokensOut: 100, cacheReadTokens: 2000 }));
    sink.record(row({ routeReason: 'failover:gemini-unavailable', fallbackTriggered: true, tokensIn: 3000, tokensOut: 100, cacheReadTokens: 2000 }));
    sink.record(row({ createdAt: new Date('2026-09-01T00:00:00Z'), tokensIn: 7, tokensOut: 7 }));
    sink.flush();

    const all = computeStats(db);
    assert.equal(all.total, 6);
    assert.equal(all.byProvider.openai.requests, 3);
    assert.equal(all.byProvider.openai.ok, 2);
    assert.equal(all.byProvider.anthropic.cacheReadTokens, 4000);
    assert.equal(all.fallbacks, 1);
    assert.equal(all.geminiFallbacks, 1);
    assert.equal(all.cheapStreamErrors, 1);
    assert.equal(all.classified, 1);
    assert.equal(all.repeatedPrompts, 1);
    assert.equal(all.repeatedSends, 1);
    assert.equal(all.estimatedTokensSaved, 1750, 'only cheap requests served OK count');

    const priced = computeStats(db, { pricing: pricingFromEnv({ JEV_PRICE_INPUT_PER_MTOK: '1', JEV_PRICE_OUTPUT_PER_MTOK: '10', CHEAP_PRICE_INPUT_PER_MTOK: '100', CHEAP_PRICE_OUTPUT_PER_MTOK: '1000' }) });
    assert.ok(Math.abs(priced.cost.jevUsd - (453 * 1 + 20 * 10) / 1_000_000) < 1e-12, 'JEV tokens come from jev_decision');
    assert.ok(Math.abs(priced.cost.failedCheapUsd - (999 * 100 + 9 * 1000) / 1_000_000) < 1e-12, 'failed cheap ignores gemini failover');
    assert.match(renderStats(priced), /Classifier \(JEV\)/);
    assert.match(renderStats(all), /JEV priced at \$0/);

    const recent = computeStats(db, { since: new Date('2026-09-15T00:00:00Z') });
    assert.equal(recent.total, 5);

    const text = renderStats(all);
    assert.match(text, /Requests\s+6/);
    assert.match(text, /Estimated Anthropic tokens saved\s+1,750/);
    assert.match(text, /NET \((PROFIT|LOSS)\)/);
    assert.match(text, /Cheap provider priced at \$0/);
    await sink.close();
  });

  it('groups main-loop calls per session by served profile, priced by the served model', async () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    const sonnetHigh = { model: 'claude-sonnet-5-5', requestedModel: 'claude-opus-5-5', effort: 'high', requestClass: 'main', tokensIn: 1000, cacheReadTokens: 800, tokensOut: 10 };
    const opusMedium = { model: 'claude-opus-5-5', requestedModel: 'claude-opus-5-5', effort: 'medium', requestClass: 'main', tokensIn: 1000, cacheReadTokens: 500, tokensOut: 10 };
    for (let i = 0; i < 3; i++) sink.record(row({ ...sonnetHigh, sessionId: 'A:main' }));
    sink.record(row({ ...opusMedium, sessionId: 'B:main' }));
    sink.record(row({ ...opusMedium, sessionId: 'C:main' }));
    sink.record(row({ ...opusMedium, sessionId: 'C:main' }));
    sink.record(row({ ...opusMedium, sessionId: 'C:sub', requestClass: 'subagent' }));
    sink.record(row({ ...sonnetHigh, sessionId: 'A:main', finalProvider: 'openai' }));
    sink.flush();

    const stats = computeProfileStats(db, pricingFromEnv({}));
    const opus = stats.find((s) => s.model === 'claude-opus-5-5')!;
    const sonnet = stats.find((s) => s.model === 'claude-sonnet-5-5')!;
    assert.equal(stats.length, 2);
    assert.deepEqual([opus.effort, opus.sessions, opus.calls, opus.medianCallsPerSession, opus.meanCallsPerSession], ['medium', 2, 3, 2, 1.5]);
    assert.deepEqual([sonnet.effort, sonnet.sessions, sonnet.calls, sonnet.cacheHitRate], ['high', 1, 3, 0.8]);
    const sonnetCall = (200 * 2 + 800 * 0.1 + 10 * 10) / 1_000_000;
    assert.ok(Math.abs(sonnet.usdPerSession - 3 * sonnetCall) < 1e-12, 'Sonnet rewrite priced as Sonnet, not the requested Opus');
    assert.match(renderProfileStats(stats), /claude-sonnet-5-5\s+high/);
    await sink.close();
  });

  it('counts cheap answers that ignored the tools they were offered (inspection miss)', async () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    sink.record(row({ finalProvider: 'openai', toolsOffered: 20, toolCalls: 0, inspectionMiss: true, humanPromptHash: 'h1' }));
    sink.record(row({ finalProvider: 'openai', toolsOffered: 20, toolCalls: 2, inspectionMiss: false, humanPromptHash: 'h2' }));
    sink.record(row({ finalProvider: 'anthropic', toolsOffered: 20, toolCalls: 0, humanPromptHash: 'h3' }));
    sink.flush();

    const stats = computeStats(db);
    assert.equal(stats.inspectionMisses, 1);
    assert.match(renderStats(stats), /Cheap answers without tool use \(inspection miss\)\s+1/);
    const stored = db.prepare('SELECT tools_offered, tool_calls, inspection_miss FROM router_logs ORDER BY id').all();
    assert.deepEqual(
      stored.map((r) => ({ ...r })),
      [
        { tools_offered: 20, tool_calls: 0, inspection_miss: 1 },
        { tools_offered: 20, tool_calls: 2, inspection_miss: 0 },
        { tools_offered: 20, tool_calls: 0, inspection_miss: null },
      ],
    );
    await sink.close();
  });

  it('splits the cheap route by model (trivial 20B vs standard Gemma)', async () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    sink.record(row({ finalProvider: 'openai', model: 'gpt-oss:20b-cloud' }));
    sink.record(row({ finalProvider: 'openai', model: 'gemma4:31b-cloud' }));
    sink.record(row({ finalProvider: 'openai', model: 'gemma4:31b-cloud', outcome: 'stream_error' }));
    sink.record(row({ finalProvider: 'anthropic', model: 'claude-opus-5-5' }));
    sink.flush();

    const stats = computeStats(db);
    assert.deepEqual(stats.cheapByModel, [
      { model: 'gemma4:31b-cloud', requests: 2, ok: 1 },
      { model: 'gpt-oss:20b-cloud', requests: 1, ok: 1 },
    ]);
    assert.match(renderStats(stats), /gemma4:31b-cloud\s+2\s+1/);
    await sink.close();
  });

  it('reports the last quota reading and the turns the quota sent cheap', async () => {
    const db = openTelemetryDb(':memory:');
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    sink.record(row({ finalProvider: 'openai', routeReason: 'quota:pressure' }));
    sink.record(row({ finalProvider: 'openai', routeReason: 'quota:reclassified' }));
    sink.record(row({ finalProvider: 'openai', routeReason: 'failover:primary-rate-limited' }));
    sink.record(row({ routeReason: 'sticky' }));
    sink.flush();
    recordQuota(db, { utilization: 0.83, window: '5h', observedAt: new Date('2026-10-07T12:00:00Z') });

    const stats = computeStats(db);
    assert.equal(stats.quotaRouted, 3);
    assert.deepEqual(stats.lastQuota && { u: stats.lastQuota.utilization, w: stats.lastQuota.window }, { u: 0.83, w: '5h' });
    assert.match(renderStats(stats), /Sent cheap by quota pressure\s+3/);
    assert.match(renderStats(stats), /Claude quota \(last seen\)\s+83% 5h/);
    await sink.close();
  });

  it('migrates a database written by the previous build without losing rows', () => {
    const dir = mkdtempSync(join(tmpdir(), 'jev-migrate-'));
    const file = join(dir, 'telemetry.db');
    const old = new DatabaseSync(file);
    for (const sql of MIGRATIONS.slice(0, 2)) old.exec(sql);
    old.exec('PRAGMA user_version = 2');
    old.exec(`INSERT INTO router_logs (final_provider, route_reason, outcome, latency_ms) VALUES ('openai', 'classified', 'ok', 5)`);
    old.close();

    const db = openTelemetryDb(file);
    assert.equal((db.prepare('PRAGMA user_version').get() as { user_version: number }).user_version, MIGRATIONS.length);
    assert.deepEqual({ ...db.prepare('SELECT tool_calls, inspection_miss FROM router_logs').get() }, { tool_calls: null, inspection_miss: null });
    assert.equal(computeStats(db).inspectionMisses, 0);
    db.close();
  });

  it('keeps rows queued when a write fails and retries on the next flush', async () => {
    const db = openTelemetryDb(':memory:');
    const errors: unknown[] = [];
    const sink = new SqliteTelemetry(db, { flushIntervalMs: 60_000, onError: (e) => errors.push(e) });
    db.exec('ALTER TABLE router_logs RENAME TO router_logs_tmp');
    sink.record(row({}));
    sink.flush();
    assert.equal(errors.length, 1);
    db.exec('ALTER TABLE router_logs_tmp RENAME TO router_logs');
    sink.flush();
    assert.equal(computeStats(db).total, 1);
    await sink.close();
  });

  it('rolls back a failed batch so no row is written twice', () => {
    const db = openTelemetryDb(':memory:');
    const insert = `INSERT INTO router_logs (final_provider, route_reason, outcome, latency_ms) VALUES ('openai', 'classified', 'ok', 5)`;
    assert.throws(() =>
      inTransaction(db, () => {
        db.exec(insert);
        throw new Error('disk full');
      }),
    );
    assert.equal(computeStats(db).total, 0);
    inTransaction(db, () => db.exec(insert));
    assert.equal(computeStats(db).total, 1);
    db.close();
  });

  it('parses --since spans', () => {
    const now = Date.parse('2026-10-03T00:00:00Z');
    assert.equal(parseSince('24h', now).toISOString(), '2026-10-02T00:00:00.000Z');
    assert.equal(parseSince('7d', now).toISOString(), '2026-09-26T00:00:00.000Z');
    assert.throws(() => parseSince('1w', now));
  });
});
