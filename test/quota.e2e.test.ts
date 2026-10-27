import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import { after, before, describe, it } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { HeuristicClassifier } from '../src/classifier/heuristic-classifier.js';
import { loadConfig } from '../src/config.js';
import type { Tier } from '../src/domain/policy.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { buildServer } from '../src/proxy/server.js';
import { QuotaStore, type QuotaSnapshot } from '../src/quota.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { openTelemetryDb, type TelemetryDb } from '../src/telemetry/db.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';
import { recordQuota } from '../src/telemetry/schema.js';
import { latestQuota } from '../src/telemetry/statusline.js';

describe('quota capture (proxy)', () => {
  let anthropic: Server;
  let proxy: FastifyInstance;
  let proxyUrl: string;
  let db: TelemetryDb;
  let telemetry: SqliteTelemetry;
  let utilization = '0.42';
  const quota = new QuotaStore({ pressure: 0.8, critical: 0.95 }, (s: QuotaSnapshot) => recordQuota(db, s));

  before(async () => {
    anthropic = createServer((req, res) => {
      req.resume();
      req.on('end', () => {
        res.writeHead(200, {
          'content-type': 'application/json',
          'anthropic-ratelimit-unified-status': 'allowed',
          'anthropic-ratelimit-unified-5h-utilization': utilization,
          'anthropic-ratelimit-unified-7d-utilization': '0.10',
        });
        res.end('{"type":"message","content":[],"usage":{"input_tokens":1,"output_tokens":1}}');
      });
    });
    await new Promise<void>((r) => anthropic.listen(0, '127.0.0.1', r));
    db = openTelemetryDb(':memory:');
    telemetry = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ANTHROPIC_UPSTREAM_URL: `http://127.0.0.1:${(anthropic.address() as AddressInfo).port}`,
      LOG_LEVEL: 'fatal',
    });
    const router = new Router(new HeuristicClassifier(), new TtlLruStore<Tier>(100, 60_000), {
      policy: { minCheapProbability: config.router.minCheapProbability },
      primaryClasses: config.router.primaryClasses,
      classifierTimeoutMs: 1_000,
      classifierMaxChars: 4_000,
    });
    proxy = buildServer({
      config,
      router,
      providers: { primary: new AnthropicProvider(config.primary) },
      telemetry,
      quota,
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  });

  after(async () => {
    await proxy.close();
    anthropic.close();
  });

  const send = (session: string) =>
    request(`${proxyUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-claude-code-session-id': session },
      body: JSON.stringify({ model: 'claude-opus-5-5', max_tokens: 10, messages: [{ role: 'user', content: 'refactor the layers' }] }),
    });

  it('reads the quota off every Anthropic response, persists changes, and shows it on /healthz', async () => {
    await (await send('Q1')).body.text();
    assert.deepEqual(quota.current() && { u: quota.current()!.utilization, w: quota.current()!.window }, { u: 0.42, w: '5h' });

    utilization = '0.97';
    await (await send('Q2')).body.text();
    assert.equal(quota.level(), 'critical');
    assert.deepEqual(latestQuota(db) && { u: latestQuota(db)!.utilization, w: latestQuota(db)!.window }, { u: 0.97, w: '5h' });
    assert.equal((db.prepare('SELECT count(*) AS n FROM quota_observations').get() as { n: number }).n, 2);

    const health = (await (await request(`${proxyUrl}/healthz`)).body.json()) as { quota: { utilization: number; level: string } };
    assert.equal(health.quota.utilization, 0.97);
    assert.equal(health.quota.level, 'critical');
  });

  it('stamps each routed exchange with the quota seen when it was routed', async () => {
    await (await send('Q3')).body.text();
    for (let i = 0; i < 20; i++) {
      await sleep(10);
      telemetry.flush();
      const row = db.prepare(`SELECT quota_utilization AS q FROM router_logs WHERE session_id = 'Q3:main'`).get() as { q: number } | undefined;
      if (row) return assert.equal(row.q, 0.97);
    }
    assert.fail('no telemetry row');
  });
});
