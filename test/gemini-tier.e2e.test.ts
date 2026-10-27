import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, describe, it, beforeEach } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { randomUUID } from 'node:crypto';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdtemp, rm } from 'node:fs/promises';
import { loadConfig } from '../src/config.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { GeminiCliProvider } from '../src/providers/gemini/provider.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import { buildServer } from '../src/proxy/server.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';
import { openTelemetryDb } from '../src/telemetry/db.js';
import { computeStats, renderStats } from '../src/telemetry/stats.js';

async function listen(server: Server): Promise<string> {
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

describe('Gemini Tier E2E', () => {
  let anthropic: Server;
  let anthropicUrl: string;
  let anthropicHits = 0;
  let lastAnthropicBody: any;
  let dbDir: string;
  let telemetry: SqliteTelemetry;
  let telemetryDb: any; // Add telemetryDb
  let spawnCalls = 0;
  let spawnQueue: (() => any)[] = [];
  
  before(async () => {
    anthropic = createServer((req, res) => {
      anthropicHits++;
      const parts: Buffer[] = [];
      req.on('data', (c) => parts.push(c));
      req.on('end', () => {
        lastAnthropicBody = JSON.parse(Buffer.concat(parts).toString('utf8') || 'null');
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ type: 'message', role: 'assistant', content: [{ type: 'text', text: 'upstream ok' }], stop_reason: 'end_turn', usage: {} }));
      });
    });
    anthropicUrl = await listen(anthropic);
    dbDir = await mkdtemp(join(tmpdir(), 'jev-gemini-test-'));
  });

  after(async () => {
    anthropic.close();
    await rm(dbDir, { recursive: true, force: true });
  });

  beforeEach(() => {
    anthropicHits = 0;
    spawnCalls = 0;
    spawnQueue = [];
  });

  const fakeSpawnFn = (bin: string, args: string[], opts: any) => {
    spawnCalls++;
    const fn = spawnQueue.shift();
    if (fn) return fn();
    const child = new EventEmitter() as any;
    child.stdin = new Writable({ write(c, e, cb) { cb(); } });
    child.stdout = new EventEmitter();
    child.stderr = new EventEmitter();
    child.pid = 300;
    setTimeout(() => {
      child.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'gemini ok', usage: { input_tokens: 1, output_tokens: 1 } } }) + '\n'));
      child.emit('close', 0);
    }, 5);
    return child;
  };

  async function createTestServer(envOverrides: NodeJS.ProcessEnv = {}) {
    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ANTHROPIC_UPSTREAM_URL: anthropicUrl,
      CHEAP_API_KEY: 'test-key',
      ANTHROPIC_API_KEY: 'test-key',
      ARGON_TIER: 'off',
      LOG_LEVEL: 'fatal',
      JEV_ROUTER_HOME: dbDir,
      ...envOverrides
    });
    
    telemetryDb = openTelemetryDb(join(dbDir, 'test-' + randomUUID() + '.db'));
    telemetry = new SqliteTelemetry(telemetryDb, { flushIntervalMs: 60_000 });
    const store = new TtlLruStore<any>(100, 60_000);
    const classifier = { name: 'scripted', classify: async () => ({ simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1.0 }) };
    
    const providers = {
      primary: new AnthropicProvider(config.primary),
      cheap: { send: async () => ({ kind: 'response', status: 200, headers: {}, body: null }) } as any,
      gemini: config.gemini ? new GeminiCliProvider({ ...config.gemini, spawnFn: fakeSpawnFn } as any) : undefined
    };

    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: false, argonEnabled: config.argon.enabled },
      primaryClasses: new Set(['complex', 'auxiliary']),
      cheapContextTokens: 4000,
      ...(config.gemini ? {
        geminiPolicy: {
          enabled: true,
          minTextOnly: config.gemini.minTextOnly,
          pressureMinTextOnly: config.gemini.pressureMinTextOnly
        },
        geminiFromPrimary: config.gemini.fromPrimary
      } : {}),
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1500
    });

    const geminiBreaker = config.gemini ? new CircuitBreaker(config.gemini.breakerFailures, config.gemini.breakerCooldownMs, config.gemini.maxConcurrency) : undefined;
    const proxyOptions: any = { config, router, telemetry, providers };
    if (geminiBreaker) proxyOptions.geminiBreaker = geminiBreaker;
    const proxy = buildServer(proxyOptions);
    await proxy.listen({ port: 0, host: '127.0.0.1' });
    return { proxy, url: `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}` };
  }

  it('disabled by default', async () => {
    const { proxy, url } = await createTestServer();
    try {
      const { statusCode, headers, body } = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello' }] })
      });
      assert.equal(statusCode, 200);
      assert.ok(headers['x-jev-route']?.toString().startsWith('primary'));
      assert.equal(anthropicHits, 1);
      
      const res = await request(`${url}/healthz`);
      const h = await res.body.json() as any;
      assert.equal(h.gemini, null);
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('enabled + fake runner success (non-stream and stream)', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on' });
    try {
      let { statusCode, headers, body } = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello' }] })
      });
      assert.equal(statusCode, 200);
      assert.equal(headers['x-jev-route'], 'free; reason=gemini:text-only');
      assert.equal(anthropicHits, 0);
      const data = await body.json() as any;
      assert.equal(data.content[0].text, 'gemini ok');

      // stream
      let res = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-claude-code-session-id': 'e2e-stream' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello' }], stream: true })
      });
      assert.equal(res.statusCode, 200);
      assert.equal(res.headers['x-jev-route'], 'free; reason=gemini:text-only');
      assert.equal(anthropicHits, 0);
      await res.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('enabled + fake runner failure -> failover', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on' });
    spawnQueue.push(() => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 301;
      setTimeout(() => fakeChild.emit('close', 1), 5);
      return fakeChild;
    });

    try {
      const { statusCode, headers, body } = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello' }] })
      });
      assert.equal(statusCode, 200);
      assert.equal(headers['x-jev-route'], 'primary; reason=failover:gemini-unavailable');
      assert.equal(anthropicHits, 1);
      const data = await body.json() as any;
      assert.equal(data.content[0].text, 'upstream ok');
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('failover gives Claude the untouched body and /route/:session reports why', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on', GEMINI_TIER_FROM_PRIMARY: 'true' });
    const geminiStdin: string[] = [];
    const failing = () => {
      const child = new EventEmitter() as any;
      let input = '';
      child.stdin = new Writable({ write(c, e, cb) { input += c.toString(); cb(); }, final(cb) { geminiStdin.push(input); cb(); } });
      child.stdout = new EventEmitter();
      child.stderr = new EventEmitter();
      child.pid = 304;
      setTimeout(() => child.emit('close', 1), 5);
      return child;
    };
    spawnQueue.push(failing, failing);
    // A cwd with no .ai-memory.toml: the handoff never touches the real ai-memory.
    const system = `Working directory: ${dbDir}`;
    const headers = { 'content-type': 'application/json', 'x-claude-code-session-id': 'e2e-route' };
    const send = (messages: any[]) => request(`${url}/v1/messages`, { method: 'POST', headers, body: JSON.stringify({ model: 'claude', system, messages }) });
    try {
      const t1 = [{ role: 'user', content: 'hello' }];
      await (await send(t1)).body.text(); // Gemini fails, Claude serves turn 1
      const r2 = await send([...t1, { role: 'assistant', content: 'upstream ok' }, { role: 'user', content: 'and now?' }]);
      assert.equal(r2.headers['x-jev-route'], 'primary; reason=failover:gemini-unavailable');
      await r2.body.text();
      assert.ok(geminiStdin.at(-1)?.includes('Model transition: primary -> gemini'), 'Gemini got the handoff');
      assert.equal(lastAnthropicBody.system, system, 'Claude got the system prompt as sent');

      const route = await request(`${url}/route/e2e-route`);
      assert.equal(route.statusCode, 200);
      const last = await route.body.json() as any;
      assert.equal(last.route, 'primary');
      assert.equal(last.reason, 'failover:gemini-unavailable');

      const unknown = await request(`${url}/route/no-such-session`);
      assert.equal(unknown.statusCode, 404);
      await unknown.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('breaker', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on', GEMINI_TIER_BREAKER: '3/1000' });
    try {
      for (let i = 0; i < 4; i++) {
        spawnQueue.push(() => {
          const fakeChild = new EventEmitter() as any;
          fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
          fakeChild.stdout = new EventEmitter();
          fakeChild.stderr = new EventEmitter();
          fakeChild.pid = 302;
          setTimeout(() => fakeChild.emit('close', 1), 2);
          return fakeChild;
        });
        await request(`${url}/v1/messages`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: `turn ${i}` }] })
        }).then(r => r.body.text());
      }
      
      // 3 attempts call spawn, 4th is skipped
      assert.equal(spawnCalls, 3);
      assert.equal(anthropicHits, 4);

      // healthz
      const hres = await request(`${url}/healthz`);
      const h = await hres.body.json() as any;
      assert.ok(h.gemini.model);
      assert.equal(h.gemini.breaker, 'open');
      assert.equal(h.gemini.inFlight, 0);
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('busy', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on', GEMINI_TIER_CONCURRENCY: '1' });
    try {
      spawnQueue.push(() => {
        const fakeChild = new EventEmitter() as any;
        fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
        fakeChild.stdout = new EventEmitter();
        fakeChild.stderr = new EventEmitter();
        fakeChild.pid = 303;
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'slow gemini ok', usage: {} } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 200);
        return fakeChild;
      });

      const p1 = request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 't1' }] })
      });
      // sleep 20ms to let p1 acquire concurrency
      await new Promise(r => setTimeout(r, 20));

      const p2 = request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 't2' }] })
      });

      const [r1, r2] = await Promise.all([p1, p2]);
      assert.equal(r1.headers['x-jev-route'], 'free; reason=gemini:text-only');
      assert.equal(r2.headers['x-jev-route'], 'primary; reason=skipped:gemini-busy');
      assert.equal(spawnCalls, 1);
      assert.equal(anthropicHits, 1);
      await r1.body.text();
      await r2.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('busy serves Claude once and keeps the session on Gemini (regression: busy pinned it to Claude)', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on' });
    try {
      spawnQueue.push(() => {
        const slow = new EventEmitter() as any;
        slow.stdin = new Writable({ write(c, e, cb) { cb(); } });
        slow.stdout = new EventEmitter();
        slow.stderr = new EventEmitter();
        slow.pid = 304;
        setTimeout(() => {
          slow.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'slow', usage: {} } }) + '\n'));
          slow.emit('close', 0);
        }, 200);
        return slow;
      });
      const send = (session: string, messages: any[]) =>
        request(`${url}/v1/messages`, {
          method: 'POST',
          headers: { 'content-type': 'application/json', 'x-claude-code-session-id': session },
          body: JSON.stringify({ model: 'claude', messages }),
        });
      const p1 = send('holder', [{ role: 'user', content: 'a' }]);
      await new Promise((r) => setTimeout(r, 20));
      const r2 = await send('other', [{ role: 'user', content: 'b' }]);
      assert.equal(r2.headers['x-jev-route'], 'primary; reason=skipped:gemini-busy');
      await r2.body.text();
      await (await p1).body.text();

      const loop = await send('other', [
        { role: 'user', content: 'b' },
        { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Read', input: {} }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }] },
      ]);
      assert.equal(loop.headers['x-jev-route'], 'free; reason=sticky');
      await loop.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('telemetry', async () => {
    const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on' });
    try {
      // 1 success
      await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello' }] })
      }).then(r => r.body.text());

      // 1 failover
      spawnQueue.push(() => {
        const fakeChild = new EventEmitter() as any;
        fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
        fakeChild.stdout = new EventEmitter();
        fakeChild.stderr = new EventEmitter();
        fakeChild.pid = 304;
        setTimeout(() => fakeChild.emit('close', 1), 2);
        return fakeChild;
      });
      await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello2' }] })
      }).then(r => r.body.text());

      // wait for telemetry async flush
      await new Promise(r => setTimeout(r, 200));

      telemetry.flush();
      const stats = computeStats(telemetryDb, { since: new Date(Date.now() - 10000) });
      assert.equal(stats.byProvider.gemini.requests, 1);
      assert.equal(stats.geminiFallbacks, 1);
      
      const txt = renderStats(stats);
      assert.ok(txt.includes('Diverted to Gemini'));
      assert.ok(txt.includes('gemini (subscription)'));
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });
});
