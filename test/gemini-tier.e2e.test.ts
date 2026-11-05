import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, describe, it, beforeEach } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { EventEmitter } from 'node:events';
import { Readable, Writable } from 'node:stream';
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
  let cheapCalls = 0;
  let cheapQueue: (() => any)[] = [];
  let cheapFails = false;
  const DEFAULT_DIST = { simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1.0 };
  let classifierDist: any = DEFAULT_DIST;

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
    cheapCalls = 0;
    cheapQueue = [];
    cheapFails = false;
    classifierDist = DEFAULT_DIST;
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
    const classifier = { name: 'scripted', classify: async () => classifierDist };
    
    const providers = {
      primary: new AnthropicProvider(config.primary),
      gemini: config.gemini ? new GeminiCliProvider({ ...config.gemini, firstByteTimeoutMs: config.timeouts.firstByteMs, spawnFn: fakeSpawnFn } as any) : undefined,
      ...(config.cheap
        ? {
            cheap: {
              send: async () => {
                cheapCalls++;
                const fn = cheapQueue.shift();
                if (fn) return fn();
                if (cheapFails) return { kind: 'unavailable', reason: 'empty response' };
                return {
                  kind: 'response',
                  status: 200,
                  headers: {},
                  body: Readable.from([
                    JSON.stringify({
                      type: 'message',
                      role: 'assistant',
                      content: [{ type: 'text', text: 'local ollama ok' }],
                      stop_reason: 'end_turn',
                      usage: {},
                    }),
                  ]),
                };
              },
            } as any,
          }
        : {}),
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

  it('busy falls back to local Ollama (CHEAP_ROUTE=fallback) instead of Claude, when configured', async () => {
    const { proxy, url } = await createTestServer({
      GEMINI_TIER: 'on',
      GEMINI_TIER_CONCURRENCY: '1',
      CHEAP_ROUTE: 'fallback',
      CHEAP_BASE_URL: 'http://127.0.0.1:11434/v1',
      CHEAP_MODEL: 'gpt-oss:20b-cloud',
    });
    try {
      spawnQueue.push(() => {
        const fakeChild = new EventEmitter() as any;
        fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
        fakeChild.stdout = new EventEmitter();
        fakeChild.stderr = new EventEmitter();
        fakeChild.pid = 304;
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
      await new Promise(r => setTimeout(r, 20));

      const p2 = request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 't2' }] })
      });

      const [r1, r2] = await Promise.all([p1, p2]);
      assert.equal(r1.headers['x-jev-route'], 'free; reason=gemini:text-only');
      assert.equal(r2.headers['x-jev-route'], 'cheap; reason=cheap:local-fallback');
      assert.equal(anthropicHits, 0); // local fallback served it: Claude never saw this turn
      await r1.body.text();
      await r2.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  // Regression: a simple turn with tool_choice any was routed 'free'; the server faked a
  // Flash skip and the local Ollama fallback answered a turn that must reach Claude.
  it('tool_choice any never reaches Flash or the local Ollama fallback', async () => {
    classifierDist = { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, textOnly: 0.95 };
    const { proxy, url } = await createTestServer({
      GEMINI_TIER: 'on',
      CHEAP_ROUTE: 'fallback',
      CHEAP_BASE_URL: 'http://127.0.0.1:1',
      CHEAP_MODEL: 'qwen2.5:14b',
    });
    try {
      const res = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          model: 'claude',
          messages: [{ role: 'user', content: 'hi' }],
          tools: [{ name: 'get_time', input_schema: { type: 'object', properties: {} } }],
          tool_choice: { type: 'any' },
        }),
      });
      await res.body.text();
      assert.equal(cheapCalls, 0);
      assert.equal(spawnCalls, 0);
      assert.equal(anthropicHits, 1);
      assert.equal(res.headers['x-jev-route'], 'primary; reason=gemini:ineligible-input');
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

  it('fallback order: Flash fails -> Argon (if enabled and answers) before Cheap; Flash and Argon both fail -> Cheap answers before Claude', async () => {
    classifierDist = { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, textOnly: 0.95 };
    const { proxy, url } = await createTestServer({
      GEMINI_TIER: 'on',
      ARGON_TIER: 'on',
      CHEAP_ROUTE: 'fallback',
      CHEAP_BASE_URL: 'http://127.0.0.1:11434/v1',
      CHEAP_MODEL: 'gpt-oss:20b-cloud',
    });
    const failing = () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 304;
      setTimeout(() => fakeChild.emit('close', 1), 2);
      return fakeChild;
    };
    try {
      // 1. Flash fails and Argon answers: cheap provider must NOT be called
      spawnQueue.push(failing); // Flash fails; Argon low/high (default fakeSpawnFn) succeeds
      const r1 = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'test-argon-first' }] }),
      });
      assert.equal(r1.statusCode, 200);
      assert.equal(r1.headers['x-jev-route'], 'argon; reason=escalated:flash-failed');
      assert.equal(cheapCalls, 0);
      assert.equal(anthropicHits, 0);
      await r1.body.text();

      // 2. Flash and Argon both fail: cheap is called and answers before Claude
      for (let i = 0; i < 5; i++) spawnQueue.push(failing);
      const r2 = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'test-cheap-after-argon' }] }),
      });
      assert.equal(r2.statusCode, 200);
      assert.equal(r2.headers['x-jev-route'], 'cheap; reason=cheap:local-fallback');
      assert.equal(cheapCalls, 1);
      assert.equal(anthropicHits, 0);
      await r2.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('first order: Cheap runs first; on failure Flash answers; on Flash+Argon failure Claude answers (cheap not retried)', async () => {
    classifierDist = { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, textOnly: 0.95 };
    const { proxy, url } = await createTestServer({
      GEMINI_TIER: 'on',
      ARGON_TIER: 'on',
      CHEAP_ROUTE: 'first',
      CHEAP_BASE_URL: 'http://127.0.0.1:11434/v1',
      CHEAP_MODEL: 'gpt-oss:20b-cloud',
    });
    const failing = () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 304;
      setTimeout(() => fakeChild.emit('close', 1), 2);
      return fakeChild;
    };
    try {
      // (a) cheap answers -> header 'cheap; reason=cheap:local-first', Flash spawn not used, anthropicHits 0
      cheapCalls = 0;
      spawnCalls = 0;
      anthropicHits = 0;
      const r1 = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'test-cheap-first' }] }),
      });
      assert.equal(r1.statusCode, 200);
      assert.equal(r1.headers['x-jev-route'], 'cheap; reason=cheap:local-first');
      assert.equal(cheapCalls, 1);
      assert.equal(spawnCalls, 0);
      assert.equal(anthropicHits, 0);
      await r1.body.text();

      // (b) cheap fails (make cheap mock fail/return 500) -> Flash answers -> header route free, cheapCalls 1
      cheapCalls = 0;
      spawnCalls = 0;
      anthropicHits = 0;
      cheapQueue.push(() => ({ kind: 'unavailable', reason: 'HTTP 500' }));
      const r2 = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'test-cheap-fail-flash-ok' }] }),
      });
      assert.equal(r2.statusCode, 200);
      assert.ok(String(r2.headers['x-jev-route']).startsWith('free;'));
      assert.equal(cheapCalls, 1);
      assert.equal(spawnCalls, 1);
      assert.equal(anthropicHits, 0);
      await r2.body.text();

      // (c) cheap fails and Flash and Argon fail -> Claude answers, cheapCalls exactly 1 (not retried)
      cheapCalls = 0;
      spawnCalls = 0;
      anthropicHits = 0;
      cheapQueue.push(() => ({ kind: 'unavailable', reason: 'HTTP 500' }));
      for (let i = 0; i < 5; i++) spawnQueue.push(failing);
      const r3 = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'test-cheap-fail-argon-fail-claude' }] }),
      });
      assert.equal(r3.statusCode, 200);
      assert.ok(String(r3.headers['x-jev-route']).startsWith('primary;'));
      assert.equal(cheapCalls, 1);
      assert.equal(anthropicHits, 1);
      await r3.body.text();
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  // Regression (PR #11 review): with CHEAP_ROUTE=first the local Ollama ran on every free turn with
  // no breaker, so a dead or hung server cost up to CHEAP_TIMEOUT_MS on each turn before Flash.
  it('CHEAP_ROUTE=first: a failing local Ollama is skipped after 3 failures, Flash keeps serving', async () => {
    classifierDist = { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, textOnly: 0.95 };
    cheapFails = true;
    const { proxy, url } = await createTestServer({
      GEMINI_TIER: 'on', ARGON_TIER: 'off',
      CHEAP_ROUTE: 'first', CHEAP_BASE_URL: 'http://127.0.0.1:1', CHEAP_MODEL: 'local',
    });
    try {
      for (let i = 0; i < 5; i++) {
        const r = await request(`${url}/v1/messages`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: `turn ${i}` }] }),
        });
        assert.match(String(r.headers['x-jev-route']), /^free;/, `turn ${i} served by Flash`);
        await r.body.text();
      }
      assert.equal(cheapCalls, 3, 'the breaker opened after 3 failures');
      const health = await (await request(`${url}/healthz`)).body.json() as any;
      assert.equal(health.cheap?.breaker, 'open');
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  it('CHEAP_ROUTE=first: an input the local model cannot translate never opens its breaker', async () => {
    classifierDist = { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, textOnly: 0.95 };
    const { proxy, url } = await createTestServer({
      GEMINI_TIER: 'on', ARGON_TIER: 'off',
      CHEAP_ROUTE: 'first', CHEAP_BASE_URL: 'http://127.0.0.1:1', CHEAP_MODEL: 'local',
    });
    for (let i = 0; i < 5; i++) cheapQueue.push(() => ({ kind: 'unavailable', reason: 'not translatable: user block: image' }));
    try {
      for (let i = 0; i < 5; i++) {
        const r = await request(`${url}/v1/messages`, {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: `turn ${i}` }] }),
        });
        await r.body.text();
      }
      assert.equal(cheapCalls, 5, 'every turn still tried the local model');
    } finally {
      await proxy.close();
      telemetry.close();
    }
  });

  // The Argon hop runs through the same `agy` as Flash: when Flash timed out or lost the network,
  // history shows Argon never rescued the turn and cost up to 180 s more before Claude answered.
  describe('Argon after a Flash failure', () => {
    const silent = () => {
      const child = new EventEmitter() as any;
      child.pid = undefined; // nothing real to kill
      child.stdin = new Writable({ write(c, e, cb) { cb(); } });
      child.stdout = new EventEmitter();
      child.stderr = new EventEmitter();
      return child;
    };
    const exitsWithError = () => {
      const child = new EventEmitter() as any;
      child.stdin = new Writable({ write(c, e, cb) { cb(); } });
      child.stdout = new EventEmitter();
      child.stderr = new EventEmitter();
      child.pid = 302;
      setTimeout(() => child.emit('close', 1), 5);
      return child;
    };
    // A simple, text-only turn: routes to Flash first (a structural one would go straight to Argon).
    beforeEach(() => {
      classifierDist = { simple: 0.97, standard: 0.02, structural: 0.01, risk: 0.05, textOnly: 1.0 };
    });
    const ask = (url: string) =>
      request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', messages: [{ role: 'user', content: 'hello' }] }),
      });

    it('skips Argon when Flash timed out waiting for the first byte', async () => {
      const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on', ARGON_TIER: 'on', TIER_FIRST_BYTE_TIMEOUT_MS: '60' });
      spawnQueue.push(silent, silent, silent);
      try {
        const res = await ask(url);
        await res.body.text();
        assert.equal(res.statusCode, 200);
        assert.equal(spawnCalls, 1, 'only Flash was spawned');
        assert.equal(anthropicHits, 1);
        assert.equal(res.headers['x-jev-route'], 'primary; reason=failover:gemini-unavailable');
      } finally {
        await proxy.close();
        telemetry.close();
      }
    });

    it('skips Argon after a Flash timeout even when the local Ollama also fails', async () => {
      // Regression: tryCheap overwrote the failure cause, so the timeout was forgotten and Argon ran.
      cheapFails = true;
      const { proxy, url } = await createTestServer({
        GEMINI_TIER: 'on', ARGON_TIER: 'on', TIER_FIRST_BYTE_TIMEOUT_MS: '60',
        CHEAP_ROUTE: 'fallback', CHEAP_BASE_URL: 'http://127.0.0.1:1', CHEAP_API_KEY: 'k', CHEAP_MODEL: 'local',
      });
      spawnQueue.push(silent, silent, silent);
      try {
        const res = await ask(url);
        await res.body.text();
        assert.equal(cheapCalls, 1, 'Ollama was tried');
        assert.equal(spawnCalls, 1, 'only Flash was spawned');
        assert.equal(res.headers['x-jev-route'], 'primary; reason=failover:gemini-unavailable');
      } finally {
        await proxy.close();
        telemetry.close();
      }
    });

    it('still tries Argon after an ordinary Flash failure', async () => {
      const { proxy, url } = await createTestServer({ GEMINI_TIER: 'on', ARGON_TIER: 'on' });
      spawnQueue.push(exitsWithError, exitsWithError, exitsWithError);
      try {
        const res = await ask(url);
        await res.body.text();
        assert.equal(res.headers['x-jev-route'], 'primary; reason=escalated:argon-failed', 'Argon was tried');
        assert.equal(anthropicHits, 1);
      } finally {
        await proxy.close();
        telemetry.close();
      }
    });
  });
});
