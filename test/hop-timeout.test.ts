import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { createServer, type IncomingHttpHeaders, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import { after, before, describe, it } from 'node:test';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { loadConfig } from '../src/config.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import type { Provider } from '../src/providers/provider.js';
import { runAgy } from '../src/providers/gemini/agy-runner.js';
import { runCursorAgent } from '../src/providers/cursor/agent-runner.js';
import { buildServer } from '../src/proxy/server.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

function fakeChild() {
  const child = new EventEmitter() as any;
  child.pid = 99999;
  child.stdin = new EventEmitter();
  child.stdin.write = () => true;
  child.stdin.end = () => {};
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  return child;
}

async function fakeCursorRoot() {
  const root = join(tmpdir(), `cursor-test-timeout-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const ver = join(root, 'versions', '2026.04.01-abc');
  await mkdir(ver, { recursive: true });
  await writeFile(join(ver, 'node.exe'), 'FAKE');
  await writeFile(join(ver, 'index.js'), 'FAKE');
  return { root, ver };
}

interface Seen {
  method: string;
  url: string;
  headers: IncomingHttpHeaders;
  raw: string;
  body: any;
}

type Handler = (seen: Seen, res: ServerResponse) => Promise<void> | void;

async function fakeServer(handler: () => Handler, log: Seen[]): Promise<{ server: Server; url: string }> {
  const server = createServer(async (req, res) => {
    const parts: Buffer[] = [];
    for await (const c of req) parts.push(c as Buffer);
    const raw = Buffer.concat(parts).toString('utf8');
    const seen = { method: req.method!, url: req.url!, headers: req.headers, raw, body: raw ? JSON.parse(raw) : undefined };
    log.push(seen);
    await handler()(seen, res);
  });
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  return { server, url: `http://127.0.0.1:${(server.address() as AddressInfo).port}` };
}

describe('Hop timeout and hang protection', () => {
  describe('runner timeout protections', () => {
    it('runAgy: fake child that never writes fails within firstByteTimeoutMs', async () => {
      const child = fakeChild();
      const spawnFn = () => child;

      const started = Date.now();
      const res = await runAgy(
        'test prompt',
        {
          bin: 'agy',
          model: 'gemini-3.8-flash-high',
          timeoutMs: 5000,
          firstByteTimeoutMs: 150,
          home: '/tmp',
        },
        spawnFn as any,
      );
      const elapsed = Date.now() - started;

      assert.equal(res.ok, false);
      if (!res.ok) {
        assert.equal(res.reason, 'first-byte-timeout');
      }
      assert.ok(elapsed >= 140 && elapsed < 1000, `elapsed ${elapsed}ms should be ~150ms`);
    });

    it('runAgy: fake child that writes then stalls is killed by idleTimeoutMs', async () => {
      const child = fakeChild();
      const spawnFn = () => {
        setTimeout(() => {
          child.stdout.emit('data', Buffer.from(JSON.stringify({ type: 'start' }) + '\n'));
        }, 20);
        return child;
      };

      const started = Date.now();
      const res = await runAgy(
        'test prompt',
        {
          bin: 'agy',
          model: 'gemini-3.8-flash-high',
          timeoutMs: 5000,
          firstByteTimeoutMs: 500,
          idleTimeoutMs: 150,
          home: '/tmp',
        },
        spawnFn as any,
      );
      const elapsed = Date.now() - started;

      assert.equal(res.ok, false);
      if (!res.ok) {
        assert.equal(res.reason, 'idle-timeout');
      }
      assert.ok(elapsed >= 150 && elapsed < 1000, `elapsed ${elapsed}ms should be ~170ms`);
    });

    it('runAgy: exit 0 with empty output is treated as failure', async () => {
      // Case 1: no result event emitted
      const child1 = fakeChild();
      const spawnFn1 = () => {
        setTimeout(() => {
          child1.emit('close', 0);
        }, 10);
        return child1;
      };

      const res1 = await runAgy(
        'test prompt',
        {
          bin: 'agy',
          model: 'gemini-3.8-flash-high',
          timeoutMs: 5000,
          home: '/tmp',
        },
        spawnFn1 as any,
      );

      assert.equal(res1.ok, false);

      // Case 2: result event emitted but response text is empty
      const child2 = fakeChild();
      const spawnFn2 = () => {
        setTimeout(() => {
          child2.stdout.emit(
            'data',
            Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: '   ' } }) + '\n'),
          );
          child2.emit('close', 0);
        }, 10);
        return child2;
      };

      const res2 = await runAgy(
        'test prompt',
        {
          bin: 'agy',
          model: 'gemini-3.8-flash-high',
          timeoutMs: 5000,
          home: '/tmp',
        },
        spawnFn2 as any,
      );

      assert.equal(res2.ok, false);
      if (!res2.ok) {
        assert.equal(res2.reason, 'empty');
      }
    });

    it('runCursorAgent: fake child that never writes fails within firstByteTimeoutMs', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const child = fakeChild();
        const spawnFn = () => child;

        const started = Date.now();
        const res = await runCursorAgent(
          'test prompt',
          {
            bin: root,
            model: 'auto',
            timeoutMs: 5000,
            firstByteTimeoutMs: 150,
          },
          spawnFn as any,
        );
        const elapsed = Date.now() - started;

        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'first-byte-timeout');
        }
        assert.ok(elapsed >= 140 && elapsed < 1000, `elapsed ${elapsed}ms should be ~150ms`);
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });
  });

  describe('circuit breaker hang recording and skipping', () => {
    it('breaker opens after N hangs and skips the provider on next turn', () => {
      const breaker = new CircuitBreaker(2, 60_000, 1);
      assert.equal(breaker.state, 'closed');

      // First hang / failure
      assert.equal(breaker.acquire(), true);
      breaker.release('failure');
      assert.equal(breaker.state, 'closed');

      // Second hang / failure
      assert.equal(breaker.acquire(), true);
      breaker.release('failure');
      assert.equal(breaker.state, 'open');

      // Subsequent attempt is blocked by breaker
      assert.equal(breaker.acquire(), false);
    });
  });

  describe('streaming keepalive SSE and fallback relay', () => {
    let anthropic: { server: Server; url: string };
    let seenAnthropic: Seen[] = [];
    let proxy: FastifyInstance;
    let proxyUrl: string;
    let geminiCalls = 0;

    const anthropicSseOk: Handler = async (seen, res) => {
      res.writeHead(200, {
        'content-type': 'text/event-stream',
        'anthropic-ratelimit-unified-status': 'allowed',
      });
      res.write(
        `event: message_start\ndata: {"type":"message_start","message":{"model":"${seen.body.model}","usage":{"input_tokens":10,"output_tokens":1}}}\n\n`,
      );
      await sleep(20);
      res.write(`event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}\n\n`);
      res.end(`event: message_stop\ndata: {"type":"message_stop"}\n\n`);
    };

    before(async () => {
      anthropic = await fakeServer(() => anthropicSseOk, seenAnthropic);
    });

    after(async () => {
      await proxy?.close();
      anthropic.server.close();
    });

    it('sends event: ping while waiting and relays valid Claude stream after hop fallback', async () => {
      const config = loadConfig({
        ANTHROPIC_UPSTREAM_URL: anthropic.url,
        ANTHROPIC_API_KEY: 'test-key',
        CLASSIFIER: 'heuristic',
        GEMINI_TIER: 'on',
        GEMINI_TIER_MODEL: 'gemini-3.8-flash-high',
        ARGON_TIER: 'off',
      });

      const primary = new AnthropicProvider(config.primary);

      // Gemini provider that hangs for 120ms before reporting failure
      const mockGemini: Provider = {
        name: 'gemini',
        send: async () => {
          geminiCalls++;
          await sleep(120);
          return { kind: 'unavailable', reason: 'first-byte-timeout' };
        },
      };

      const classifier = {
        name: 'test',
        classify: async () => ({ simple: 0.95, standard: 0.05, structural: 0 }),
      };
      const store = new TtlLruStore<any>(10, 60_000);
      const router = new Router(classifier as any, store, {
        policy: { minCheapProbability: 0.8 },
        primaryClasses: new Set(),
        cheapContextTokens: 100_000,
        classifierMaxChars: 4000,
        classifierTimeoutMs: 1000,
        geminiPolicy: {
          enabled: true,
          minTextOnly: 0.8,
          pressureMinTextOnly: 0.6,
          replacesCheap: true,
        },
      });

      const geminiBreaker = new CircuitBreaker(3, 60_000, 1);

      proxy = buildServer({
        config,
        router,
        providers: {
          primary,
          gemini: mockGemini,
        },
        geminiBreaker,
        timeouts: {
          keepaliveIntervalMs: 40, // ping every 40ms
        },
      });

      await proxy.ready();
      await proxy.listen({ port: 0, host: '127.0.0.1' });
      const port = (proxy.server.address() as AddressInfo).port;
      proxyUrl = `http://127.0.0.1:${port}`;

      const res = await request(`${proxyUrl}/v1/messages`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'accept': 'text/event-stream',
          'x-api-key': 'test-key',
          'anthropic-version': '2023-06-01',
        },
        body: JSON.stringify({
          model: 'claude-3-5-sonnet',
          stream: true,
          messages: [{ role: 'user', content: 'hello' }],
        }),
      });

      assert.equal(res.statusCode, 200);
      assert.ok(res.headers['content-type']?.includes('text/event-stream'));

      const bodyText = await res.body.text();

      // Verify at least one ping was sent during the wait
      assert.ok(
        bodyText.includes('event: ping\ndata: {"type": "ping"}\n\n'),
        'Stream must contain keepalive ping',
      );

      // Verify Claude's fallback response was relayed cleanly
      assert.ok(bodyText.includes('event: message_start'), 'Stream must contain message_start');
      assert.ok(bodyText.includes('event: message_delta'), 'Stream must contain message_delta');
      assert.ok(bodyText.includes('event: message_stop'), 'Stream must contain message_stop');

      // Verify Gemini was tried
      assert.ok(geminiCalls >= 1);
    });
  });
});
