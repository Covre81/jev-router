import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { describe, it } from 'node:test';
import { request } from 'undici';
import type { AddressInfo } from 'node:net';
import { loadConfig } from '../src/config.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import { buildServer } from '../src/proxy/server.js';
import type { Provider, ProviderRequest, ProviderResult } from '../src/providers/provider.js';

const jsonResponse = (text: string, toolUse = false): ProviderResult => ({
  kind: 'response',
  status: 200,
  headers: { 'content-type': 'application/json' },
  body: Readable.from([Buffer.from(
    JSON.stringify({
      type: 'message',
      role: 'assistant',
      model: 'stub',
      content: toolUse
        ? [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: 'x' } }]
        : [{ type: 'text', text }],
      stop_reason: toolUse ? 'tool_use' : 'end_turn',
      usage: { input_tokens: 1, output_tokens: 1 },
    })
  )]),
});

const ORIGINAL_429_BODY = JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: '5h limit hit' } });

function stub(name: string, fn: (r: ProviderRequest) => ProviderResult) {
  const p = {
    name,
    hits: 0,
    async send(r: ProviderRequest) {
      p.hits++;
      return fn(r);
    },
  };
  return p as typeof p & Provider;
}

async function createTestServer(opts: {
  cursor?: Provider & { hits: number };
  gemini?: Provider & { hits: number };
  primary?: Provider & { hits: number };
  classification?: any;
  quota?: () => any;
}) {
  const config = loadConfig({
    CLASSIFIER: 'heuristic',
    LOG_LEVEL: 'fatal',
    CURSOR_TIER: 'on',
    GEMINI_TIER: 'on',
    ARGON_TIER: 'off',
  });
  const classifier = { name: 'scripted', classify: async () => opts.classification ?? ({ simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1 }) };
  const router = new Router(classifier as any, new TtlLruStore<any>(100, 60_000), {
    policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: false, argonEnabled: false },
    primaryClasses: new Set(['auxiliary']),
    cheapContextTokens: 100_000,
    classifierMaxChars: 4000,
    classifierTimeoutMs: 1500,
    ...(opts.quota ? { quota: { level: () => (opts.quota!()?.utilization >= 1 ? 'critical' : 'none'), current: opts.quota } } : {}),
  } as any);
  const providers: any = {
    primary: opts.primary,
    ...(opts.cursor ? { cursor: opts.cursor } : {}),
    ...(opts.gemini ? { gemini: opts.gemini } : {}),
  };
  const proxy: any = buildServer({
    config,
    router,
    providers,
    ...(opts.cursor ? { cursorBreaker: new CircuitBreaker(3, 300_000, 1) } : {}),
    ...(opts.gemini ? { geminiBreaker: new CircuitBreaker(3, 300_000, 1) } : {}),
  } as any);
  await proxy.listen({ port: 0, host: '127.0.0.1' });
  return { proxy, url: `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}` };
}

const rateLimitedPrimary = (status = 429) =>
  stub('primary', () => ({
    kind: 'response',
    status,
    headers: { 'content-type': 'application/json', 'x-original': 'yes' },
    body: Readable.from([Buffer.from(ORIGINAL_429_BODY)]),
  }));

const send = (url: string, body: unknown, session = 'cascade-s1') =>
  request(`${url}/v1/messages`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-claude-code-session-id': session },
    body: JSON.stringify(body),
  });

describe('failover cascade Claude -> Cursor -> Gemini', () => {
  it('1. primary 429 served by Cursor; tool-loop turn sticks to Cursor', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok', true));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r1.statusCode, 200);
      assert.equal(r1.headers['x-jev-route'], 'cursor; reason=failover:primary-rate-limited');
      await r1.body.text();

      const r2 = await send(url, {
        model: 'claude',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: {} }] },
          { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'file body' }] },
        ],
      });
      assert.equal(r2.statusCode, 200);
      assert.equal(r2.headers['x-jev-route'], 'cursor; reason=sticky');
      await r2.body.text();

      assert.equal(primary.hits, 1, 'primary rate limited once');
      assert.equal(cursor.hits, 2, 'cursor served initial failover and continuation');
      assert.equal(gemini.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('2. Cursor unavailable => Gemini serves', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'process exited 1' }));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'free; reason=failover:cursor-unavailable');
      await r.body.text();
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('3. Cursor+Gemini unavailable => 429 relayed verbatim', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'cursor down' }));
    const gemini = stub('gemini', () => ({ kind: 'unavailable', reason: 'gemini down' }));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r.statusCode, 429);
      assert.equal(r.headers['x-original'], 'yes');
      assert.deepEqual(await r.body.json(), JSON.parse(ORIGINAL_429_BODY));
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('4. all down => original 429 relayed verbatim', async () => {
    const primary = rateLimitedPrimary();
    const { proxy, url } = await createTestServer({ primary });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r.statusCode, 429);
      assert.equal(r.headers['x-original'], 'yes');
      await r.body.text();
      assert.equal(primary.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('5. quota probe (max_tokens=1) 429 => never cascades', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini });
    try {
      const r = await send(url, { model: 'claude', max_tokens: 1, messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r.statusCode, 429);
      assert.equal(r.headers['x-original'], 'yes');
      await r.body.text();
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('6. throwing Cursor provider => Gemini serves (cascade advances)', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => { throw new Error('boom'); });
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'free; reason=failover:cursor-unavailable');
      await r.body.text();
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('7. primary 529 served by Cursor', async () => {
    const primary = rateLimitedPrimary(529);
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const { proxy, url } = await createTestServer({ primary, cursor });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] });
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'cursor; reason=failover:primary-rate-limited');
      await r.body.text();
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 1);
    } finally {
      await proxy.close();
    }
  });
});
