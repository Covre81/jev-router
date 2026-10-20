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
import type { Provider, ProviderResult } from '../src/providers/provider.js';

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

function stub(name: string, fn: () => ProviderResult) {
  const p = {
    name,
    hits: 0,
    async send() {
      p.hits++;
      return fn();
    },
  };
  return p as typeof p & Provider;
}

async function createTestServer(opts: {
  cursor?: Provider & { hits: number };
  gemini?: Provider & { hits: number };
  cheap?: Provider & { hits: number };
  primary?: Provider & { hits: number };
  classification?: any;
  quota?: () => any;
}) {
  const config = loadConfig({
    CLASSIFIER: 'heuristic',
    CHEAP_API_KEY: 'test-key',
    LOG_LEVEL: 'fatal',
    CURSOR_TIER: 'on',
    GEMINI_TIER: 'on',
  });
  const classifier = { name: 'scripted', classify: async () => opts.classification ?? ({ simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1 }) };
  const router = new Router(classifier as any, new TtlLruStore<any>(100, 60_000), {
    policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: false },
    primaryClasses: new Set(['auxiliary']),
    cheapContextTokens: 100_000,
    classifierMaxChars: 4000,
    classifierTimeoutMs: 1500,
    ...(opts.quota ? { quota: { level: () => (opts.quota!()?.utilization >= 1 ? 'critical' : 'none'), current: opts.quota } } : {}),
  } as any);
  const providers: any = {
    primary: opts.primary,
    cheap: opts.cheap,
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

describe('failover cascade Claude -> Cursor -> Gemini -> cheap', () => {
  it('1. primary 429 served by Cursor; tool-loop turn sticks to Cursor', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok', true));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
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

      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 2);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);

      const health = await request(`${url}/healthz`);
      const h = (await health.body.json()) as any;
      assert.equal(h.cursor.model, 'auto');
      assert.equal(h.cursor.breaker, 'closed');
    } finally {
      await proxy.close();
    }
  });

  it('2. Cursor unavailable => Gemini serves, cheap untouched', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s2');
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'gemini; reason=failover:cursor-unavailable');
      const data = (await r.body.json()) as any;
      assert.equal(data.content[0].text, 'gemini ok');
      assert.equal(cheap.hits, 0);
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('3. Cursor+Gemini unavailable => cheap serves', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const gemini = stub('gemini', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s3');
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'cheap; reason=failover:gemini-unavailable');
      await r.body.text();
      assert.equal(cheap.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('4. all down => original 429 relayed verbatim', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const gemini = stub('gemini', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const cheap = stub('cheap', () => ({ kind: 'unavailable', reason: 'down' }));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s4');
      assert.equal(r.statusCode, 429);
      assert.equal(r.headers['x-original'], 'yes');
      assert.equal(await r.body.text(), ORIGINAL_429_BODY);
    } finally {
      await proxy.close();
    }
  });

  it('5. quota probe (max_tokens=1) 429 => never cascades', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(url, { model: 'claude', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }, 'cascade-s5');
      assert.equal(r.statusCode, 429);
      await r.body.text();
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('3b. cheap-served cascade turn keeps the tool loop on cheap (primary not re-hit)', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const gemini = stub('gemini', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const cheap = stub('cheap', () => jsonResponse('cheap ok', true));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s3b');
      assert.equal(r1.statusCode, 200);
      assert.equal(r1.headers['x-jev-route'], 'cheap; reason=failover:gemini-unavailable');
      await r1.body.text();

      const r2 = await send(url, {
        model: 'claude',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: {} }] },
          { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'file body' }] },
        ],
      }, 'cascade-s3b');
      assert.equal(r2.statusCode, 200);
      assert.equal(r2.headers['x-jev-route'], 'cheap; reason=sticky');
      await r2.body.text();
      assert.equal(primary.hits, 1);
      assert.equal(cheap.hits, 2);
    } finally {
      await proxy.close();
    }
  });

  it('3c. cheap cascade turn does not demote the session: recovered quota sends the next human turn to Claude', async () => {
    let limited = true;
    const primary = stub('primary', () =>
      limited
        ? { kind: 'response', status: 429, headers: { 'content-type': 'application/json' }, body: Readable.from([Buffer.from(ORIGINAL_429_BODY)]) }
        : jsonResponse('claude ok'));
    const cursor = stub('cursor', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const gemini = stub('gemini', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    let snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const cls = { simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1 };
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap, quota: () => snap, classification: cls });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s3c');
      assert.equal(r1.headers['x-jev-route'], 'cheap; reason=failover:gemini-unavailable');
      await r1.body.text();

      limited = false;
      snap = { utilization: 0.2, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
      Object.assign(cls, { simple: 0.99, structural: 0.01, risk: 0 });
      const r2 = await send(url, {
        model: 'claude',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: 'cheap ok' },
          { role: 'user', content: 'thanks, what time is it in London?' },
        ],
      }, 'cascade-s3c');
      assert.equal(r2.statusCode, 200);
      assert.match(String(r2.headers['x-jev-route']), /^primary;/);
      await r2.body.text();
      assert.equal(cheap.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('6. throwing Cursor provider => Gemini serves (cascade advances)', async () => {
    const primary = rateLimitedPrimary();
    const cursor: any = { name: 'cursor', hits: 0, async send() { cursor.hits++; throw new Error('boom secret detail'); } };
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s-throw');
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'gemini; reason=failover:cursor-unavailable');
      await r.body.text();
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 1);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('7. quota probe inside a marked loop never reaches Cursor/Gemini', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s-probe-loop');
      assert.equal(r1.statusCode, 200);
      await r1.body.text();
      assert.equal(cursor.hits, 1);

      const r2 = await send(url, { model: 'claude', max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }, 'cascade-s-probe-loop');
      assert.equal(r2.statusCode, 429);
      assert.equal(await r2.body.text(), ORIGINAL_429_BODY);
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('8. risk-vetoed primary decision + 429 => original 429 verbatim, no cascade hits', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const risky = { simple: 0.99, standard: 0, structural: 0.01, risk: 0.95, riskScores: { security_sensitive: 0.95, destructive_or_production: 0, requires_inspection: 0 }, textOnly: 1 };
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap, classification: risky });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'store this Stripe key in the code' }] }, 'cascade-s-risk');
      assert.equal(r.statusCode, 429);
      assert.equal(r.headers['x-original'], 'yes');
      assert.equal(await r.body.text(), ORIGINAL_429_BODY);
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('8b. sticky primary turn: second human turn classified risky at cascade time => 429, no cascade hits', async () => {
    let limited = false;
    const primary = stub('primary', () =>
      limited
        ? { kind: 'response', status: 429, headers: { 'content-type': 'application/json', 'x-original': 'yes' }, body: Readable.from([Buffer.from(ORIGINAL_429_BODY)]) }
        : jsonResponse('claude ok'));
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const cls = { simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1 };
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap, classification: cls });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s-risk-sticky');
      assert.equal(r1.statusCode, 200);
      await r1.body.text();

      Object.assign(cls, {
        risk: 0.95,
        riskScores: { security_sensitive: 0.95, destructive_or_production: 0, requires_inspection: 0 },
      });
      limited = true;
      const r2 = await send(url, {
        model: 'claude',
        messages: [
          { role: 'user', content: 'hello' },
          { role: 'assistant', content: 'claude ok' },
          { role: 'user', content: 'delete the production database' },
        ],
      }, 'cascade-s-risk-sticky');
      assert.equal(r2.statusCode, 429);
      assert.equal(await r2.body.text(), ORIGINAL_429_BODY);
      assert.equal(primary.hits, 2);
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('8c. tool continuation: driving human text classified risky at cascade time => 429, no cascade hits', async () => {
    let limited = false;
    const primary = stub('primary', () =>
      limited
        ? { kind: 'response', status: 429, headers: { 'content-type': 'application/json', 'x-original': 'yes' }, body: Readable.from([Buffer.from(ORIGINAL_429_BODY)]) }
        : jsonResponse('claude ok'));
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const cls = { simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 1 };
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap, classification: cls });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'delete the production database' }] }, 'cascade-s-risk-tool');
      assert.equal(r1.statusCode, 200);
      await r1.body.text();

      Object.assign(cls, {
        risk: 0.95,
        riskScores: { security_sensitive: 0, destructive_or_production: 0.95, requires_inspection: 0 },
      });
      limited = true;
      const r2 = await send(url, {
        model: 'claude',
        messages: [
          { role: 'user', content: 'delete the production database' },
          { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: {} }] },
          { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'file body' }] },
        ],
      }, 'cascade-s-risk-tool');
      assert.equal(r2.statusCode, 429);
      assert.equal(await r2.body.text(), ORIGINAL_429_BODY);
      assert.equal(primary.hits, 2);
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('9. cursor->cheap advance records trivial: continuation stays on cheap, primary not re-hit', async () => {
    const primary = rateLimitedPrimary();
    const cursor: any = {
      name: 'cursor',
      hits: 0,
      async send() {
        cursor.hits++;
        return cursor.hits === 1 ? jsonResponse('cursor ok', true) : { kind: 'unavailable', reason: 'exit 1' };
      },
    };
    const gemini = stub('gemini', () => ({ kind: 'unavailable', reason: 'exit 1' }));
    const cheap = stub('cheap', () => jsonResponse('cheap ok', true));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    const toolTurn = (extra: any[] = []) => ({
      model: 'claude',
      messages: [
        { role: 'user', content: 'hello' },
        { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: {} }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'file body' }] },
        ...extra,
      ],
    });
    try {
      const r1 = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s-adv');
      assert.equal(r1.statusCode, 200);
      assert.equal(r1.headers['x-jev-route'], 'cursor; reason=failover:primary-rate-limited');
      await r1.body.text();

      const r2 = await send(url, toolTurn(), 'cascade-s-adv');
      assert.equal(r2.statusCode, 200);
      assert.equal(r2.headers['x-jev-route'], 'cheap; reason=failover:gemini-unavailable');
      await r2.body.text();

      const r3 = await send(url, toolTurn([
        { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_2', name: 'Read', input: {} }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_2', content: 'more' }] },
      ]), 'cascade-s-adv');
      assert.equal(r3.statusCode, 200);
      assert.equal(r3.headers['x-jev-route'], 'cheap; reason=sticky');
      await r3.body.text();
      assert.equal(primary.hits, 1);
      assert.equal(cheap.hits, 2);
    } finally {
      await proxy.close();
    }
  });

  it('10. primary 529 served by Cursor', async () => {
    const primary = rateLimitedPrimary(529);
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(url, { model: 'claude', messages: [{ role: 'user', content: 'hello' }] }, 'cascade-s529');
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'cursor; reason=failover:primary-rate-limited');
      const data = (await r.body.json()) as any;
      assert.equal(data.content[0].text, 'cursor ok');
      assert.equal(primary.hits, 1);
      assert.equal(cursor.hits, 1);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 0);
    } finally {
      await proxy.close();
    }
  });

  it('11. image-content turn skips Cursor and Gemini straight to cheap', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(
        url,
        {
          model: 'claude',
          messages: [{
            role: 'user',
            content: [
              { type: 'text', text: 'what is in this picture?' },
              { type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'iVBORw0KGgo=' } },
            ],
          }],
        },
        'cascade-s-image',
      );
      assert.equal(r.statusCode, 200);
      assert.equal(r.headers['x-jev-route'], 'cheap; reason=failover:gemini-unavailable');
      const data = (await r.body.json()) as any;
      assert.equal(data.content[0].text, 'cheap ok');
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 1);
    } finally {
      await proxy.close();
    }
  });

  it('image/tool_choice turn skips Cursor and Gemini straight to cheap', async () => {
    const primary = rateLimitedPrimary();
    const cursor = stub('cursor', () => jsonResponse('cursor ok'));
    const gemini = stub('gemini', () => jsonResponse('gemini ok'));
    const cheap = stub('cheap', () => jsonResponse('cheap ok'));
    const { proxy, url } = await createTestServer({ primary, cursor, gemini, cheap });
    try {
      const r = await send(
        url,
        { model: 'claude', tool_choice: { type: 'any' }, messages: [{ role: 'user', content: 'hello' }] },
        'cascade-s6',
      );
      assert.equal(r.statusCode, 200);
      await r.body.text();
      assert.equal(cursor.hits, 0);
      assert.equal(gemini.hits, 0);
      assert.equal(cheap.hits, 1);
    } finally {
      await proxy.close();
    }
  });
});
