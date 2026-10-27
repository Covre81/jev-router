import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { describe, it } from 'node:test';
import { buildServer } from '../src/proxy/server.js';
import { loadConfig } from '../src/config.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('SSE stream integrity under keepalive pings', () => {
  const baseConfig = loadConfig({
    CHEAP_API_KEY: 'k',
    CLASSIFIER: 'heuristic',
    KEEPALIVE_INTERVAL_MS: '20',
  });

  const createRouter = () =>
    new Router(
      { name: 'mock', classify: async () => ({ simple: 0, standard: 0, structural: 1 }) } as any,
      new TtlLruStore(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, standardEnabled: false, argonEnabled: false },
        primaryClasses: new Set(['auxiliary']),
        classifierTimeoutMs: 1000,
        classifierMaxChars: 4000,
      },
    );

  it('(a) all hops fail after pings -> client receives event: error', async () => {
    const fakePrimary = {
      name: 'anthropic',
      send: async () => {
        // Delay long enough for keepalive ping to be sent
        await sleep(50);
        return { kind: 'unavailable' as const, reason: 'network timeout' };
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter(),
      providers: { primary: fakePrimary as any },
      timeouts: { keepaliveIntervalMs: 20 },
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hello' }], stream: true },
    });

    const body = res.body;
    assert.ok(body.includes('event: ping'), 'Expected keepalive ping in stream');
    assert.ok(body.includes('event: error'), 'Expected event: error in stream');
    assert.ok(body.includes('"type":"api_error"'), 'Expected api_error type in payload');
    assert.ok(body.includes('network timeout'), 'Expected failure reason in payload');
    // Ensure every non-empty block has an event name
    const events = body.split('\n\n').filter((b) => b.trim().length > 0);
    for (const ev of events) {
      assert.ok(ev.startsWith('event: '), `Every block must start with "event: ", saw: ${ev}`);
    }
  });

  it('(b) hop fails then Claude upstream returns 429/500 after pings -> event: error with status info', async () => {
    const fakePrimary = {
      name: 'anthropic',
      send: async () => {
        await sleep(50);
        const errPayload = JSON.stringify({
          type: 'error',
          error: {
            type: 'rate_limit_error',
            message: 'Number of request tokens has exceeded your daily rate limit',
          },
        });
        return {
          kind: 'response' as const,
          status: 429,
          headers: { 'content-type': 'application/json' },
          body: Readable.from([Buffer.from(errPayload)]) as any,
        };
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter(),
      providers: { primary: fakePrimary as any },
      timeouts: { keepaliveIntervalMs: 20 },
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hello' }], stream: true },
    });

    const body = res.body;
    assert.ok(body.includes('event: ping'), 'Expected keepalive ping in stream');
    assert.ok(body.includes('event: error'), 'Expected event: error in stream');
    assert.ok(body.includes('rate_limit_error'), 'Expected rate_limit_error from upstream');
    assert.ok(body.includes('exceeded your daily rate limit'), 'Expected message from upstream');

    // Verify it is not written as bare JSON outside of SSE
    const events = body.split('\n\n').filter((b) => b.trim().length > 0);
    for (const ev of events) {
      assert.ok(ev.startsWith('event: '), `Every block must start with "event: ", saw: ${ev}`);
    }
  });

  it('(c) success after pings -> complete stream with message_stop', async () => {
    const fakePrimary = {
      name: 'anthropic',
      send: async () => {
        await sleep(50);
        const chunks = [
          'event: message_start\ndata: {"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","content":[],"model":"claude-opus-5-5"}}\n\n',
          'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
          'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello world"}}\n\n',
          'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
          'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":3}}\n\n',
          'event: message_stop\ndata: {"type":"message_stop"}\n\n',
        ];
        return {
          kind: 'response' as const,
          status: 200,
          headers: { 'content-type': 'text/event-stream' },
          body: (async function* () {
            for (const c of chunks) yield c;
          })() as any,
        };
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter(),
      providers: { primary: fakePrimary as any },
      timeouts: { keepaliveIntervalMs: 20 },
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hello' }], stream: true },
    });

    const body = res.body;
    assert.ok(body.includes('event: ping'), 'Expected keepalive ping in stream');
    assert.ok(body.includes('event: message_start'), 'Expected message_start in stream');
    assert.ok(body.includes('Hello world'), 'Expected stream text');
    assert.ok(body.includes('event: message_stop'), 'Expected message_stop in stream');

    const events = body.split('\n\n').filter((b) => b.trim().length > 0);
    assert.ok(events[events.length - 1]?.startsWith('event: message_stop'));
  });
});
