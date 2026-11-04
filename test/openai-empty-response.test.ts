import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, beforeEach, describe, it } from 'node:test';
import { request } from 'undici';
import { loadConfig } from '../src/config.js';
import { OpenAICompatibleProvider } from '../src/providers/openai/provider.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { buildServer } from '../src/proxy/server.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

// Regression: a reasoning model on the local Ollama spent max_tokens thinking and
// returned no content; the provider relayed `content: []` as a 200 and the cascade
// never ran, so the client got an empty answer logged as a success.

type Reply = { stream: false; content: string } | { stream: true; deltas: string[]; truncated?: boolean };

async function listen(server: Server): Promise<string> {
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  return `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
}

describe('cheap provider: empty answer is a failure, not a response', () => {
  let ollama: Server;
  let ollamaUrl: string;
  let anthropic: Server;
  let anthropicUrl: string;
  let reply: Reply;
  let ollamaHits = 0;
  let anthropicHits = 0;

  before(async () => {
    ollama = createServer((req, res) => {
      ollamaHits++;
      req.resume();
      req.on('end', () => {
        if (!reply.stream) {
          res.writeHead(200, { 'content-type': 'application/json' });
          res.end(JSON.stringify({
            choices: [{ index: 0, message: { role: 'assistant', content: reply.content, reasoning: 'thinking...' }, finish_reason: reply.content ? 'stop' : 'length' }],
            usage: { prompt_tokens: 90, completion_tokens: 64 },
          }));
          return;
        }
        res.writeHead(200, { 'content-type': 'text/event-stream' });
        const chunk = (delta: object, finish: string | null) =>
          `data: ${JSON.stringify({ choices: [{ index: 0, delta, finish_reason: finish }] })}\n\n`;
        res.write(chunk({ role: 'assistant', reasoning: 'thinking...' }, null));
        if (reply.truncated) return void res.end(); // upstream dies mid-thought: no finish_reason, no [DONE]
        for (const text of reply.deltas) res.write(chunk({ content: text }, null));
        res.write(chunk({}, reply.deltas.length ? 'stop' : 'length'));
        res.end('data: [DONE]\n\n');
      });
    });
    ollamaUrl = await listen(ollama);

    anthropic = createServer((req, res) => {
      anthropicHits++;
      req.resume();
      req.on('end', () => {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ type: 'message', role: 'assistant', content: [{ type: 'text', text: 'claude ok' }], stop_reason: 'end_turn', usage: {} }));
      });
    });
    anthropicUrl = await listen(anthropic);
  });

  after(() => {
    ollama.close();
    anthropic.close();
  });

  beforeEach(() => {
    ollamaHits = 0;
    anthropicHits = 0;
  });

  const provider = () => new OpenAICompatibleProvider({ baseUrl: ollamaUrl, apiKey: 'k', model: 'local', maxOutputTokens: 8192, timeoutMs: 5_000 });
  const send = (stream: boolean) =>
    provider().send({
      body: { model: 'claude', max_tokens: 64, stream, messages: [{ role: 'user', content: 'quanto é 2 + 2?' }] },
      signal: new AbortController().signal,
    } as any);

  it('non-stream: no content -> unavailable', async () => {
    reply = { stream: false, content: '' };
    const res = await send(false);
    assert.equal(res.kind, 'unavailable');
  });

  it('stream: reasoning only, no content block -> unavailable', async () => {
    reply = { stream: true, deltas: [] };
    const res = await send(true);
    assert.equal(res.kind, 'unavailable');
  });

  it('stream that breaks before any content block -> unavailable, not a 200 carrying an error event', async () => {
    reply = { stream: true, deltas: [], truncated: true };
    const res = await send(true);
    assert.equal(res.kind, 'unavailable');
  });

  it('stream with text still relays every event', async () => {
    reply = { stream: true, deltas: ['4'] };
    const res = await send(true);
    assert.equal(res.kind, 'response');
    if (res.kind !== 'response') return;
    let raw = '';
    for await (const c of res.body) raw += c.toString();
    const events = [...raw.matchAll(/^event: (\S+)/gm)].map((m) => m[1]);
    assert.deepEqual(events, ['message_start', 'content_block_start', 'content_block_delta', 'content_block_stop', 'message_delta', 'message_stop']);
    assert.match(raw, /"text":"4"/);
  });

  it('server: empty cheap answer falls through to Claude instead of being served', async () => {
    reply = { stream: false, content: '' };
    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ANTHROPIC_UPSTREAM_URL: anthropicUrl,
      ANTHROPIC_API_KEY: 'test-key',
      ARGON_TIER: 'off',
      LOG_LEVEL: 'fatal',
      TELEMETRY_ENABLED: 'false',
      CHEAP_ROUTE: 'fallback',
      CHEAP_BASE_URL: ollamaUrl,
      CHEAP_API_KEY: 'k',
      CHEAP_MODEL: 'local',
    });
    const simple = { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, textOnly: 0.95 };
    const router = new Router({ name: 'scripted', classify: async () => simple } as any, new TtlLruStore<any>(100, 60_000), {
      policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: false, argonEnabled: false },
      primaryClasses: new Set(['auxiliary']),
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1500,
    } as any);
    const logs: any[] = [];
    const proxy = buildServer({
      config,
      router,
      telemetry: { record: (entry: any) => logs.push(entry), close: async () => {} },
      providers: { primary: new AnthropicProvider(config.primary), cheap: new OpenAICompatibleProvider(config.cheap!) },
    } as any);
    await proxy.listen({ port: 0, host: '127.0.0.1' });
    try {
      const url = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
      const res = await request(`${url}/v1/messages`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ model: 'claude', max_tokens: 64, messages: [{ role: 'user', content: 'quanto é 2 + 2?' }] }),
      });
      const text = await res.body.text();
      assert.equal(ollamaHits, 1);
      assert.equal(anthropicHits, 1);
      assert.match(String(res.headers['x-jev-route']), /^primary/);
      assert.match(text, /claude ok/);
      // The Ollama hop's failure must be measurable, not hidden behind Flash's reason.
      await new Promise((r) => setTimeout(r, 50));
      const turn = logs.find((l) => l.requestClass !== 'auxiliary' && l.finalProvider === 'anthropic');
      assert.ok(turn, 'the turn served by Claude is logged');
      assert.equal(turn.upstreamFailure, 'cheap:empty');
    } finally {
      await proxy.close();
    }
  });
});
