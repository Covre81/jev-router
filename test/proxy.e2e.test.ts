import assert from 'node:assert/strict';
import { createServer, type IncomingHttpHeaders, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import { after, before, beforeEach, describe, it } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { HeuristicClassifier } from '../src/classifier/heuristic-classifier.js';
import { loadConfig } from '../src/config.js';
import type { Tier } from '../src/domain/policy.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { OpenAICompatibleProvider } from '../src/providers/openai/provider.js';
import { buildServer } from '../src/proxy/server.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { openTelemetryDb, type TelemetryDb } from '../src/telemetry/db.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';
import type { RouterLog } from '../src/telemetry/schema.js';

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

const anthropicOk: Handler = async (seen, res) => {
  if (seen.method === 'HEAD') return void res.writeHead(200).end();
  res.writeHead(200, { 'content-type': 'text/event-stream', 'anthropic-ratelimit-unified-status': 'allowed' });
  res.write(
    `event: message_start\ndata: {"type":"message_start","message":{"model":"${seen.body.model}","usage":{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":1}}}\n\n`,
  );
  await sleep(50);
  res.write(`event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}\n\n`);
  res.end(`event: message_stop\ndata: {"type":"message_stop"}\n\n`);
};

const anthropic429: Handler = (_seen, res) => {
  res.writeHead(429, { 'content-type': 'application/json', 'retry-after': '30' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: 'quota' } }));
};

const openAiStream: Handler = async (_seen, res) => {
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  res.write('data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n');
  await sleep(200);
  res.write('data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n');
  res.write('data: {"choices":[],"usage":{"prompt_tokens":40,"completion_tokens":5}}\n\n');
  res.end('data: [DONE]\n\n');
};

/** What Ollama sends when it fails mid-generation: content, then EOF with no finish_reason and no [DONE]. */
const openAiCutMidStream: Handler = (_seen, res) => {
  res.writeHead(200, { 'content-type': 'text/event-stream' });
  res.end('data: {"choices":[{"delta":{"content":"Hi"}}]}\n\n');
};


/** Non-streaming chat completion: what the cheap provider returns when Claude Code retries without a stream. */
const openAiJson: Handler = (_seen, res) => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ choices: [{ message: { content: 'Hi' }, finish_reason: 'stop' }], usage: { prompt_tokens: 40, completion_tokens: 5 } }));
};

/** A tool-result continuation: not a fresh conversation, so the router reuses the sticky route. */
const toolLoop = (text: string) => [
  { role: 'user', content: [{ type: 'text', text }] },
  { role: 'assistant', content: [{ type: 'tool_use', id: 'toolu_1', name: 'Read', input: { file_path: '/r/README.md' } }] },
  { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'toolu_1', content: 'Teh readme' }] },
];

const openAi503: Handler = (_seen, res) => {
  res.writeHead(503, { 'content-type': 'application/json' }).end('{"error":{"message":"over capacity"}}');
};

describe('proxy end-to-end (fake Anthropic + fake OpenAI-compatible upstreams)', () => {
  let anthropic: { server: Server; url: string };
  let cheap: { server: Server; url: string };
  let proxy: FastifyInstance;
  let proxyUrl: string;
  let anthropicLog: Seen[] = [];
  let cheapLog: Seen[] = [];
  let anthropicHandler: Handler = anthropicOk;
  let cheapHandler: Handler = openAiStream;
  let db: TelemetryDb;
  let telemetry: SqliteTelemetry;

  before(async () => {
    anthropic = await fakeServer(() => anthropicHandler, anthropicLog);
    cheap = await fakeServer(() => cheapHandler, cheapLog);

    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ROUTER_MIN_CHEAP_PROBABILITY: '0.8', // the heuristic tops out at 0.825
      ANTHROPIC_UPSTREAM_URL: anthropic.url,
      CHEAP_BASE_URL: `${cheap.url}/openai/v1`,
      CHEAP_API_KEY: 'gsk_test',
      FAILOVER_ON_PRIMARY_RATE_LIMIT: 'true',
      LOG_LEVEL: 'fatal',
    });
    const router = new Router(new HeuristicClassifier(), new TtlLruStore<Tier>(100, 60_000), {
      policy: { minCheapProbability: config.router.minCheapProbability, standardRoute: config.router.standardRoute },
      primaryClasses: config.router.primaryClasses,
      cheapContextTokens: 100_000,
      classifierTimeoutMs: 1_000,
      classifierMaxChars: 4_000,
    });
    db = openTelemetryDb(':memory:');
    telemetry = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    proxy = buildServer({
      config,
      router,
      providers: { primary: new AnthropicProvider(config.primary), cheap: new OpenAICompatibleProvider(config.cheap) },
      telemetry,
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  });

  after(async () => {
    await proxy.close();
    await telemetry.close();
    anthropic.server.close();
    cheap.server.close();
  });

  beforeEach(() => {
    anthropicLog.length = 0;
    cheapLog.length = 0;
    anthropicHandler = anthropicOk;
    cheapHandler = openAiStream;
  });

  const claudeCode = (sessionId: string, turn: string | object[], { stream = true } = {}) =>
    request(`${proxyUrl}/v1/messages?beta=true`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': 'sk-ant-client',
        'anthropic-version': '2023-06-01',
        'anthropic-beta': 'some-future-beta',
        'x-claude-code-session-id': sessionId,
        'x-claude-code-request-class': 'main',
      },
      body: JSON.stringify({
        model: 'claude-opus-5-5',
        max_tokens: 64_000,
        stream,
        system: [{ type: 'text', text: 'You are Claude Code' }],
        messages: typeof turn === 'string' ? [{ role: 'user', content: [{ type: 'text', text: turn }] }] : turn,
      }),
    });

  it('serves a simple task from the cheap provider, translated to Anthropic SSE, streamed live', async () => {
    const res = await claudeCode('S-simple', 'fix the typo in the README');
    let firstChunkAt = 0;
    let text = '';
    for await (const chunk of res.body) {
      firstChunkAt ||= Date.now();
      text += chunk.toString();
    }
    const endedAt = Date.now();

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'cheap; reason=classified');
    assert.ok(endedAt - firstChunkAt >= 100, 'first event must be relayed before the cheap stream ends');
    assert.match(text, /event: message_start/);
    assert.match(text, /"text_delta","text":"Hi"/);
    assert.match(text, /"stop_reason":"end_turn"/);
    assert.match(text, /event: message_stop/);

    assert.equal(anthropicLog.length, 0, 'the primary quota is untouched');
    assert.equal(cheapLog[0]?.url, '/openai/v1/chat/completions');
    assert.equal(cheapLog[0]?.headers.authorization, 'Bearer gsk_test');
    assert.equal(cheapLog[0]?.headers['x-api-key'], undefined, 'Anthropic credentials never leak to the cheap provider');
    assert.equal(cheapLog[0]?.body.model, 'gpt-oss:20b-cloud');
  });

  it('sends structural work to Anthropic byte-for-byte', async () => {
    const res = await claudeCode('S-structural', 'refactor the data layer to clean architecture');
    await res.body.text();
    assert.equal(res.headers['x-jev-route'], 'primary; reason=classified');
    assert.equal(res.headers['anthropic-ratelimit-unified-status'], 'allowed');
    assert.equal(cheapLog.length, 0);
    assert.equal(anthropicLog[0]?.body.model, 'claude-opus-5-5');
    assert.equal(anthropicLog[0]?.headers['anthropic-beta'], 'some-future-beta');
    assert.equal(anthropicLog[0]?.headers['x-api-key'], 'sk-ant-client');
  });

  it('fails over to Anthropic when the cheap provider is down, and pins the conversation', async () => {
    cheapHandler = openAi503;
    const res = await claudeCode('S-down', 'fix the typo');
    await res.body.text();
    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'primary; reason=failover:cheap-unavailable');

    const next = await claudeCode('S-down', toolLoop('fix the typo'));
    await next.body.text();
    assert.equal(next.headers['x-jev-route'], 'primary; reason=sticky');
    assert.equal(cheapLog.length, 2, 'one attempt plus one retry');
    assert.equal(anthropicLog.length, 2);
  });

  it("ends a stream cut mid-way with an error event and sends Claude Code's retry to Anthropic", async () => {
    await (await claudeCode('S-cut', 'fix the typo')).body.text();

    cheapHandler = openAiCutMidStream;
    const broken = await claudeCode('S-cut', toolLoop('fix the typo'));
    const text = await broken.body.text();
    assert.equal(broken.headers['x-jev-route'], 'cheap; reason=sticky');
    assert.match(text, /event: error\ndata: \{"type":"error","error":\{"type":"api_error"/);

    // Claude Code 2.1.289 re-sends the same turn as a non-streaming request.
    const retry = await claudeCode('S-cut', toolLoop('fix the typo'), { stream: false });
    await retry.body.text();
    assert.equal(retry.headers['x-jev-route'], 'primary; reason=sticky');
    assert.equal(cheapLog.length, 2, 'the cheap provider is not tried again');
    assert.equal(anthropicLog.length, 1);
  });

  it('re-classifies the retry of a first turn cut mid-stream (ADR rule 4: the pin does not hold)', async () => {
    cheapHandler = openAiCutMidStream;
    await (await claudeCode('S-cut-first', 'fix the typo')).body.text();

    cheapHandler = openAiJson;
    const retry = await claudeCode('S-cut-first', 'fix the typo', { stream: false });
    await retry.body.text();
    assert.equal(retry.headers['x-jev-route'], 'cheap; reason=classified');
  });

  it('fails over to the cheap provider when the primary quota is exhausted (opt-in)', async () => {
    anthropicHandler = anthropic429;
    const res = await claudeCode('S-quota', 'refactor the data layer to clean architecture');
    const text = await res.body.text();
    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'cheap; reason=failover:primary-rate-limited');
    assert.match(text, /"text":"Hi"/);
  });

  it('relays the primary 429 verbatim when the cheap provider cannot help either', async () => {
    anthropicHandler = anthropic429;
    cheapHandler = openAi503;
    const res = await claudeCode('S-both-down', 'refactor everything');
    const body = await res.body.json();
    assert.equal(res.statusCode, 429);
    assert.equal(res.headers['retry-after'], '30');
    assert.deepEqual(body, { type: 'error', error: { type: 'rate_limit_error', message: 'quota' } });
  });

  /** Rows land after the client connection closes; flush and give the close handler a tick. */
  async function logsFor(sessionId: string): Promise<RouterLog[]> {
    for (let i = 0; i < 20; i++) {
      await sleep(10);
      telemetry.flush();
      const rows = (
        db.prepare(`SELECT * FROM router_logs WHERE session_id LIKE ? ORDER BY id`).all(`${sessionId}:%`) as Record<string, any>[]
      ).map(
        (r): RouterLog => ({
          createdAt: new Date(r.created_at),
          sessionId: r.session_id,
          humanPromptHash: r.human_prompt_hash,
          jevDecision: r.jev_decision === null ? null : JSON.parse(r.jev_decision),
          finalProvider: r.final_provider,
          model: r.model,
          requestedModel: r.requested_model,
          routeReason: r.route_reason,
          requestClass: r.request_class,
          httpStatus: r.http_status,
          outcome: r.outcome,
          tokensIn: r.tokens_in,
          tokensOut: r.tokens_out,
          cacheReadTokens: r.cache_read_tokens,
          cacheWriteTokens: r.cache_write_tokens,
          latencyMs: r.latency_ms,
          fallbackTriggered: r.fallback_triggered === 1,
          toolsOffered: r.tools_offered,
          toolCalls: r.tool_calls,
          inspectionMiss: r.inspection_miss === null ? null : r.inspection_miss === 1,
          quotaUtilization: r.quota_utilization,
          upstreamFailure: r.upstream_failure,
        }),
      );
      if (rows.length > 0) return rows;
    }
    return [];
  }

  it('records one telemetry row per exchange, with provider, tokens and JEV verdict', async () => {
    await (await claudeCode('T-cheap', 'fix the typo in the README')).body.text();
    await (await claudeCode('T-primary', 'refactor the data layer to clean architecture')).body.text();

    const [cheapRow] = await logsFor('T-cheap');
    assert.ok(cheapRow);
    assert.equal(cheapRow.finalProvider, 'openai');
    assert.equal(cheapRow.model, 'gpt-oss:20b-cloud');
    assert.equal(cheapRow.requestedModel, 'claude-opus-5-5', 'the baseline prices what Anthropic would have run');
    assert.equal(cheapRow.outcome, 'ok');
    assert.equal(cheapRow.tokensIn, 40);
    assert.equal(cheapRow.tokensOut, 5);
    assert.equal(cheapRow.fallbackTriggered, false);
    assert.match(cheapRow.humanPromptHash ?? '', /^[0-9a-f]{64}$/);
    assert.ok(cheapRow.jevDecision && cheapRow.jevDecision.pSimple >= 0.8);
    assert.ok(cheapRow.latencyMs >= 150, 'latency covers the whole stream');
    assert.equal(cheapRow.toolsOffered, 0);
    assert.equal(cheapRow.toolCalls, 0);
    assert.equal(cheapRow.inspectionMiss, false, 'no tools offered: answering in text is not a miss');

    const [primaryRow] = await logsFor('T-primary');
    assert.ok(primaryRow);
    assert.equal(primaryRow.finalProvider, 'anthropic');
    assert.equal(primaryRow.model, 'claude-opus-5-5');
    assert.equal(primaryRow.tokensIn, 100, 'uncached + cache reads');
    assert.equal(primaryRow.cacheReadTokens, 90);
    assert.equal(primaryRow.cacheWriteTokens, 0);
    assert.equal(primaryRow.tokensOut, 25);
    assert.equal(primaryRow.httpStatus, 200);
    assert.equal(primaryRow.inspectionMiss, null, 'only cheap answers are judged');
  });

  it('flags the fallback when the cheap provider fails and Anthropic answers', async () => {
    cheapHandler = openAi503;
    await (await claudeCode('T-fallback', 'fix the typo')).body.text();
    const [row] = await logsFor('T-fallback');
    assert.ok(row);
    assert.equal(row.finalProvider, 'anthropic');
    assert.equal(row.routeReason, 'failover:cheap-unavailable');
    assert.equal(row.fallbackTriggered, true);
  });

  it('records relayed upstream errors as http_error', async () => {
    anthropicHandler = anthropic429;
    cheapHandler = openAi503;
    await (await claudeCode('T-429', 'refactor everything')).body.text();
    const [row] = await logsFor('T-429');
    assert.ok(row);
    assert.equal(row.httpStatus, 429);
    assert.equal(row.outcome, 'http_error');
  });

  it('records proxy_error when Anthropic is unreachable', async () => {
    anthropicHandler = (_seen, res) => void res.socket?.destroy();
    const res = await claudeCode('T-down', 'refactor the data layer to clean architecture');
    await res.body.text();
    assert.equal(res.statusCode, 502);
    const [row] = await logsFor('T-down');
    assert.ok(row, 'an Anthropic outage must show up in stats');
    assert.equal(row.outcome, 'proxy_error');
    assert.equal(row.httpStatus, 502);
    assert.match(row.upstreamFailure ?? '', /^network: /, 'the cause of the 502 is kept, not just its status');
  });

  it('records client_abort when Claude Code hangs up mid-stream', async () => {
    anthropicHandler = async (_seen, res) => {
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      res.write('event: message_start\ndata: {"type":"message_start","message":{"usage":{"input_tokens":1,"output_tokens":0}}}\n\n');
      await sleep(500);
      res.end('event: message_stop\ndata: {"type":"message_stop"}\n\n');
    };
    const res = await claudeCode('T-abort', 'refactor the data layer to clean architecture');
    for await (const _chunk of res.body) {
      res.body.destroy();
      break;
    }
    const [row] = await logsFor('T-abort');
    assert.ok(row);
    assert.equal(row.outcome, 'client_abort');
  });

  it('passes count_tokens and unknown endpoints straight to Anthropic', async () => {
    anthropicHandler = (_s, res) => void res.writeHead(200, { 'content-type': 'application/json' }).end('{"input_tokens":12}');
    const counted = await request(`${proxyUrl}/v1/messages/count_tokens`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hi' }] }),
    });
    assert.deepEqual(await counted.body.json(), { input_tokens: 12 });

    anthropicHandler = anthropicOk;
    const head = await request(`${proxyUrl}/api/hello`, { method: 'HEAD' });
    await head.body.dump();
    assert.equal(head.statusCode, 200);
    assert.deepEqual(
      anthropicLog.map((s) => s.url),
      ['/v1/messages/count_tokens', '/api/hello'],
    );
    assert.equal(cheapLog.length, 0);
  });
});

describe('inject mode: the proxy holds the Anthropic key and gates clients with PROXY_AUTH_TOKEN', () => {
  const PROXY_TOKEN = 'proxy-token-0123456789';
  let anthropic: { server: Server; url: string };
  let proxy: FastifyInstance;
  let proxyUrl: string;
  const anthropicLog: Seen[] = [];

  before(async () => {
    anthropic = await fakeServer(() => anthropicOk, anthropicLog);
    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ROUTER_MIN_CHEAP_PROBABILITY: '0.8', // the heuristic tops out at 0.825
      ANTHROPIC_UPSTREAM_URL: anthropic.url,
      UPSTREAM_AUTH_MODE: 'inject',
      ANTHROPIC_API_KEY: 'sk-ant-proxy-test',
      PROXY_AUTH_TOKEN: PROXY_TOKEN,
      CHEAP_API_KEY: 'unused',
      LOG_LEVEL: 'fatal',
    });
    const router = new Router(new HeuristicClassifier(), new TtlLruStore<Tier>(100, 60_000), {
      policy: { minCheapProbability: config.router.minCheapProbability, standardRoute: config.router.standardRoute },
      primaryClasses: config.router.primaryClasses,
      cheapContextTokens: 100_000,
      classifierTimeoutMs: 1_000,
      classifierMaxChars: 4_000,
    });
    proxy = buildServer({
      config,
      router,
      providers: { primary: new AnthropicProvider(config.primary), cheap: new OpenAICompatibleProvider(config.cheap) },
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  });

  after(async () => {
    await proxy.close();
    anthropic.server.close();
  });

  beforeEach(() => void (anthropicLog.length = 0));

  // Structural work stays primary, so the request reaches the fake Anthropic.
  const send = (headers: Record<string, string>) =>
    request(`${proxyUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'anthropic-version': '2023-06-01', ...headers },
      body: JSON.stringify({
        model: 'claude-opus-5-5',
        max_tokens: 1024,
        stream: true,
        messages: [{ role: 'user', content: 'refactor the data layer to clean architecture' }],
      }),
    });

  it('rejects a request without the proxy token before it reaches Anthropic', async () => {
    const res = await send({});
    assert.equal(res.statusCode, 401);
    assert.deepEqual(await res.body.json(), {
      type: 'error',
      error: { type: 'authentication_error', message: 'invalid proxy token' },
    });
    assert.equal(anthropicLog.length, 0);
  });

  it('rejects a wrong token, including a client-side Anthropic key', async () => {
    for (const headers of [{ 'x-api-key': 'proxy-token-WRONG-6789' }, { 'x-api-key': 'sk-ant-client' }]) {
      const res = await send(headers);
      await res.body.dump();
      assert.equal(res.statusCode, 401);
    }
    assert.equal(anthropicLog.length, 0);
  });

  it('replaces the client credential with the proxy key, from either header', async () => {
    for (const headers of [{ 'x-api-key': PROXY_TOKEN }, { authorization: `Bearer ${PROXY_TOKEN}` }]) {
      const res = await send(headers);
      await res.body.text();
      assert.equal(res.statusCode, 200);
    }
    assert.equal(anthropicLog.length, 2);
    for (const seen of anthropicLog) {
      assert.equal(seen.headers['x-api-key'], 'sk-ant-proxy-test');
      assert.equal(seen.headers.authorization, undefined, 'the proxy token never reaches Anthropic');
    }
  });

  it('leaves /healthz open', async () => {
    const res = await request(`${proxyUrl}/healthz`);
    assert.equal(res.statusCode, 200);
    const body = (await res.body.json()) as Record<string, unknown>;
    assert.equal(body['ok'], true);
    assert.equal(body['cheap'], 'unknown', 'no health check wired');
  });
});

describe('body limit', () => {
  let upstream: { server: Server; url: string };
  let proxy: FastifyInstance;
  let proxyUrl: string;
  const seen: Seen[] = [];

  before(async () => {
    upstream = await fakeServer(() => anthropicOk, seen);
    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ROUTER_MIN_CHEAP_PROBABILITY: '0.8', // the heuristic tops out at 0.825
      CHEAP_API_KEY: 'unused',
      ANTHROPIC_UPSTREAM_URL: upstream.url,
      BODY_LIMIT_BYTES: '2000',
      LOG_LEVEL: 'fatal',
    });
    const router = new Router(new HeuristicClassifier(), new TtlLruStore<Tier>(100, 60_000), {
      policy: { minCheapProbability: config.router.minCheapProbability, standardRoute: config.router.standardRoute },
      primaryClasses: config.router.primaryClasses,
      cheapContextTokens: 100_000,
      classifierTimeoutMs: 1_000,
      classifierMaxChars: 4_000,
    });
    proxy = buildServer({
      config,
      router,
      providers: { primary: new AnthropicProvider(config.primary), cheap: new OpenAICompatibleProvider(config.cheap) },
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  });

  after(async () => {
    await proxy.close();
    upstream.server.close();
  });

  it('answers an oversized body with 413, not an upstream error', async () => {
    const res = await request(`${proxyUrl}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ model: 'claude-opus-5-5', max_tokens: 10, messages: [{ role: 'user', content: 'x'.repeat(5_000) }] }),
    });
    const body = (await res.body.json()) as { error: { type: string } };
    assert.equal(res.statusCode, 413);
    assert.equal(body.error.type, 'request_too_large');
    assert.equal(seen.length, 0, 'nothing reaches Anthropic');
  });
});
