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
import { buildServer, isQuotaProbe, failureCode } from '../src/proxy/server.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { openTelemetryDb, type TelemetryDb } from '../src/telemetry/db.js';
import { SqliteTelemetry } from '../src/telemetry/recorder.js';

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
    `event: message_start\ndata: {"type":"message_start","message":{"model":"${seen.body?.model ?? 'claude-opus-5'}","usage":{"input_tokens":10,"cache_read_input_tokens":90,"output_tokens":1}}}\n\n`,
  );
  await sleep(10);
  res.write(`event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}\n\n`);
  res.end(`event: message_stop\ndata: {"type":"message_stop"}\n\n`);
};

const anthropic429: Handler = (_seen, res) => {
  res.writeHead(429, { 
    'content-type': 'application/json', 
    'retry-after': '30',
    'anthropic-ratelimit-unified-status': 'rejected',
    'anthropic-ratelimit-unified-7d-utilization': '1.0'
  });
  res.end(JSON.stringify({ type: 'error', error: { type: 'rate_limit_error', message: 'quota' } }));
};

const anthropic529: Handler = (_seen, res) => {
  res.writeHead(529, { 
    'content-type': 'application/json', 
    'retry-after': '30',
    'anthropic-ratelimit-unified-status': 'rejected',
    'anthropic-ratelimit-unified-7d-utilization': '1.0'
  });
  res.end(JSON.stringify({ type: 'error', error: { type: 'overloaded_error', message: 'overloaded' } }));
};

describe('real-use fixes (A and C)', () => {
  let anthropic: { server: Server; url: string };
  let proxy: FastifyInstance;
  let proxyUrl: string;
  let anthropicLog: Seen[] = [];
  let anthropicHandler: Handler = anthropicOk;
  let db: TelemetryDb;
  let telemetry: SqliteTelemetry;

  async function startServer(env: NodeJS.ProcessEnv) {
    if (proxy) await proxy.close();
    if (telemetry) await telemetry.close();
    
    const config = loadConfig(env);
    const router = new Router(new HeuristicClassifier(), new TtlLruStore<Tier>(100, 60_000), {
      policy: { minCheapProbability: config.router.minCheapProbability },
      primaryClasses: config.router.primaryClasses,
      classifierTimeoutMs: 1_000,
      classifierMaxChars: 4_000,
    });
    db = openTelemetryDb(':memory:');
    telemetry = new SqliteTelemetry(db, { flushIntervalMs: 60_000 });
    proxy = buildServer({
      config,
      router,
      providers: { primary: new AnthropicProvider(config.primary) },
      telemetry,
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  }

  before(async () => {
    anthropic = await fakeServer(() => anthropicHandler, anthropicLog);

    await startServer({
      CLASSIFIER: 'heuristic',
      ANTHROPIC_UPSTREAM_URL: anthropic.url,
      FAILOVER_ON_PRIMARY_RATE_LIMIT: 'true',
      LOG_LEVEL: 'fatal',
    });
  });

  after(async () => {
    await proxy.close();
    await telemetry.close();
    anthropic.server.close();
  });

  beforeEach(() => {
    anthropicLog.length = 0;
    anthropicHandler = anthropicOk;
  });

  const claudeCode = (sessionId: string, maxTokens: number, classHdr: string, text = 'fix the typo in the README') =>
    request(`${proxyUrl}/v1/messages?beta=true`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-api-key': 'sk-ant-client',
        'anthropic-version': '2023-06-01',
        'x-claude-code-session-id': sessionId,
        'x-claude-code-request-class': classHdr,
      },
      body: JSON.stringify({
        model: 'claude-opus-5-5',
        max_tokens: maxTokens,
        stream: true,
        messages: [{ role: 'user', content: [{ type: 'text', text }] }],
      }),
    });

  async function logsFor(sessionId: string): Promise<any[]> {
    for (let i = 0; i < 20; i++) {
      await sleep(10);
      telemetry.flush();
      const rows = db.prepare(`SELECT * FROM router_logs WHERE session_id LIKE ? ORDER BY id`).all(`${sessionId}:%`) as any[];
      if (rows.length > 0) return rows;
    }
    return [];
  }

  it('1) probe 429 passes through', async () => {
    anthropicHandler = anthropic429;
    const res = await claudeCode('probe-429', 1, 'auxiliary', 'probe');
    const bodyText = await res.body.text();
    
    assert.equal(res.statusCode, 429);
    assert.equal(res.headers['retry-after'], '30');
    assert.equal(res.headers['anthropic-ratelimit-unified-status'], 'rejected');
    assert.equal(res.headers['anthropic-ratelimit-unified-7d-utilization'], '1.0');
    assert.deepEqual(JSON.parse(bodyText), { type: 'error', error: { type: 'rate_limit_error', message: 'quota' } });

    let row;
    for (let i = 0; i < 20; i++) {
      await sleep(10);
      telemetry.flush();
      const rows = db.prepare(`SELECT * FROM router_logs WHERE request_class = 'auxiliary' ORDER BY id DESC`).all() as any[];
      if (rows.length > 0) { row = rows[0]; break; }
    }

    assert.ok(row);
    assert.equal(row.http_status, 429);
    assert.equal(row.upstream_status, 429);
    assert.ok(row.upstream_ratelimit_headers.includes('anthropic-ratelimit-unified-status'));
    assert.ok(row.upstream_ratelimit_headers.includes('anthropic-ratelimit-unified-7d-utilization'));
    assert.ok(!row.upstream_ratelimit_headers.includes('1.0'), 'no header value in column');
    assert.ok(!row.upstream_ratelimit_headers.includes('rejected'), 'no header value in column');
  });

  it('2) probe 529 passes through', async () => {
    anthropicHandler = anthropic529;
    const res = await claudeCode('probe-529', 1, 'main', 'probe');
    const bodyText = await res.body.text();
    
    assert.equal(res.statusCode, 529);
    assert.deepEqual(JSON.parse(bodyText), { type: 'error', error: { type: 'overloaded_error', message: 'overloaded' } });

    const [row] = await logsFor('probe-529');
    assert.ok(row);
    assert.equal(row.http_status, 529);
    assert.equal(row.upstream_status, 529);
    assert.ok(row.upstream_ratelimit_headers.includes('anthropic-ratelimit-unified-status'));
  });

  describe('unit tests', () => {
    it('isQuotaProbe', () => {
      assert.equal(isQuotaProbe({ max_tokens: 1 } as any), true);
      assert.equal(isQuotaProbe({ max_tokens: 2 } as any), false);
      assert.equal(isQuotaProbe(undefined), false);
    });

    it('failureCode', () => {
      assert.equal(failureCode('HTTP 500: {...}'), 'HTTP 500');
      assert.equal(failureCode('network: connect ECONNREFUSED'), 'network');
      assert.equal(failureCode('network: upstream response headers deadline exceeded'), 'timeout');
      assert.equal(failureCode('not translatable: x'), 'not translatable');
      assert.equal(failureCode('bad response: y'), 'bad response');
    });
  });
});
