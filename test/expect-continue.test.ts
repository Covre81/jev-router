import assert from 'node:assert/strict';
import { createServer, type Server, type IncomingHttpHeaders, type ServerResponse } from 'node:http';
import * as http from 'node:http';
import type { AddressInfo } from 'node:net';
import { setTimeout as sleep } from 'node:timers/promises';
import { after, before, beforeEach, describe, it } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request, Dispatcher } from 'undici';

import { HeuristicClassifier } from '../src/classifier/heuristic-classifier.js';
import { loadConfig } from '../src/config.js';
import type { Tier } from '../src/domain/policy.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { OpenAICompatibleProvider } from '../src/providers/openai/provider.js';
import { buildServer } from '../src/proxy/server.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { forwardableHeaders } from '../src/proxy/headers.js';

interface Seen {
  method: string;
  url: string;
  headers: IncomingHttpHeaders;
  raw: string;
  body: any;
}

type Handler = (seen: Seen, res: ServerResponse) => Promise<void> | void;

async function fakeServer(handler: () => Handler, log: Seen[], port = 0): Promise<{ server: Server; url: string; port: number }> {
  const server = createServer(async (req, res) => {
    const parts: Buffer[] = [];
    for await (const c of req) parts.push(c as Buffer);
    const raw = Buffer.concat(parts).toString('utf8');
    const seen = { method: req.method!, url: req.url!, headers: req.headers, raw, body: raw ? JSON.parse(raw) : undefined };
    log.push(seen);
    await handler()(seen, res);
  });
  await new Promise<void>((r) => server.listen(port, '127.0.0.1', r));
  const addr = server.address() as AddressInfo;
  return { server, url: `http://127.0.0.1:${addr.port}`, port: addr.port };
}

const anthropicOk: Handler = async (seen, res) => {
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end('{"ok":true}');
};

describe('expect: 100-continue and wedged pool fixes', () => {
  describe('unit', () => {
    it('a) forwardableHeaders drops expect and keeps anthropic-* and x-claude-code-* headers', () => {
      const headers = {
        'expect': '100-continue',
        'Expect': '100-continue',
        'anthropic-version': '2023-06-01',
        'x-claude-code-session-id': '123',
        'content-type': 'application/json'
      };
      const forwarded = forwardableHeaders(headers);
      assert.equal(forwarded['expect'], undefined);
      assert.equal(forwarded['anthropic-version'], '2023-06-01');
      assert.equal(forwarded['x-claude-code-session-id'], '123');
      assert.equal(forwarded['content-type'], 'application/json');
    });

    it('f) AnthropicProvider with wedged dispatcher times out and recovers', async () => {
      class WedgedDispatcher extends Dispatcher {
        override dispatch(options: Dispatcher.DispatchOptions, handlers: Dispatcher.DispatchHandler): boolean {
          return true; // Wedged: do not call handlers
        }
      }

      const provider = new AnthropicProvider({
        baseUrl: 'http://127.0.0.1:0',
        timeoutMs: 200,
        authMode: 'passthrough',
        apiKey: undefined,
        dispatcher: new WedgedDispatcher()
      });

      const start = Date.now();
      const res = await provider.send({ method: 'POST', url: '/v1/messages', headers: {}, signal: new AbortController().signal, rawBody: undefined, body: undefined });
      const duration = Date.now() - start;

      assert.equal(res.kind, 'unavailable');
      if (res.kind === 'unavailable') {
        assert.match(res.reason, /upstream response headers deadline exceeded/);
      }
      assert.ok(duration < 1000, `duration ${duration} ms should be around 200ms`);

      const fake = await fakeServer(() => anthropicOk, []);
      const provider2 = new AnthropicProvider({
        baseUrl: fake.url,
        timeoutMs: 200,
        authMode: 'passthrough',
        apiKey: undefined,
      });
      const res2 = await provider2.send({ method: 'POST', url: '/v1/messages', headers: {}, signal: new AbortController().signal, rawBody: undefined, body: undefined });
      assert.equal(res2.kind, 'response', res2.kind === 'unavailable' ? res2.reason : '');
      fake.server.close();
    });
  });

  describe('e2e', () => {
    let upstream: { server: Server; url: string; port: number };
    let proxy: FastifyInstance;
    let proxyUrl: string;
    let seenLogs: Seen[] = [];
    let upstreamHandler: Handler = anthropicOk;

    before(async () => {
      upstream = await fakeServer(() => upstreamHandler, seenLogs);
      const config = loadConfig({
        CLASSIFIER: 'heuristic',
        ROUTER_MIN_CHEAP_PROBABILITY: '0.8',
        ANTHROPIC_UPSTREAM_URL: upstream.url,
        CHEAP_API_KEY: 'unused',
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
      });
      await proxy.listen({ host: '127.0.0.1', port: 0 });
      proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
    });

    after(async () => {
      await proxy.close();
      upstream.server.close();
    });

    beforeEach(() => {
      seenLogs.length = 0;
      upstreamHandler = anthropicOk;
    });

    const normalRequest = async () => {
      const start = Date.now();
      const res = await request(`${proxyUrl}/v1/messages`, {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'sk-ant-client',
          'anthropic-version': '2023-06-01',
          'x-claude-code-request-class': 'main',
        },
        body: JSON.stringify({
          model: 'claude-opus-5-5',
          max_tokens: 64_000,
          messages: [{ role: 'user', content: 'test' }],
        }),
      });
      await res.body.text();
      return { status: res.statusCode, duration: Date.now() - start };
    };

    it('b) request with expect: 100-continue succeeds and upstream does not see expect', async () => {
      const req = http.request(proxyUrl + '/v1/messages', {
        method: 'POST',
        headers: {
          'expect': '100-continue',
          'content-type': 'application/json',
          'x-api-key': 'sk-ant-client',
          'anthropic-version': '2023-06-01',
          'x-claude-code-request-class': 'main',
        },
      });

      const body = JSON.stringify({
        model: 'claude-opus-5-5',
        max_tokens: 64_000,
        messages: [{ role: 'user', content: 'hello' }],
      });

      const [res] = await Promise.all([
        new Promise<http.IncomingMessage>((resolve) => req.on('response', resolve)),
        new Promise<void>((resolve) => {
          req.on('continue', () => {
            req.write(body);
            req.end();
            resolve();
          });
        })
      ]);

      const resBodyChunks: Buffer[] = [];
      for await (const chunk of res) resBodyChunks.push(chunk as Buffer);
      
      assert.equal(res.statusCode, 200);
      assert.equal(seenLogs.length, 1);
      assert.equal(seenLogs[0]?.headers['expect'], undefined);
    });

    it('c) three normal requests succeed quickly after expect', async () => {
      for (let i = 0; i < 3; i++) {
        const { status, duration } = await normalRequest();
        assert.equal(status, 200);
        assert.ok(duration < 1000, `duration ${duration} ms should be under 1s`);
      }
    });

    it('d) after a forced 502, normal request succeeds quickly', async () => {
      upstream.server.close();
      const res502 = await normalRequest();
      assert.equal(res502.status, 502);

      upstream = await fakeServer(() => upstreamHandler, seenLogs, upstream.port);
      const resOk = await normalRequest();
      assert.equal(resOk.status, 200);
      assert.ok(resOk.duration < 1000);
    });

    it('e) client destroys socket mid-body, normal request succeeds', async () => {
      const req = http.request(proxyUrl + '/v1/messages', {
        method: 'POST',
        headers: {
          'content-type': 'application/json',
          'x-api-key': 'sk-ant-client',
          'anthropic-version': '2023-06-01',
          'x-claude-code-request-class': 'main',
        },
      });
      req.on('error', () => {}); // swallow the hang up error

      req.write('{"model":"claude-opus-5-5","messages":[{"role":"user","content":"half');
      req.destroy();
      await sleep(50);

      const res = await normalRequest();
      assert.equal(res.status, 200);
      assert.ok(res.duration < 1000);
    });
  });
});
