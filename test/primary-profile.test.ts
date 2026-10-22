import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, beforeEach, describe, it } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { loadConfig } from '../src/config.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { OpenAICompatibleProvider } from '../src/providers/openai/provider.js';
import { buildServer } from '../src/proxy/server.js';
import type { ClassificationInput, ComplexityClassifier } from '../src/classifier/classifier.js';
import type { ComplexityDistribution } from '../src/domain/complexity.js';
import { primaryProfile, type Tier } from '../src/domain/policy.js';
import { MessagesBody } from '../src/routing/messages-body.js';
import { withPrimaryProfile } from '../src/routing/primary-profile.js';
import { Router, type RequestContext } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

const OPUS = 'claude-opus-5-5';
const SONNET = 'claude-sonnet-5-5';

describe('primaryProfile', () => {
  it('keeps risky work on Opus at high effort, however simple it looks', () => {
    assert.deepEqual(primaryProfile({ simple: 0.96, standard: 0.03, structural: 0.01, risk: 0.94 }), { model: OPUS, effort: 'high' });
  });
  it('gives structural work Opus high', () => {
    assert.deepEqual(primaryProfile({ simple: 0.03, standard: 0.12, structural: 0.85 }), { model: OPUS, effort: 'high' });
  });
  it('leaves mixed work on the client default (Opus medium)', () => {
    assert.deepEqual(primaryProfile({ simple: 0.37, standard: 0.31, structural: 0.32 }), { model: OPUS, effort: 'medium' });
  });
  it('sends standard work to Sonnet medium', () => {
    assert.deepEqual(primaryProfile({ simple: 0.4, standard: 0.5, structural: 0.1 }), { model: SONNET, effort: 'medium' });
  });
  it('sends simple work that still reached Claude to Sonnet low', () => {
    assert.deepEqual(primaryProfile({ simple: 0.75, standard: 0.2, structural: 0.05 }), { model: SONNET, effort: 'low' });
  });
});

describe('withPrimaryProfile', () => {
  const raw = (o: object) => Buffer.from(JSON.stringify(o));
  const base = { model: OPUS, max_tokens: 128000, thinking: { type: 'adaptive' }, output_config: { effort: 'medium' }, messages: [] };

  it('rewrites model and effort of the client default, keeping every other field', () => {
    const out = withPrimaryProfile(raw(base), { model: SONNET, effort: 'low' });
    assert.ok(out);
    assert.deepEqual(JSON.parse(out.toString()), { ...base, model: SONNET, output_config: { effort: 'low' } });
  });
  it('leaves a suffixed id alone ([1m] is a client alias, not a public model id)', () => {
    assert.equal(withPrimaryProfile(raw({ ...base, model: `${OPUS}[1m]` }), { model: SONNET, effort: 'low' }), undefined);
  });
  it('forwards a malformed body untouched instead of throwing', () => {
    assert.equal(withPrimaryProfile(Buffer.from('{"model":'), { model: SONNET, effort: 'low' }), undefined);
    assert.equal(withPrimaryProfile(Buffer.from('null'), { model: SONNET, effort: 'low' }), undefined);
  });
  it('leaves an explicit user choice alone (/model, /effort)', () => {
    assert.equal(withPrimaryProfile(raw({ ...base, model: SONNET }), { model: OPUS, effort: 'high' }), undefined);
    assert.equal(withPrimaryProfile(raw({ ...base, output_config: { effort: 'xhigh' } }), { model: SONNET, effort: 'low' }), undefined);
  });
  it('leaves fast mode alone', () => {
    assert.equal(withPrimaryProfile(raw({ ...base, speed: 'fast' }), { model: SONNET, effort: 'low' }), undefined);
  });
  it('is a no-op when the profile is the default', () => {
    assert.equal(withPrimaryProfile(raw(base), { model: OPUS, effort: 'medium' }), undefined);
  });
});

class ScriptedClassifier implements ComplexityClassifier {
  readonly name = 'scripted';
  constructor(private readonly next: () => ComplexityDistribution) {}
  classify(_input: ClassificationInput): Promise<ComplexityDistribution> {
    return Promise.resolve(this.next());
  }
}

const SIMPLE: ComplexityDistribution = { simple: 0.92, standard: 0.06, structural: 0.02 };
const STANDARD: ComplexityDistribution = { simple: 0.4, standard: 0.5, structural: 0.1 };
const STRUCTURAL: ComplexityDistribution = { simple: 0.03, standard: 0.12, structural: 0.85 };

function setup(script: () => ComplexityDistribution, primaryProfiles = true) {
  return new Router(new ScriptedClassifier(script), new TtlLruStore<Tier>(100, 60_000), {
    policy: { minCheapProbability: 0.8, standardRoute: 'primary' },
    primaryClasses: new Set(['auxiliary', 'compaction']),
    cheapContextTokens: 100_000,
    classifierTimeoutMs: 1_000,
    classifierMaxChars: 4_000,
    primaryProfiles,
  });
}

const user = (text: string) => ({ role: 'user', content: [{ type: 'text', text }] });
const assistantToolUse = { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Read', input: {} }] };
const toolResult = { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }] };
const assistantText = { role: 'assistant', content: [{ type: 'text', text: 'done' }] };

function ctx(messages: unknown[], overrides: Partial<RequestContext> = {}): RequestContext {
  return {
    body: MessagesBody.parse({ model: OPUS, messages }),
    rawByteLength: 30_000,
    sessionId: 'S1',
    agentId: undefined,
    requestClass: 'main',
    contextCompacted: false,
    ...overrides,
  };
}

describe('Router primary profile', () => {
  it('picks a profile when a session starts on Claude and keeps it for the whole session', async () => {
    let next = STANDARD;
    const router = setup(() => next);
    const first = await router.decide(ctx([user('refatora esse modulo')]));
    assert.equal(first.route, 'primary');
    assert.deepEqual(first.primaryProfile, { model: SONNET, effort: 'medium' });

    const loop = await router.decide(ctx([user('refatora esse modulo'), assistantToolUse, toolResult]));
    assert.deepEqual(loop.primaryProfile, { model: SONNET, effort: 'medium' });

    next = STRUCTURAL; // a harder turn mid-session does not switch model: that would rewrite the whole cache
    const later = await router.decide(ctx([user('refatora esse modulo'), assistantText, user('agora redesenha a arquitetura')]));
    assert.deepEqual(later.primaryProfile, { model: SONNET, effort: 'medium' });
  });

  it('re-picks at compaction (the cache is rebuilt there anyway) and keeps that pick after it', async () => {
    let next = STANDARD;
    const router = setup(() => next);
    await router.decide(ctx([user('refatora esse modulo')]));
    next = STRUCTURAL;
    const compacted = await router.decide(ctx([user('summary'), assistantText, user('continua')], { contextCompacted: true }));
    assert.equal(compacted.route, 'primary');
    assert.equal(compacted.reason, 'pinned:compaction');
    assert.deepEqual(compacted.primaryProfile, { model: OPUS, effort: 'high' });
    next = SIMPLE;
    const after = await router.decide(ctx([user('summary'), assistantText, user('continua'), assistantToolUse, toolResult]));
    assert.deepEqual(after.primaryProfile, { model: OPUS, effort: 'high' });
  });

  it('a session it held no state for gets a profile at its next compaction', async () => {
    const router = setup(() => STANDARD);
    await router.decide(ctx([user('a'), assistantText, user('b')]));
    const compacted = await router.decide(ctx([user('summary'), assistantText, user('continua')], { contextCompacted: true }));
    assert.deepEqual(compacted.primaryProfile, { model: SONNET, effort: 'medium' });
  });

  it('keeps the default at compaction when JEV fails', async () => {
    let fail = false;
    const router = setup(() => {
      if (fail) throw new Error('down');
      return STANDARD;
    });
    await router.decide(ctx([user('refatora esse modulo')]));
    fail = true;
    const compacted = await router.decide(ctx([user('summary'), assistantText, user('continua')], { contextCompacted: true }));
    assert.equal(compacted.primaryProfile, undefined);
  });

  it('a session whose profile Anthropic refused stays off it after compaction', async () => {
    const router = setup(() => STANDARD);
    router.refusePrimaryProfile(await router.decide(ctx([user('refatora esse modulo')])));
    const compacted = await router.decide(ctx([user('summary'), assistantText, user('continua')], { contextCompacted: true }));
    assert.equal(compacted.primaryProfile, undefined);
  });

  it('the compaction request reads the history on the session model, without a new pick', async () => {
    let next = STANDARD;
    const router = setup(() => next);
    await router.decide(ctx([user('refatora esse modulo')]));
    next = STRUCTURAL;
    const summary = await router.decide(ctx([user('refatora esse modulo'), assistantText, user('resume')], { requestClass: 'compaction' }));
    assert.equal(summary.route, 'primary');
    assert.deepEqual(summary.primaryProfile, { model: SONNET, effort: 'medium' });
    const fresh = await setup(() => STANDARD).decide(ctx([user('resume')], { requestClass: 'compaction' }));
    assert.equal(fresh.primaryProfile, undefined);
  });

  it('carries the session profile on non-primary decisions too, so a failover to Claude keeps the cached model', async () => {
    const router = setup(() => STANDARD);
    await router.decide(ctx([user('refatora esse modulo')]));
    const loop = await router.decide(ctx([user('refatora esse modulo'), assistantToolUse, toolResult]));
    assert.deepEqual(loop.primaryProfile, { model: SONNET, effort: 'medium' });
  });

  it('picks a profile when a cheap conversation escalates to Claude (no Claude cache to lose)', async () => {
    let next = SIMPLE;
    const router = setup(() => next);
    assert.equal((await router.decide(ctx([user('lista os arquivos')]))).route, 'cheap');
    next = STRUCTURAL;
    const up = await router.decide(ctx([user('lista os arquivos'), assistantText, user('redesenha o roteador')]));
    assert.equal(up.route, 'primary');
    assert.deepEqual(up.primaryProfile, { model: OPUS, effort: 'high' });
  });

  it('never switches a session it holds no profile for (proxy restart mid-session)', async () => {
    const router = setup(() => STANDARD);
    const d = await router.decide(ctx([user('a'), assistantText, user('b')]));
    assert.equal(d.primaryProfile, undefined);
  });

  it('only touches the main loop, and only when enabled', async () => {
    const sub = await setup(() => STANDARD).decide(ctx([user('refatora')], { requestClass: 'subagent' }));
    assert.equal(sub.primaryProfile, undefined);
    const off = await setup(() => STANDARD, false).decide(ctx([user('refatora')]));
    assert.equal(off.primaryProfile, undefined);
  });
});

describe('server: primary profile on the wire', () => {
  const sent: any[] = [];
  let status = 200;
  let fails = 1;
  let anthropic: Server;
  let proxy: FastifyInstance;
  let url: string;

  before(async () => {
    anthropic = createServer(async (req, res) => {
      const parts: Buffer[] = [];
      for await (const c of req) parts.push(c as Buffer);
      sent.push(JSON.parse(Buffer.concat(parts).toString('utf8')));
      const code = sent.length <= fails ? status : 200;
      res.writeHead(code, { 'content-type': 'application/json' }).end(code === 200 ? '{"type":"message","content":[]}' : '{"type":"error","error":{"type":"invalid_request_error","message":"no"}}');
    });
    await new Promise<void>((r) => anthropic.listen(0, '127.0.0.1', r));
    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ANTHROPIC_UPSTREAM_URL: `http://127.0.0.1:${(anthropic.address() as AddressInfo).port}`,
      CHEAP_BASE_URL: 'http://127.0.0.1:9/v1',
      CHEAP_API_KEY: 'k',
      LOG_LEVEL: 'fatal',
    });
    const router = setup(() => STANDARD);
    proxy = buildServer({
      config,
      router,
      providers: { primary: new AnthropicProvider(config.primary), cheap: new OpenAICompatibleProvider(config.cheap) },
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    url = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  });
  after(async () => {
    await proxy.close();
    anthropic.close();
  });
  beforeEach(() => {
    sent.length = 0;
    status = 200;
    fails = 1;
  });

  const turn = (session: string, messages: unknown[] = [user('refatora esse modulo')], requestClass = 'main') =>
    request(`${url}/v1/messages`, {
      method: 'POST',
      headers: { 'content-type': 'application/json', 'x-api-key': 'sk-ant-client', 'x-claude-code-session-id': session, 'x-claude-code-request-class': requestClass },
      body: JSON.stringify({ model: OPUS, max_tokens: 1000, output_config: { effort: 'medium' }, messages }),
    });

  it('sends the profiled model and effort to Anthropic', async () => {
    const res = await turn('W1');
    await res.body.text();
    assert.equal(res.statusCode, 200);
    assert.equal(sent[0].model, SONNET);
    assert.deepEqual(sent[0].output_config, { effort: 'medium' });
  });

  it('resends the original request once when Anthropic refuses the rewritten one', async () => {
    status = 400;
    const res = await turn('W2');
    await res.body.text();
    assert.equal(res.statusCode, 200);
    assert.equal(sent.length, 2);
    assert.equal(sent[0].model, SONNET);
    assert.equal(sent[1].model, OPUS);

    // The refused profile is cleared: the next turn goes out as the client sent it, with no new attempt.
    const next = await turn('W2', [user('refatora esse modulo'), assistantText, user('continua')]);
    await next.body.text();
    assert.equal(next.statusCode, 200);
    assert.equal(sent.length, 3);
    assert.equal(sent[2].model, OPUS);
  });

  it('keeps the profile when the original is refused too (the request itself is bad, e.g. prompt too long)', async () => {
    status = 400;
    fails = 2;
    const res = await turn('W3');
    await res.body.text();
    assert.equal(res.statusCode, 400);
    assert.deepEqual(sent.map((b) => b.model), [SONNET, OPUS]);

    const next = await turn('W3', [user('refatora esse modulo'), assistantText, user('continua')]);
    await next.body.text();
    assert.equal(sent[2].model, SONNET);
  });

  it('sends the compaction request of a Sonnet session as Sonnet', async () => {
    await (await turn('W4')).body.text();
    const summary = await turn('W4', [user('refatora esse modulo'), assistantText, user('resume')], 'compaction');
    await summary.body.text();
    assert.equal(sent[1].model, SONNET);
  });
});
