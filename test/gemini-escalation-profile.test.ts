import assert from 'node:assert/strict';
import { createServer, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, beforeEach, describe, it } from 'node:test';
import type { FastifyInstance } from 'fastify';
import { request } from 'undici';
import { loadConfig } from '../src/config.js';
import { AnthropicProvider } from '../src/providers/anthropic.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import { buildServer } from '../src/proxy/server.js';
import type { ClassificationInput, ComplexityClassifier } from '../src/classifier/classifier.js';
import type { ComplexityDistribution } from '../src/domain/complexity.js';
import {
  BUMP_LADDER,
  primaryProfile,
  type EscalationReason,
  type PrimaryProfile,
  type Tier,
} from '../src/domain/policy.js';
import { MessagesBody } from '../src/routing/messages-body.js';
import { Router, type RequestContext, type RouteDecision } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

const OPUS = 'claude-opus-5-5';
const SONNET = 'claude-sonnet-5-5';

describe('Gemini escalation profile policy', () => {
  it('reaches every model/effort cell', () => {
    // sonnet low: simple >= 0.6
    assert.deepEqual(primaryProfile({ simple: 0.7, standard: 0.2, structural: 0.1 }), {
      model: SONNET,
      effort: 'low',
    });

    // sonnet medium: base fallback (simple < 0.6, structural < 0.3)
    assert.deepEqual(primaryProfile({ simple: 0.4, standard: 0.5, structural: 0.1 }), {
      model: SONNET,
      effort: 'medium',
    });

    // sonnet high: structural >= 0.3
    assert.deepEqual(primaryProfile({ simple: 0.3, standard: 0.3, structural: 0.4 }), {
      model: SONNET,
      effort: 'high',
    });

    // sonnet xhigh: bump from sonnet high
    assert.deepEqual(
      primaryProfile({ simple: 0.3, standard: 0.3, structural: 0.4 }, 'gemini-failed'),
      { model: SONNET, effort: 'xhigh' },
    );
    assert.deepEqual(
      primaryProfile({ simple: 0.3, standard: 0.3, structural: 0.4 }, 'escalated-from-gemini'),
      { model: SONNET, effort: 'xhigh' },
    );

    // opus low: context escalation with simple >= 0.6
    assert.deepEqual(
      primaryProfile({ simple: 0.7, standard: 0.2, structural: 0.1 }, 'context'),
      { model: OPUS, effort: 'low' },
    );

    // opus medium: structural >= 0.45 or requires_inspection >= 0.5
    assert.deepEqual(primaryProfile({ simple: 0.2, standard: 0.3, structural: 0.5 }), {
      model: OPUS,
      effort: 'medium',
    });
    assert.deepEqual(
      primaryProfile({
        simple: 0.8,
        standard: 0.15,
        structural: 0.05,
        riskScores: { security_sensitive: 0.05, destructive_or_production: 0.05, requires_inspection: 0.6 },
      }),
      { model: OPUS, effort: 'medium' },
    );

    // opus high: hardRiskVeto(d) || structural >= 0.6 (without both)
    assert.deepEqual(primaryProfile({ simple: 0.1, standard: 0.2, structural: 0.7 }), {
      model: OPUS,
      effort: 'high',
    });
    assert.deepEqual(
      primaryProfile({ simple: 0.8, standard: 0.15, structural: 0.05, risk: 0.9 }),
      { model: OPUS, effort: 'high' },
    );

    // opus xhigh: hardRiskVeto(d) && structural >= 0.6
    assert.deepEqual(
      primaryProfile({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.9 }),
      { model: OPUS, effort: 'xhigh' },
    );
  });

  it('bumps along the ladder and caps at opus xhigh, never picking max', () => {
    assert.deepEqual(BUMP_LADDER, [
      { model: SONNET, effort: 'low' },
      { model: SONNET, effort: 'medium' },
      { model: SONNET, effort: 'high' },
      { model: SONNET, effort: 'xhigh' },
      { model: OPUS, effort: 'medium' },
      { model: OPUS, effort: 'high' },
      { model: OPUS, effort: 'xhigh' },
    ]);

    // Step 0 -> 1: sonnet low -> sonnet medium
    assert.deepEqual(
      primaryProfile({ simple: 0.7, standard: 0.2, structural: 0.1 }, 'gemini-failed'),
      { model: SONNET, effort: 'medium' },
    );
    // Step 1 -> 2: sonnet medium -> sonnet high
    assert.deepEqual(
      primaryProfile({ simple: 0.4, standard: 0.5, structural: 0.1 }, 'gemini-failed'),
      { model: SONNET, effort: 'high' },
    );
    // Step 2 -> 3: sonnet high -> sonnet xhigh
    assert.deepEqual(
      primaryProfile({ simple: 0.3, standard: 0.3, structural: 0.4 }, 'gemini-failed'),
      { model: SONNET, effort: 'xhigh' },
    );
    // Step 4 -> 5: opus medium -> opus high
    assert.deepEqual(
      primaryProfile({ simple: 0.2, standard: 0.3, structural: 0.5 }, 'gemini-failed'),
      { model: OPUS, effort: 'high' },
    );
    // Step 5 -> 6: opus high -> opus xhigh
    assert.deepEqual(
      primaryProfile({ simple: 0.1, standard: 0.2, structural: 0.7 }, 'gemini-failed'),
      { model: OPUS, effort: 'xhigh' },
    );
    // Step 6 -> 6: opus xhigh capped at opus xhigh
    assert.deepEqual(
      primaryProfile({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.9 }, 'gemini-failed'),
      { model: OPUS, effort: 'xhigh' },
    );

    // Verify 'max' is never returned for any base or escalation combinations
    const escalations: (EscalationReason | undefined)[] = [
      undefined,
      'gemini-failed',
      'escalated-from-gemini',
      'gemini-busy',
      'gemini-unhealthy',
      'context',
    ];
    const testCases: ComplexityDistribution[] = [
      { simple: 0.9, standard: 0.05, structural: 0.05 },
      { simple: 0.5, standard: 0.4, structural: 0.1 },
      { simple: 0.3, standard: 0.35, structural: 0.35 },
      { simple: 0.2, standard: 0.3, structural: 0.5 },
      { simple: 0.1, standard: 0.1, structural: 0.8 },
    ];
    for (const tc of testCases) {
      for (const esc of escalations) {
        const p = primaryProfile(tc, esc);
        assert.notEqual(p.effort, 'max');
      }
    }
  });

  it('leaves gemini-busy and gemini-unhealthy unbumped', () => {
    const d = { simple: 0.4, standard: 0.5, structural: 0.1 };
    assert.deepEqual(primaryProfile(d, 'gemini-busy'), primaryProfile(d));
    assert.deepEqual(primaryProfile(d, 'gemini-unhealthy'), primaryProfile(d));
  });
});

class ScriptedClassifier implements ComplexityClassifier {
  readonly name = 'scripted';
  constructor(private readonly next: () => ComplexityDistribution) {}
  classify(_input: ClassificationInput): Promise<ComplexityDistribution> {
    return Promise.resolve(this.next());
  }
}

const user = (text: string) => ({ role: 'user', content: [{ type: 'text', text }] });
const assistantText = { role: 'assistant', content: [{ type: 'text', text: 'ok' }] };

function ctx(messages: unknown[], overrides: Partial<RequestContext> = {}): RequestContext {
  return {
    body: MessagesBody.parse({ model: OPUS, messages }),
    rawByteLength: 30_000,
    sessionId: 'S_GEMINI',
    agentId: undefined,
    requestClass: 'main',
    contextCompacted: false,
    ...overrides,
  };
}

describe('Router Gemini escalation profile handling', () => {
  it('treats previousTier gemini as opening and applies escalated-from-gemini bump', async () => {
    let score = { simple: 0.4, standard: 0.5, structural: 0.1, textOnly: 0.9 };
    const router = new Router(
      new ScriptedClassifier(() => score),
      new TtlLruStore<Tier>(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, minStandardProbability: 0.95, argonEnabled: false },
        primaryClasses: new Set(['auxiliary', 'compaction']),
        cheapContextTokens: 100_000,
        classifierTimeoutMs: 1_000,
        classifierMaxChars: 4_000,
        primaryProfiles: true,
        geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 },
      },
    );

    // Turn 1 on Gemini
    const t1 = await router.decide(ctx([user('oi')]));
    assert.equal(t1.route, 'free');

    // Turn 2: non-text-only (textOnly: 0.1), so escalates to Claude
    score = { simple: 0.4, standard: 0.5, structural: 0.1, textOnly: 0.1 };
    const t2 = await router.decide(ctx([user('oi'), assistantText, user('refatora o modulo')]));
    assert.equal(t2.route, 'primary');
    assert.equal(t2.previousTier, 'gemini');
    // Bumped from sonnet medium to sonnet high
    assert.deepEqual(t2.primaryProfile, { model: SONNET, effort: 'high' });
    assert.equal(t2.escalationReason, 'escalated-from-gemini');
  });

  it('profileForFallback: gemini-busy does not store profile', async () => {
    const router = new Router(
      new ScriptedClassifier(() => ({ simple: 0.4, standard: 0.5, structural: 0.1 })),
      new TtlLruStore<Tier>(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, minStandardProbability: 0.95, argonEnabled: false },
        primaryClasses: new Set(),
        cheapContextTokens: 100_000,
        classifierTimeoutMs: 1_000,
        classifierMaxChars: 4_000,
        primaryProfiles: true,
      },
    );

    const c = ctx([user('turn 1')]);
    const dec: RouteDecision = {
      route: 'gemini',
      tier: 'gemini',
      reason: 'gemini:text-only',
      conversationKey: 'S_BUSY:main',
      distribution: { simple: 0.4, standard: 0.5, structural: 0.1 },
    };

    const profiled = await router.profileForFallback(dec, c, 'gemini-busy');
    // Base table for simple: 0.4, standard: 0.5, structural: 0.1 is sonnet medium (no bump)
    assert.deepEqual(profiled.primaryProfile, { model: SONNET, effort: 'medium' });
    assert.equal(profiled.escalationReason, 'gemini-busy');

    // Nothing was stored in router.profiles!
    const key = 'S_BUSY:main';
    assert.equal((router as any).profiles.has(key), false);
  });

  it('profileForFallback: gemini-failed stores bumped profile and reuses it', async () => {
    const router = new Router(
      new ScriptedClassifier(() => ({ simple: 0.4, standard: 0.5, structural: 0.1 })),
      new TtlLruStore<Tier>(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, minStandardProbability: 0.95, argonEnabled: false },
        primaryClasses: new Set(),
        cheapContextTokens: 100_000,
        classifierTimeoutMs: 1_000,
        classifierMaxChars: 4_000,
        primaryProfiles: true,
      },
    );

    const c = ctx([user('turn 1')]);
    const dec: RouteDecision = {
      route: 'gemini',
      tier: 'gemini',
      reason: 'failover:gemini-unavailable',
      conversationKey: 'S_FAILED:main',
      distribution: { simple: 0.4, standard: 0.5, structural: 0.1 },
    };

    const profiled = await router.profileForFallback(dec, c, 'gemini-failed');
    // Bumped from sonnet medium to sonnet high
    assert.deepEqual(profiled.primaryProfile, { model: SONNET, effort: 'high' });
    assert.equal(profiled.escalationReason, 'gemini-failed');

    // Stored in router.profiles!
    const key = 'S_FAILED:main';
    assert.deepEqual((router as any).profiles.get(key), { model: SONNET, effort: 'high' });

    // Subsequent call reuses the stored profile
    const nextDec: RouteDecision = {
      route: 'primary',
      reason: 'sticky',
      conversationKey: key,
    };
    const reused = await router.profileForFallback(nextDec, c, 'gemini-failed');
    assert.deepEqual(reused.primaryProfile, { model: SONNET, effort: 'high' });
  });

  it('profileForFallback respects options.primaryProfiles=false', async () => {
    const router = new Router(
      new ScriptedClassifier(() => ({ simple: 0.4, standard: 0.5, structural: 0.1 })),
      new TtlLruStore<Tier>(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, minStandardProbability: 0.95, argonEnabled: false },
        primaryClasses: new Set(),
        cheapContextTokens: 100_000,
        classifierTimeoutMs: 1_000,
        classifierMaxChars: 4_000,
        primaryProfiles: false,
      },
    );

    const dec: RouteDecision = {
      route: 'gemini',
      reason: 'failover:gemini-unavailable',
      conversationKey: 'S_OFF:main',
      distribution: { simple: 0.4, standard: 0.5, structural: 0.1 },
    };
    const res = await router.profileForFallback(dec, undefined, 'gemini-failed');
    assert.equal(res.primaryProfile, undefined);
  });
});

describe('Server-level Gemini escalation on the wire', () => {
  let anthropic: Server;
  let anthropicUrl: string;
  const sentAnthropic: any[] = [];
  let proxy: FastifyInstance;
  let proxyUrl: string;
  let geminiBreaker: CircuitBreaker;

  let geminiOutcome: 'success' | 'busy' | 'real-fail' = 'success';

  before(async () => {
    anthropic = createServer(async (req, res) => {
      const parts: Buffer[] = [];
      for await (const chunk of req) parts.push(chunk as Buffer);
      const text = Buffer.concat(parts).toString('utf8');
      sentAnthropic.push(text ? JSON.parse(text) : null);
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(
        JSON.stringify({
          type: 'message',
          role: 'assistant',
          content: [{ type: 'text', text: 'anthropic ok' }],
          stop_reason: 'end_turn',
          usage: {},
        }),
      );
    });
    await new Promise<void>((r) => anthropic.listen(0, '127.0.0.1', r));
    anthropicUrl = `http://127.0.0.1:${(anthropic.address() as AddressInfo).port}`;

    const config = loadConfig({
      CLASSIFIER: 'heuristic',
      ANTHROPIC_UPSTREAM_URL: anthropicUrl,
      CHEAP_API_KEY: 'test-key',
      LOG_LEVEL: 'fatal',
      GEMINI_TIER: 'on',
      ARGON_TIER: 'off',
      PRIMARY_PROFILES: 'true',
    });

    const router = new Router(
      new ScriptedClassifier(() => ({ simple: 0.4, standard: 0.5, structural: 0.1, textOnly: 0.9 })),
      new TtlLruStore<Tier>(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, minStandardProbability: 0.95, argonEnabled: false },
        primaryClasses: new Set(['auxiliary', 'compaction']),
        cheapContextTokens: 100_000,
        classifierTimeoutMs: 1_000,
        classifierMaxChars: 4_000,
        primaryProfiles: true,
        geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 },
      },
    );

    const fakeGeminiProvider = {
      send: async () => {
        if (geminiOutcome === 'real-fail') {
          return { kind: 'unavailable' as const, reason: 'process exited with code 1' };
        }
        return {
          kind: 'response' as const,
          status: 200,
          headers: { 'content-type': 'application/json' },
          body: (async function* () {
            yield Buffer.from(
              JSON.stringify({
                type: 'message',
                role: 'assistant',
                content: [{ type: 'text', text: 'gemini ok' }],
                stop_reason: 'end_turn',
                usage: {},
              }),
            );
          })(),
        };
      },
    };

    geminiBreaker = new CircuitBreaker(1, 100_000, 1);

    proxy = buildServer({
      config,
      router,
      providers: {
        primary: new AnthropicProvider(config.primary),
        gemini: fakeGeminiProvider as any,
      },
      geminiBreaker,
    });
    await proxy.listen({ host: '127.0.0.1', port: 0 });
    proxyUrl = `http://127.0.0.1:${(proxy.server.address() as AddressInfo).port}`;
  });

  after(async () => {
    await proxy.close();
    anthropic.closeAllConnections?.();
    anthropic.close();
  });

  beforeEach(() => {
    sentAnthropic.length = 0;
    geminiOutcome = 'success';
  });

  it('gemini-busy turn sends base model/effort in outgoing body', async () => {
    geminiOutcome = 'busy';
    // Acquire the single slot to force next request to be busy
    assert.ok(geminiBreaker.acquire());

    const res = await request(`${proxyUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-claude-code-session-id': 'sess-busy-turn',
        'x-claude-code-request-class': 'main',
      },
      body: JSON.stringify({
        model: OPUS,
        max_tokens: 1000,
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'ajuda com codigo' }],
      }),
    });
    await res.body.text();
    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'primary; reason=skipped:gemini-busy');

    // Base profile for standard (simple 0.4, standard 0.5, structural 0.1) is sonnet medium
    assert.equal(sentAnthropic.length, 1);
    assert.equal(sentAnthropic[0].model, SONNET);
    assert.deepEqual(sentAnthropic[0].output_config, { effort: 'medium' });

    geminiBreaker.release('neutral');
  });

  it('gemini-failed turn sends bumped model/effort in outgoing body', async () => {
    geminiOutcome = 'real-fail';

    const res = await request(`${proxyUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-claude-code-session-id': 'sess-failed-turn',
        'x-claude-code-request-class': 'main',
      },
      body: JSON.stringify({
        model: OPUS,
        max_tokens: 1000,
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'ajuda com codigo' }],
      }),
    });
    await res.body.text();
    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'primary; reason=failover:gemini-unavailable');

    // Bumped profile: base sonnet medium bumped to sonnet high
    assert.equal(sentAnthropic.length, 1);
    assert.equal(sentAnthropic[0].model, SONNET);
    assert.deepEqual(sentAnthropic[0].output_config, { effort: 'high' });
  });

  it('explicit /model (non-default model) is not rewritten', async () => {
    geminiOutcome = 'real-fail';

    const customModel = 'claude-sonnet-4';
    const res = await request(`${proxyUrl}/v1/messages`, {
      method: 'POST',
      headers: {
        'content-type': 'application/json',
        'x-claude-code-session-id': 'sess-explicit-model',
        'x-claude-code-request-class': 'main',
      },
      body: JSON.stringify({
        model: customModel,
        max_tokens: 1000,
        messages: [{ role: 'user', content: 'ajuda com codigo' }],
      }),
    });
    await res.body.text();
    assert.equal(res.statusCode, 200);

    assert.equal(sentAnthropic.length, 1);
    assert.equal(sentAnthropic[0].model, customModel);
  });
});

describe('Config GEMINI_TIER_MAX_PROMPT_CHARS default', () => {
  it('defaults to 3_200_000', () => {
    const config = loadConfig({
      CHEAP_API_KEY: 'test-key',
      CLASSIFIER: 'heuristic',
      GEMINI_TIER: 'on',
    });
    assert.equal(config.gemini?.maxPromptChars, 3_200_000);
  });
});
