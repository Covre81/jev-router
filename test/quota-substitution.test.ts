import assert from 'node:assert/strict';
import { Readable } from 'node:stream';
import { describe, it } from 'node:test';
import { loadConfig } from '../src/config.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { ModelQuotaState } from '../src/providers/quota-state.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import { buildServer } from '../src/proxy/server.js';
import type { Provider, ProviderRequest, ProviderResult } from '../src/providers/provider.js';
import { raiseToFloor, type PrimaryProfile } from '../src/domain/policy.js';

function okStream(text = 'ok'): Readable {
  return Readable.from([
    Buffer.from(
      JSON.stringify({
        type: 'message',
        role: 'assistant',
        content: [{ type: 'text', text }],
        stop_reason: 'end_turn',
        usage: { input_tokens: 10, output_tokens: 5 },
      }),
    ),
  ]);
}

function okResult(text = 'ok'): ProviderResult {
  return {
    kind: 'response',
    status: 200,
    headers: { 'content-type': 'application/json' },
    body: okStream(text),
  };
}

describe('Quota Substitution Ladder', () => {
  const baseConfig = loadConfig({
    CHEAP_API_KEY: 'k',
    CLASSIFIER: 'heuristic',
    GEMINI_CLI_TIER: 'on',
    ARGON_TIER: 'on',
    ARGON_MODEL_LOW: 'gemini-4-argon-low',
    ARGON_MODEL_MEDIUM: 'gemini-4-argon-medium',
    ARGON_MODEL_HIGH: 'gemini-4-argon-high',
    AGY_CLAUDE_SUBSTITUTE: 'on',
  });

  const createRouter = (route: 'free' | 'argon' | 'cursor' = 'free', profile: 'low' | 'medium' | 'high' = 'low') => {
    return new Router(
      {
        name: 'mock',
        classify: async () => {
          if (route === 'cursor') {
            return { simple: 0.1, standard: 0.4, structural: 0.1, textOnly: 0.2, risk: 0.1 };
          }
          if (route === 'argon') {
            if (profile === 'high') return { simple: 0.1, standard: 0.3, structural: 0.2, textOnly: 0.2, risk: 0.1, riskScores: { security_sensitive: 0, destructive_or_production: 0, requires_inspection: 0.5 } };
            if (profile === 'medium') return { simple: 0.2, standard: 0.55, structural: 0.15, textOnly: 0.2, risk: 0.1 };
            return { simple: 0.6, standard: 0.3, structural: 0.1, textOnly: 1.0, risk: 0.1 };
          }
          // route === 'free'
          return { simple: 0.95, standard: 0.05, structural: 0, textOnly: 1.0, risk: 0.1 };
        },
      } as any,
      new TtlLruStore(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, standardEnabled: false, argonEnabled: true },
        primaryClasses: new Set(['auxiliary']),
        geminiPolicy: { enabled: true, minTextOnly: 0.5, pressureMinTextOnly: 0.5, replacesCheap: true },
        classifierTimeoutMs: 1000,
        classifierMaxChars: 4000,
        cursorCoding: true,
        cursorEnabled: true,
        primaryProfiles: true,
      },
    );
  };

  it('1. Flash exhausted -> Argon low wins when available', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    const geminiCalls: { model?: string | undefined }[] = [];

    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        if (!req.modelOverride) {
          // Flash call returns quota
          return {
            kind: 'unavailable',
            reason: 'Individual quota reached. Resets in 100s',
            quota: { resetAt: now + 100_000 },
          };
        }
        return okResult(`argon response: ${req.modelOverride}`);
      },
    };

    const fakePrimary: Provider = {
      name: 'anthropic',
      send: async () => okResult('claude'),
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: fakePrimary, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'argon; reason=quota-substitute:gemini-3.8-flash-high->gemini-4-argon-low');
    assert.equal(res.headers['x-jev-substituted-from'], 'gemini-3.8-flash-high');
    assert.equal(geminiCalls.length, 2);
    assert.equal(geminiCalls[0]?.model, undefined); // Flash
    assert.equal(geminiCalls[1]?.model, 'gemini-4-argon-low');
  });

  it('2. Flash and Argon-low exhausted -> Argon medium wins', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-3.8-flash-high', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-low', now + 100_000, 'quota');

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`argon response: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'argon; reason=quota-substitute:gemini-3.8-flash-high->gemini-4-argon-medium');
    assert.equal(res.headers['x-jev-substituted-from'], 'gemini-3.8-flash-high');
    // Exhausted models (Flash and Argon low) were NEVER called!
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'gemini-4-argon-medium');
  });

  it('3. Flash, Argon low and Argon medium exhausted -> Argon high wins', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-3.8-flash-high', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-low', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-medium', now + 100_000, 'quota');

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`argon response: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'argon; reason=quota-substitute:gemini-3.8-flash-high->gemini-4-argon-high');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'gemini-4-argon-high');
  });

  it('4. Flash and all Argons exhausted -> agy claude-sonnet-5-5-low wins', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-3.8-flash-high', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-low', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-medium', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-high', now + 100_000, 'quota');

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`agy-claude response: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'free; reason=quota-substitute:gemini-3.8-flash-high->claude-sonnet-5-5-low');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'claude-sonnet-5-5-low');
  });

  it('5. Flash, all Argons and agy claude exhausted -> Anthropic (floor sonnet low)', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-3.8-flash-high', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-low', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-medium', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-high', now + 100_000, 'quota');
    quotaState.markExhausted('claude-sonnet-5-5-low', now + 100_000, 'quota');

    let anthropicReceivedBody: any;
    const fakePrimary: Provider = {
      name: 'anthropic',
      send: async (req: ProviderRequest) => {
        if (req.rawBody) {
          anthropicReceivedBody = JSON.parse(req.rawBody.toString('utf8'));
        }
        return okResult('anthropic response');
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: fakePrimary, gemini: { name: 'g', send: async () => okResult() } },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'primary; reason=quota-substitute:gemini-3.8-flash-high->claude');
    assert.equal(res.headers['x-jev-substituted-from'], 'gemini-3.8-flash-high');
    assert.ok(anthropicReceivedBody);
    // Profile is raised to at least floor sonnet low (PR #6 selector gives sonnet medium)
    assert.equal(anthropicReceivedBody.model, 'claude-sonnet-5-5');
    assert.ok(anthropicReceivedBody.output_config?.effort === 'low' || anthropicReceivedBody.output_config?.effort === 'medium');
  });

  it('6. Flash exhausted with ARGON_TIER=off -> skips Argon directly to agy claude-sonnet-5-5-low', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-3.8-flash-high', now + 100_000, 'quota');

    const configArgonOff = loadConfig({
      CHEAP_API_KEY: 'k',
      CLASSIFIER: 'heuristic',
      GEMINI_CLI_TIER: 'on',
      ARGON_TIER: 'off',
      AGY_CLAUDE_SUBSTITUTE: 'on',
    });

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`agy-claude: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: configArgonOff,
      router: createRouter('free'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.equal(res.headers['x-jev-route'], 'free; reason=quota-substitute:gemini-3.8-flash-high->claude-sonnet-5-5-low');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'claude-sonnet-5-5-low');
  });

  it('7. argon-low exhausted -> argon-medium -> argon-high -> agy claude-sonnet-5-5-medium -> Anthropic floor sonnet medium', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    // (a) argon-low exhausted -> argon-medium
    quotaState.markExhausted('gemini-4-argon-low', now + 100_000, 'quota');

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`argon: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('argon', 'low'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const resA = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resA.statusCode, 200);
    assert.equal(resA.headers['x-jev-route'], 'argon; reason=quota-substitute:gemini-4-argon-low->gemini-4-argon-medium');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'gemini-4-argon-medium');

    // (b) argon-low and medium exhausted -> agy claude-sonnet-5-5-medium when argon-high also exhausted
    quotaState.markExhausted('gemini-4-argon-medium', now + 100_000, 'quota');
    quotaState.markExhausted('gemini-4-argon-high', now + 100_000, 'quota');
    geminiCalls.length = 0;

    const resB = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resB.statusCode, 200);
    assert.equal(resB.headers['x-jev-route'], 'free; reason=quota-substitute:gemini-4-argon-low->claude-sonnet-5-5-medium');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'claude-sonnet-5-5-medium');

    // (c) agy claude also exhausted -> Anthropic floor sonnet medium
    quotaState.markExhausted('claude-sonnet-5-5-medium', now + 100_000, 'quota');
    let primaryBody: any;
    const fakePrimary: Provider = {
      name: 'anthropic',
      send: async (req: ProviderRequest) => {
        if (req.rawBody) primaryBody = JSON.parse(req.rawBody.toString('utf8'));
        return okResult();
      },
    };
    const appC = buildServer({
      config: baseConfig,
      router: createRouter('argon', 'low'),
      providers: { primary: fakePrimary, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const resC = await appC.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resC.statusCode, 200);
    assert.equal(resC.headers['x-jev-route'], 'primary; reason=quota-substitute:gemini-4-argon-low->claude');
    assert.equal(primaryBody.model, 'claude-sonnet-5-5');
    assert.equal(primaryBody.output_config?.effort, 'medium');
  });

  it('8. argon-medium exhausted -> argon-high -> agy claude-sonnet-5-5-high -> Anthropic floor sonnet high', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-4-argon-medium', now + 100_000, 'quota');

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`argon: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('argon', 'medium'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    // (a) argon-medium -> argon-high
    const resA = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resA.statusCode, 200);
    assert.equal(resA.headers['x-jev-route'], 'argon; reason=quota-substitute:gemini-4-argon-medium->gemini-4-argon-high');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'gemini-4-argon-high');

    // (b) argon-high exhausted -> agy claude-sonnet-5-5-high
    quotaState.markExhausted('gemini-4-argon-high', now + 100_000, 'quota');
    geminiCalls.length = 0;
    const resB = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resB.statusCode, 200);
    assert.equal(resB.headers['x-jev-route'], 'free; reason=quota-substitute:gemini-4-argon-medium->claude-sonnet-5-5-high');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'claude-sonnet-5-5-high');

    // (c) agy claude also exhausted -> Anthropic floor sonnet high
    quotaState.markExhausted('claude-sonnet-5-5-high', now + 100_000, 'quota');
    let primaryBody: any;
    const fakePrimary: Provider = {
      name: 'anthropic',
      send: async (req: ProviderRequest) => {
        if (req.rawBody) primaryBody = JSON.parse(req.rawBody.toString('utf8'));
        return okResult();
      },
    };
    const appC = buildServer({
      config: baseConfig,
      router: createRouter('argon', 'medium'),
      providers: { primary: fakePrimary, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const resC = await appC.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resC.statusCode, 200);
    assert.equal(resC.headers['x-jev-route'], 'primary; reason=quota-substitute:gemini-4-argon-medium->claude');
    assert.equal(primaryBody.model, 'claude-sonnet-5-5');
    assert.equal(primaryBody.output_config?.effort, 'high');
  });

  it('9. argon-high exhausted -> agy claude-opus-5-5-high -> Anthropic floor opus high', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-4-argon-high', now + 100_000, 'quota');

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`agy-claude: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('argon', 'high'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    // (a) argon-high exhausted -> agy claude-opus-5-5-high
    const resA = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resA.statusCode, 200);
    assert.equal(resA.headers['x-jev-route'], 'free; reason=quota-substitute:gemini-4-argon-high->claude-opus-5-5-high');
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'claude-opus-5-5-high');

    // (b) agy claude-opus-5-5-high exhausted -> Anthropic floor opus high
    quotaState.markExhausted('claude-opus-5-5-high', now + 100_000, 'quota');
    let primaryBody: any;
    const fakePrimary: Provider = {
      name: 'anthropic',
      send: async (req: ProviderRequest) => {
        if (req.rawBody) primaryBody = JSON.parse(req.rawBody.toString('utf8'));
        return okResult();
      },
    };
    const appB = buildServer({
      config: baseConfig,
      router: createRouter('argon', 'high'),
      providers: { primary: fakePrimary, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const resB = await appB.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        output_config: { effort: 'medium' },
        messages: [{ role: 'user', content: 'say hi' }],
      },
    });
    assert.equal(resB.statusCode, 200);
    assert.equal(resB.headers['x-jev-route'], 'primary; reason=quota-substitute:gemini-4-argon-high->claude');
    assert.equal(primaryBody.model, 'claude-opus-5-5');
    assert.equal(primaryBody.output_config?.effort, 'high');
  });

  it('10. Cursor quota-exhausted -> coding turn routes to Argon chain', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    let cursorCalled = false;
    const fakeCursor: Provider = {
      name: 'cursor',
      send: async () => {
        cursorCalled = true;
        return {
          kind: 'unavailable',
          reason: "You've hit your usage limit. Resets in 2h",
          quota: { resetAt: now + 7_200_000 },
        };
      },
    };

    const geminiCalls: { model?: string | undefined }[] = [];
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        geminiCalls.push({ model: req.modelOverride });
        return okResult(`argon: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('cursor'),
      providers: { primary: { name: 'p', send: async () => okResult() }, cursor: fakeCursor, gemini: fakeGemini },
      quotaState,
      cursorBreaker: new CircuitBreaker(3, 60_000, 1),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    const res = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'write a function' }],
      },
    });

    assert.equal(res.statusCode, 200);
    assert.ok(cursorCalled);
    assert.equal(quotaState.isAvailable('cursor'), false);
    assert.equal(res.headers['x-jev-route'], 'argon; reason=quota-substitute:cursor->gemini-4-argon-low');
    assert.equal(res.headers['x-jev-substituted-from'], 'cursor');

    // Subsequent call: cursor is NOT called at all
    cursorCalled = false;
    geminiCalls.length = 0;
    const res2 = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: {
        model: 'claude-opus-5-5',
        messages: [{ role: 'user', content: 'write another function' }],
      },
    });
    assert.equal(res2.statusCode, 200);
    assert.equal(cursorCalled, false);
    assert.equal(geminiCalls.length, 1);
    assert.equal(geminiCalls[0]?.model, 'gemini-4-argon-low');
  });

  it('11. Auto-restore after resetAt expires', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-3.8-flash-high', now + 5000, 'quota: 429');

    let flashCalled = false;
    const fakeGemini: Provider = {
      name: 'gemini-cli',
      send: async (req: ProviderRequest) => {
        if (!req.modelOverride) {
          flashCalled = true;
          return okResult('flash ok');
        }
        return okResult(`argon: ${req.modelOverride}`);
      },
    };

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: { name: 'p', send: async () => okResult() }, gemini: fakeGemini },
      quotaState,
      geminiBreaker: new CircuitBreaker(5, 60_000, 2),
      argonBreaker: new CircuitBreaker(5, 60_000, 2),
    });

    // Call 1: Flash is exhausted, routes to Argon low
    const res1 = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'hi' }] },
    });
    assert.equal(res1.headers['x-jev-route'], 'argon; reason=quota-substitute:gemini-3.8-flash-high->gemini-4-argon-low');
    assert.equal(flashCalled, false);

    // Advance clock past resetAt
    now += 6000;

    // Call 2: Flash is restored!
    const res2 = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'completely distinct new query' }] },
    });
    assert.ok(
      res2.headers['x-jev-route']?.toString().startsWith('free;'),
      `Expected free route on restore, saw: ${res2.headers['x-jev-route']}`,
    );
    assert.equal(flashCalled, true);
  });

  it('12. raiseToFloor helper never lowers higher profile', () => {
    const opusHigh: PrimaryProfile = { model: 'claude-opus-5-5', effort: 'high' };
    const sonnetMedium: PrimaryProfile = { model: 'claude-sonnet-5-5', effort: 'medium' };
    const sonnetLow: PrimaryProfile = { model: 'claude-sonnet-5-5', effort: 'low' };

    // Higher profile raised to lower floor remains unchanged
    assert.deepEqual(raiseToFloor(opusHigh, sonnetMedium), opusHigh);
    assert.deepEqual(raiseToFloor(opusHigh, sonnetLow), opusHigh);

    // Lower profile raised to higher floor becomes the floor
    assert.deepEqual(raiseToFloor(sonnetLow, sonnetMedium), sonnetMedium);
    assert.deepEqual(raiseToFloor(sonnetLow, opusHigh), opusHigh);
  });

  it('13. /healthz returns model snapshot with state, until and reason', async () => {
    let now = 1000;
    const quotaState = new ModelQuotaState(() => now);
    quotaState.markExhausted('gemini-4-argon-low', now + 10_000, 'Individual quota reached');

    const app = buildServer({
      config: baseConfig,
      router: createRouter('free'),
      providers: { primary: { name: 'p', send: async () => okResult() } },
      quotaState,
    });

    const res = await app.inject({
      method: 'GET',
      url: '/healthz',
    });

    assert.equal(res.statusCode, 200);
    const body = JSON.parse(res.body);
    assert.ok(body.models, 'Expected models in /healthz');
    assert.equal(body.models['gemini-4-argon-low']?.state, 'exhausted');
    assert.ok(body.models['gemini-4-argon-low']?.until);
    assert.equal(body.models['gemini-4-argon-low']?.reason, 'Individual quota reached');
    assert.equal(body.models['gemini-3.8-flash-high']?.state, 'ok');
  });
});
