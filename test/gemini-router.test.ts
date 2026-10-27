import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

describe('Router decisions', () => {
  function setupRouter(dist: any = {}, options: any = {}) {
    const classifier = { name: 'test', classify: async () => dist };
    const store = new TtlLruStore<any>(100, 60_000);
    const quota = options.quota || { level: () => 'none' };
    const router = new Router(classifier as any, store, {
      policy: {
        minCheapProbability: 0.8,
        standardRoute: 'primary',
        standardEnabled: false,
        argonEnabled: options.argonEnabled ?? false,
      },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      ...(options.noGeminiPolicy ? {} : { geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 } }),
      geminiFromPrimary: options.geminiFromPrimary ?? true,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota,
    });
    return { router, store, quota, classifier };
  }

  function req(body: any, headers: any = {}, rawByteLength = 100) {
    return { body, rawByteLength, sessionId: headers['x-claude-code-session-id'], requestClass: headers['x-claude-code-request-class'], contextCompacted: false, agentId: undefined };
  }

  it('fresh human turn gemini eligible', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's1' }) as any);
    assert.equal(dec.route, 'free');
    assert.equal(dec.reason, 'gemini:text-only');
    assert.equal(dec.fallbackTier, 'primary');
    assert.equal(store.get('s1:main'), 'gemini');
  });

  it('high risk -> primary', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.7, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's2' }) as any);
    assert.equal(dec.route, 'primary');
  });

  it('highly simple -> free', async () => {
    const { router } = setupRouter({ simple: 0.95, standard: 0.02, structural: 0.03, risk: 0.1, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's3' }) as any);
    assert.equal(dec.tier, 'free');
    assert.equal(dec.route, 'free');
  });

  it('quota level pressure', async () => {
    const { router } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.65 }, { quota: { level: () => 'pressure' } });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's4' }) as any);
    assert.equal(dec.route, 'free');
    assert.equal(dec.reason, 'gemini:text-only:pressure');

    const { router: routerNone } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.65 }, { quota: { level: () => 'none' } });
    const decNone = await routerNone.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's5' }) as any);
    assert.equal(decNone.route, 'primary');
  });

  it('tool-result continuation on gemini', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    store.set('s6:main', 'free');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: [{ type: 'tool_result' }] }] }, { 'x-claude-code-session-id': 's6' }) as any);
    assert.equal(dec.route, 'free');
    assert.equal(dec.reason, 'sticky');
  });

  it('next human turn of a gemini conversation can return free', async () => {
    const { router, store } = setupRouter({ simple: 0.95, standard: 0.02, structural: 0.03, risk: 0.1, textOnly: 0.95 });
    store.set('s6:main', 'free');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's7' }) as any);
    assert.equal(dec.tier, 'free');
  });

  it('stored primary + geminiFromPrimary true + text-only turn', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 }, { geminiFromPrimary: true });
    store.set('s8:main', 'primary');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's8' }) as any);
    assert.equal(dec.route, 'free');
    assert.equal(store.get('s8:main'), 'primary');
  });

  it('stored primary + geminiFromPrimary false', async () => {
    let callCount = 0;
    const classifier = { name: 'test', classify: async () => { callCount++; return { simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 }; } };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, { policy: { minCheapProbability: 0.8, standardRoute: 'primary', standardEnabled: false, argonEnabled: false }, primaryClasses: new Set([]), cheapContextTokens: 100_000, geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 }, geminiFromPrimary: false, classifierMaxChars: 4000, classifierTimeoutMs: 1000, quota: { level: () => 'none' } });
    store.set('s9:main', 'primary');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's9' }) as any);
    assert.equal(dec.route, 'primary');
    assert.equal(dec.reason, 'sticky');
    assert.equal(callCount, 0);
  });

  it('a Gemini turn that fell back to Claude makes no fake transition on the next turn', async () => {
    const dist: any = { simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 };
    const { router, store } = setupRouter(dist);
    store.set('s15:main', 'primary');
    const turn = { messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'hi' }] };
    const toGemini = await router.decide(req(turn, { 'x-claude-code-session-id': 's15' }) as any);
    assert.equal(toGemini.route, 'free');
    router.clearGeminiLoop(toGemini, 'primary'); // Gemini failed: Claude served it
    dist.textOnly = 0.1;
    const next = await router.decide(req(turn, { 'x-claude-code-session-id': 's15' }) as any);
    assert.equal(next.route, 'primary');
    assert.equal(next.modelTransition, undefined);
  });

  it('the tool loop of a fresh turn Claude rescued from Gemini stays on Claude', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    const sid = { 'x-claude-code-session-id': 's16' };
    const first = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, sid) as any);
    assert.equal(first.route, 'free');
    router.clearGeminiLoop(first, 'primary');
    const toolStep = await router.decide(req({ messages: [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: [{ type: 'tool_use', id: '1', name: 'Read', input: {} }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: '1', content: 'ok' }] },
    ] }, sid) as any);
    assert.equal(toolStep.route, 'primary');
    assert.equal(store.get('s16:main'), 'primary');
  });

  it('stored primary + geminiFromPrimary true + classifier rejects', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.1 }, { geminiFromPrimary: true });
    store.set('s10:main', 'primary');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's10' }) as any);
    assert.equal(dec.route, 'primary');
    assert.equal(dec.reason, 'sticky');
  });

  it('requestClass auxiliary -> primary', async () => {
    const { router } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's11', 'x-claude-code-request-class': 'auxiliary' }) as any);
    assert.equal(dec.route, 'primary');
    assert.equal(dec.reason, 'passthrough:request-class');
  });

  it('tool_choice any or image -> not gemini', async () => {
    const { router } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    const dec1 = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }], tool_choice: { type: 'any' } }, { 'x-claude-code-session-id': 's12' }) as any);
    assert.notEqual(dec1.route, 'free');

    const dec2 = await router.decide(req({ messages: [{ role: 'user', content: [{ type: 'image' }] }] }, { 'x-claude-code-session-id': 's13' }) as any);
    assert.notEqual(dec2.route, 'free');
  });

  it('no geminiPolicy + text-only -> primary', async () => {
    const { router } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 }, { noGeminiPolicy: true });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's14' }) as any);
    assert.equal(dec.route, 'primary');
    assert.equal(dec.reason, 'classified');
  });

  it('walks the economy ladder: free -> argon -> primary', async () => {
    const scripts = [
      // Turn 1: simple -> free (Flash)
      { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.05 }, textOnly: 0.95 },
      // Turn 2: level-2 -> free (standard bar met: 0.3 + 0.55 >= 0.75, no inspection veto)
      { simple: 0.3, standard: 0.55, structural: 0.15, risk: 0.1, riskScores: { security_sensitive: 0.02, destructive_or_production: 0.02, requires_inspection: 0.1 }, textOnly: 0.05 },
      // Turn 3: structural requiring inspection -> argon
      { simple: 0.05, standard: 0.1, structural: 0.85, risk: 0.95, riskScores: { security_sensitive: 0.05, destructive_or_production: 0.05, requires_inspection: 0.95 }, textOnly: 0.02 },
      // Turn 4: security/destructive high risk -> primary (Claude)
      { simple: 0.05, standard: 0.1, structural: 0.85, risk: 0.95, riskScores: { security_sensitive: 0.95, destructive_or_production: 0.1, requires_inspection: 0.9 }, textOnly: 0.02 },
    ];
    const classifier = { name: 'test', classify: async () => scripts.shift()! };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.9, standardEnabled: true, minStandardProbability: 0.75, argonEnabled: true },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      geminiPolicy: { enabled: true, minTextOnly: 0, pressureMinTextOnly: 0 },
      geminiFromPrimary: true,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota: { level: () => 'none' },
    });

    const sid = { 'x-claude-code-session-id': 's-ladder' };
    const m1 = [{ role: 'user', content: 't1' }];
    const d1 = await router.decide(req({ messages: m1 }, sid) as any);
    assert.deepEqual([d1.route, d1.tier, d1.reason], ['free', 'free', 'free:flash']);

    const m2 = [...m1, { role: 'assistant', content: 'a1' }, { role: 'user', content: 't2' }];
    const d2 = await router.decide(req({ messages: m2 }, sid) as any);
    assert.deepEqual([d2.route, d2.tier, d2.reason], ['free', 'free', 'free:flash']);

    const m3 = [...m2, { role: 'assistant', content: 'a2' }, { role: 'user', content: 't3' }];
    const d3 = await router.decide(req({ messages: m3 }, sid) as any);
    assert.deepEqual([d3.route, d3.tier, d3.reason], ['argon', 'argon', 'argon:high']);

    const m4 = [...m3, { role: 'assistant', content: 'a3' }, { role: 'user', content: 't4' }];
    const d4 = await router.decide(req({ messages: m4 }, sid) as any);
    assert.deepEqual([d4.route, d4.tier, d4.reason], ['primary', 'primary', 'escalated']);
  });
});
