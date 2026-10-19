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
      policy: { minCheapProbability: 0.8, standardRoute: 'primary', standardEnabled: false },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      ...(options.noGeminiPolicy ? {} : { geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 } }),
      geminiFromPrimary: options.geminiFromPrimary ?? true,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota
    });
    return { router, store, quota, classifier };
  }

  function req(body: any, headers: any = {}, rawByteLength = 100) {
    return { body, rawByteLength, sessionId: headers['x-claude-code-session-id'], requestClass: headers['x-claude-code-request-class'], contextCompacted: false, agentId: undefined };
  }

  it('fresh human turn gemini eligible', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's1' }) as any);
    assert.equal(dec.route, 'gemini');
    assert.equal(dec.reason, 'gemini:text-only');
    assert.equal(dec.fallbackTier, 'primary');
    assert.equal(store.get('s1:main'), 'gemini');
  });

  it('high risk -> primary', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.7, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's2' }) as any);
    assert.equal(dec.route, 'primary');
  });

  it('highly simple -> trivial', async () => {
    const { router, store } = setupRouter({ simple: 0.95, standard: 0.02, structural: 0.03, risk: 0.1, textOnly: 0.95 });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's3' }) as any);
    assert.equal(dec.tier, 'trivial');
  });

  it('quota level pressure', async () => {
    const { router } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.65 }, { quota: { level: () => 'pressure' } });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's4' }) as any);
    assert.equal(dec.route, 'gemini');
    assert.equal(dec.reason, 'gemini:text-only:pressure');
    
    const { router: routerNone } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.65 }, { quota: { level: () => 'none' } });
    const decNone = await routerNone.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's5' }) as any);
    assert.equal(decNone.route, 'primary');
  });

  it('tool-result continuation on gemini', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 });
    store.set('s6:main', 'gemini');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: [{ type: 'tool_result' }] }] }, { 'x-claude-code-session-id': 's6' }) as any);
    assert.equal(dec.route, 'gemini');
    assert.equal(dec.reason, 'sticky');
  });

  it('next human turn of a gemini conversation can return trivial', async () => {
    const { router, store } = setupRouter({ simple: 0.95, standard: 0.02, structural: 0.03, risk: 0.1, textOnly: 0.95 });
    store.set('s6:main', 'gemini');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's7' }) as any);
    assert.equal(dec.tier, 'trivial');
  });

  it('stored primary + geminiFromPrimary true + text-only turn', async () => {
    const { router, store } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 }, { geminiFromPrimary: true });
    store.set('s8:main', 'primary');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's8' }) as any);
    assert.equal(dec.route, 'gemini');
    assert.equal(store.get('s8:main'), 'primary');
  });

  it('stored primary + geminiFromPrimary false', async () => {
    let callCount = 0;
    const classifier = { name: 'test', classify: async () => { callCount++; return { simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 }; } };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, { policy: { minCheapProbability: 0.8, standardRoute: 'primary', standardEnabled: false }, primaryClasses: new Set([]), cheapContextTokens: 100_000, geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 }, geminiFromPrimary: false, classifierMaxChars: 4000, classifierTimeoutMs: 1000, quota: { level: () => 'none' } });
    store.set('s9:main', 'primary');
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's9' }) as any);
    assert.equal(dec.route, 'primary');
    assert.equal(dec.reason, 'sticky');
    assert.equal(callCount, 0);
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
    assert.notEqual(dec1.route, 'gemini');
    
    const dec2 = await router.decide(req({ messages: [{ role: 'user', content: [{ type: 'image' }] }] }, { 'x-claude-code-session-id': 's13' }) as any);
    assert.notEqual(dec2.route, 'gemini');
  });

  it('no geminiPolicy + text-only -> primary', async () => {
    const { router } = setupRouter({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.1, textOnly: 0.95 }, { noGeminiPolicy: true });
    const dec = await router.decide(req({ messages: [{ role: 'user', content: 'hi' }] }, { 'x-claude-code-session-id': 's14' }) as any);
    assert.equal(dec.route, 'primary');
    assert.equal(dec.reason, 'classified');
  });

  it('walks the full economy ladder: trivial -> standard -> gemini (with tools) -> primary only at the last moment', async () => {
    const scripts = [
      // Turn 1: simple -> trivial (gpt-oss:20b-cloud)
      { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.05 }, textOnly: 0.95 },
      // Turn 2: level-2 -> standard (gemma4:31b-cloud)
      { simple: 0.3, standard: 0.55, structural: 0.15, risk: 0.9, riskScores: { security_sensitive: 0.02, destructive_or_production: 0.02, requires_inspection: 0.9 }, textOnly: 0.05 },
      // Turn 3: structural requiring tools -> gemini (gemini-3.8-flash-high), keeps session sticky at standard
      { simple: 0.05, standard: 0.1, structural: 0.85, risk: 0.95, riskScores: { security_sensitive: 0.05, destructive_or_production: 0.05, requires_inspection: 0.95 }, textOnly: 0.02 },
      // Turn 4: simple follow-up -> stays at standard (not pinned to primary!)
      { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.05 }, textOnly: 0.9 },
      // Turn 5: security/destructive high risk -> finally escalates to primary (Claude) at the last moment
      { simple: 0.05, standard: 0.1, structural: 0.85, risk: 0.95, riskScores: { security_sensitive: 0.95, destructive_or_production: 0.1, requires_inspection: 0.9 }, textOnly: 0.02 },
    ];
    const classifier = { name: 'test', classify: async () => scripts.shift()! };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: true, minStandardProbability: 0.75 },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      standardContextTokens: 250_000,
      geminiPolicy: { enabled: true, minTextOnly: 0, pressureMinTextOnly: 0 },
      geminiFromPrimary: true,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota: { level: () => 'none' },
    });

    const sid = { 'x-claude-code-session-id': 's-ladder' };
    const m1 = [{ role: 'user', content: 't1' }];
    const d1 = await router.decide(req({ messages: m1 }, sid) as any);
    assert.deepEqual([d1.route, d1.tier, d1.reason], ['cheap', 'trivial', 'classified']);

    const m2 = [...m1, { role: 'assistant', content: 'a1' }, { role: 'user', content: 't2' }];
    const d2 = await router.decide(req({ messages: m2 }, sid) as any);
    assert.deepEqual([d2.route, d2.tier, d2.reason], ['cheap', 'standard', 'escalated:standard']);

    const m3 = [...m2, { role: 'assistant', content: 'a2' }, { role: 'user', content: 't3' }];
    const d3 = await router.decide(req({ messages: m3 }, sid) as any);
    assert.deepEqual([d3.route, d3.tier, d3.fallbackTier, d3.reason], ['gemini', 'gemini', 'standard', 'gemini:text-only']);
    assert.equal(store.get('s-ladder:main'), 'standard');

    // Tool-result continuation of Turn 3 stays on gemini!
    const m3Tool = [...m3, { role: 'assistant', content: [{ type: 'tool_use', id: '1', name: 'Read', input: {} }] }, { role: 'user', content: [{ type: 'tool_result', tool_use_id: '1', content: 'ok' }] }];
    const d3Tool = await router.decide(req({ messages: m3Tool }, sid) as any);
    assert.deepEqual([d3Tool.route, d3Tool.tier, d3Tool.fallbackTier, d3Tool.reason], ['gemini', 'gemini', 'standard', 'sticky']);

    const m4 = [...m3Tool, { role: 'assistant', content: 'a3' }, { role: 'user', content: 't4' }];
    const d4 = await router.decide(req({ messages: m4 }, sid) as any);
    assert.deepEqual([d4.route, d4.tier, d4.reason], ['cheap', 'standard', 'sticky']);

    const m5 = [...m4, { role: 'assistant', content: 'a4' }, { role: 'user', content: 't5' }];
    const d5 = await router.decide(req({ messages: m5 }, sid) as any);
    assert.deepEqual([d5.route, d5.tier, d5.reason], ['primary', 'primary', 'escalated']);
  });
});
