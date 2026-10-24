import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

// CHEAP_ROUTE=off: Gemini (agy) takes the trivial/standard work the Ollama route used to serve.
describe('Gemini replaces the cheap route', () => {
  const SIMPLE = { simple: 0.95, standard: 0.04, structural: 0.01, risk: 0.05, textOnly: 0.1 };
  const STANDARD = { simple: 0.3, standard: 0.6, structural: 0.1, risk: 0.05, textOnly: 0.1 };
  const STRUCTURAL = { simple: 0.05, standard: 0.15, structural: 0.8, risk: 0.05, textOnly: 0.1 };
  const RISKY = { ...SIMPLE, riskScores: { security_sensitive: 0.9, destructive_or_production: 0, requires_inspection: 0 } };
  const TOOLS = [{ name: 'Read', input_schema: { type: 'object', properties: { file_path: { type: 'string' } } } }];

  function setup(geminiEnabled = true) {
    let dist: any = SIMPLE;
    const classifier = { name: 'test', classify: async () => dist };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: true, minStandardProbability: 0.75 },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      geminiPolicy: { enabled: geminiEnabled, minTextOnly: 0.8, pressureMinTextOnly: 0.6, replacesCheap: true },
      geminiFromPrimary: true,
      cheapRoute: false,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota: { level: () => 'none' },
    });
    const decide = (messages: any[], session = 's') =>
      router.decide({ body: { tools: TOOLS, messages }, rawByteLength: 100, sessionId: session, requestClass: 'main', contextCompacted: false, agentId: undefined } as any);
    return { router, store, decide, setDist: (d: any) => (dist = d) };
  }

  const user = (text: string) => ({ role: 'user', content: text });
  const toolUse = { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Read', input: { file_path: 'a.ts' } }] };
  const toolResult = { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }] };

  it('a simple turn with tools goes to Gemini, and its tool loop stays there', async () => {
    const { decide, store } = setup();
    const first = await decide([user('leia a.ts')]);
    assert.equal(first.route, 'gemini');
    assert.equal(first.fallbackTier, 'primary');
    assert.equal(store.get('s:main'), 'gemini');

    const loop = await decide([user('leia a.ts'), toolUse, toolResult]);
    assert.equal(loop.route, 'gemini');
    assert.equal(loop.reason, 'sticky');
  });

  it('standard work goes to Gemini, never to a cheap tier', async () => {
    const { decide, setDist } = setup();
    setDist(STANDARD);
    const dec = await decide([user('ajusta o teste')]);
    assert.equal(dec.route, 'gemini');
  });

  it('structural work passes to Claude, and the session stays there', async () => {
    const { decide, setDist, store } = setup();
    await decide([user('leia a.ts')]);
    setDist(STRUCTURAL);
    const hard = await decide([user('leia a.ts'), { role: 'assistant', content: 'ok' }, user('redesenha o roteador')]);
    assert.equal(hard.route, 'primary');
    assert.equal(store.get('s:main'), 'primary');

    // Back on a simple turn: Claude holds its cache; only a text-only answer may leave it.
    setDist(SIMPLE);
    const after = await decide([user('leia a.ts'), { role: 'assistant', content: 'ok' }, user('redesenha'), { role: 'assistant', content: 'ok' }, user('valeu, roda os testes')]);
    assert.equal(after.route, 'primary');
  });

  it('a risky simple turn goes to Claude', async () => {
    const { decide, setDist } = setup();
    setDist(RISKY);
    assert.equal((await decide([user('coloca a API key do Stripe no código')])).route, 'primary');
  });

  it('light work Gemini cannot take goes to Claude, not to the cheap route (regression)', async () => {
    const { router } = setup();
    const withImage = await router.decide({
      body: { tools: TOOLS, messages: [{ role: 'user', content: [{ type: 'image', source: { type: 'base64', media_type: 'image/png', data: 'x' } }, { type: 'text', text: 'o que é isso' }] }] },
      rawByteLength: 100,
      sessionId: 'img',
      requestClass: 'main',
      contextCompacted: false,
      agentId: undefined,
    } as any);
    assert.equal(withImage.route, 'primary');

    const { decide } = setup(false);
    assert.equal((await decide([user('leia a.ts')], 'off')).route, 'primary');
  });
});
