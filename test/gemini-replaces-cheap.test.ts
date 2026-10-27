import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

describe('Gemini replaces the cheap route (Free tier)', () => {
  const SIMPLE = { simple: 0.95, standard: 0.04, structural: 0.01, risk: 0.05, textOnly: 0.1 };
  const STANDARD = { simple: 0.3, standard: 0.6, structural: 0.1, risk: 0.05, textOnly: 0.1 };
  const STRUCTURAL = { simple: 0.05, standard: 0.15, structural: 0.8, risk: 0.05, textOnly: 0.1 };
  const RISKY = { ...SIMPLE, riskScores: { security_sensitive: 0.9, destructive_or_production: 0, requires_inspection: 0 } };
  const TOOLS = [{ name: 'Read', input_schema: { type: 'object', properties: { file_path: { type: 'string' } } } }];

  function setup(argonEnabled = true) {
    let dist: any = SIMPLE;
    const classifier = { name: 'test', classify: async () => dist };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.9, minStandardProbability: 0.75, argonEnabled },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
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

  it('a simple turn with tools goes to Free tier (Gemini Flash), and its tool loop stays there', async () => {
    const { decide, store } = setup();
    const first = await decide([user('leia a.ts')]);
    assert.equal(first.route, 'free');
    assert.equal(first.fallbackTier, 'primary');
    assert.equal(store.get('s:main'), 'free');

    const loop = await decide([user('leia a.ts'), toolUse, toolResult]);
    assert.equal(loop.route, 'free');
    assert.equal(loop.reason, 'sticky');
  });

  it('standard work goes to Free tier, never to a cheap tier', async () => {
    const { decide, setDist } = setup();
    setDist(STANDARD);
    const dec = await decide([user('ajusta o teste')]);
    assert.equal(dec.route, 'free');
  });

  it('structural work passes to Argon tier (or Claude when Argon is off)', async () => {
    const { decide, setDist, store } = setup();
    await decide([user('leia a.ts')]);
    setDist(STRUCTURAL);
    const hard = await decide([user('leia a.ts'), { role: 'assistant', content: 'ok' }, user('redesenha o roteador')]);
    assert.equal(hard.route, 'argon');
    assert.equal(store.get('s:main'), 'argon');

    const { decide: decideNoArgon, setDist: setDistNoArgon } = setup(false);
    setDistNoArgon(STRUCTURAL);
    const hardClaude = await decideNoArgon([user('redesenha o roteador')], 's-claude');
    assert.equal(hardClaude.route, 'primary');
  });

  it('a risky simple turn goes to Claude', async () => {
    const { decide, setDist } = setup();
    setDist(RISKY);
    assert.equal((await decide([user('coloca a API key do Stripe no código')])).route, 'primary');
  });
});
