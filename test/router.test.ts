import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import type { ClassificationInput, ComplexityClassifier } from '../src/classifier/classifier.js';
import type { ComplexityDistribution } from '../src/domain/complexity.js';
import type { Tier } from '../src/domain/policy.js';
import type { QuotaLevel } from '../src/quota.js';
import { MessagesBody, lastHumanText } from '../src/routing/messages-body.js';
import { Router, type RequestContext } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

const SIMPLE: ComplexityDistribution = { simple: 0.92, standard: 0.06, structural: 0.02 };
const STRUCTURAL: ComplexityDistribution = { simple: 0.03, standard: 0.12, structural: 0.85 };

class ScriptedClassifier implements ComplexityClassifier {
  readonly name = 'scripted';
  calls: ClassificationInput[] = [];
  constructor(private readonly next: () => ComplexityDistribution | Error) {}
  classify(input: ClassificationInput): Promise<ComplexityDistribution> {
    this.calls.push(input);
    const r = this.next();
    return r instanceof Error ? Promise.reject(r) : Promise.resolve(r);
  }
}

function setup(script: () => ComplexityDistribution | Error, { argon = true } = {}) {
  const classifier = new ScriptedClassifier(script);
  const router = new Router(classifier, new TtlLruStore<Tier>(100, 60_000), {
    policy: { minCheapProbability: 0.8, argonEnabled: argon },
    primaryClasses: new Set(['auxiliary', 'compaction']),
    cheapContextTokens: 100_000,
    classifierTimeoutMs: 1_000,
    classifierMaxChars: 4_000,
  });
  return { classifier, router };
}

/** Level-2 work: not confidently simple, but little structural mass. */
const STANDARD: ComplexityDistribution = { simple: 0.4, standard: 0.5, structural: 0.1 };

const user = (text: string) => ({ role: 'user', content: [{ type: 'text', text }] });
const assistantToolUse = { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Read', input: {} }] };
const toolResult = { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }] };
const assistantText = { role: 'assistant', content: [{ type: 'text', text: 'done' }] };

function ctx(messages: unknown[], overrides: Partial<RequestContext> = {}): RequestContext {
  return {
    body: MessagesBody.parse({ model: 'claude-opus-5-5', messages }),
    rawByteLength: 30_000,
    sessionId: 'S1',
    agentId: undefined,
    requestClass: 'main',
    contextCompacted: false,
    ...overrides,
  };
}

const body = (messages: unknown[]) => MessagesBody.parse({ model: 'claude-opus-5-5', messages });

/**
 * First turn as Claude Code 2.1.289 actually sends it (captured, trimmed): context
 * reminders, then the typed prompt, then SessionStart hook output as an inline
 * `system` message closing the array.
 */
const realFirstTurn = [
  {
    role: 'user',
    content: [
      { type: 'text', text: '<system-reminder>\nCodebase and user instructions…\n</system-reminder>' },
      { type: 'text', text: 'explain this hook' },
    ],
  },
  {
    role: 'system',
    content: [
      { type: 'text', text: 'SessionStart:startup hook success: PONYTAIL MODE ACTIVE' },
      { type: 'tool_addition', tool: { type: 'tool_reference', name: 'advisor' } },
    ],
  },
];
const hookOutput = { role: 'system', content: [{ type: 'text', text: 'UserPromptSubmit hook success' }] };

describe('Router (multi-provider)', () => {
  it('routes a simple fresh conversation to the free tier', async () => {
    const { router } = setup(() => SIMPLE);
    const d = await router.decide(ctx([user('explain this hook')]));
    assert.deepEqual([d.route, d.reason], ['free', 'free:flash']);
  });

  it('classifies a real first turn whose hook output trails as a system message', async () => {
    const { router, classifier } = setup(() => SIMPLE);
    const d = await router.decide(ctx(realFirstTurn));
    assert.deepEqual([d.route, d.reason], ['free', 'free:flash']);
    assert.equal(classifier.calls[0]?.text, 'explain this hook');
  });

  it('re-classifies a new human turn followed by hook output', async () => {
    const { router, classifier } = setup(() => SIMPLE);
    await router.decide(ctx(realFirstTurn));
    await router.decide(ctx([...realFirstTurn, assistantText, user('now rename it'), hookOutput]));
    assert.equal(classifier.calls[1]?.text, 'now rename it');
  });

  it('routes structural work to the argon tier', async () => {
    const { router } = setup(() => STRUCTURAL);
    const d = await router.decide(ctx([user('refactor to clean architecture')]));
    assert.deepEqual([d.route, d.reason], ['argon', 'argon:high']);
  });

  it('routes structural work to primary when argon is disabled', async () => {
    const { router } = setup(() => STRUCTURAL, { argon: false });
    const d = await router.decide(ctx([user('refactor to clean architecture')]));
    assert.deepEqual([d.route, d.reason], ['primary', 'classified']);
  });

  it('keeps the agent loop on its route without re-classifying', async () => {
    const { router, classifier } = setup(() => SIMPLE);
    await router.decide(ctx([user('rename x')]));
    const d = await router.decide(ctx([user('rename x'), assistantToolUse, toolResult]));
    assert.deepEqual([d.route, d.reason], ['free', 'sticky']);
    assert.equal(classifier.calls.length, 1);
  });

  it('escalates a free conversation on a structural turn, then stays on escalated route without asking JEV', async () => {
    const scripts = [SIMPLE, STRUCTURAL];
    const { router, classifier } = setup(() => scripts.shift()!);
    await router.decide(ctx([user('rename x')]));

    const up = await router.decide(ctx([user('rename x'), assistantText, user('now redesign the layers')]));
    assert.deepEqual([up.route, up.reason], ['argon', 'argon:high']);

    const stay = await router.decide(ctx([user('rename x'), assistantText, user('now redesign the layers'), assistantToolUse, toolResult]));
    assert.deepEqual([stay.route, stay.reason], ['argon', 'sticky']);
    assert.equal(classifier.calls.length, 2);
  });

  it('fails toward the primary when JEV errors', async () => {
    const { router } = setup(() => new Error('timeout'));
    const d = await router.decide(ctx([user('anything')]));
    assert.deepEqual([d.route, d.reason], ['primary', 'passthrough:classifier-failed']);
  });

  it('escalates a free conversation when JEV fails on a new human turn', async () => {
    let script: () => ComplexityDistribution | Error = () => SIMPLE;
    const { router } = setup(() => script());
    assert.equal((await router.decide(ctx([user('fix the typo')]))).route, 'free');

    script = () => new Error('timeout');
    const d = await router.decide(ctx([user('fix the typo'), assistantText, user('now redesign the module')]));
    assert.deepEqual([d.route, d.reason], ['primary', 'passthrough:classifier-failed']);

    const next = await router.decide(ctx([user('fix the typo'), assistantText, user('now redesign the module'), assistantToolUse, toolResult]));
    assert.deepEqual([next.route, next.reason], ['primary', 'sticky'], 'primary is terminal');
  });

  it('keeps auxiliary and compaction traffic on the primary', async () => {
    const { router, classifier } = setup(() => SIMPLE);
    const d = await router.decide(ctx([user('title this')], { requestClass: 'auxiliary' }));
    assert.deepEqual([d.route, d.reason], ['primary', 'passthrough:request-class']);
    assert.equal(classifier.calls.length, 0);
  });

  it('treats unknown ongoing conversations as primary', async () => {
    const { router } = setup(() => SIMPLE);
    const d = await router.decide(ctx([user('a'), assistantText, user('b')]));
    assert.deepEqual([d.route, d.reason], ['primary', 'sticky']);
  });

  it('escalates when the conversation outgrows context budget', async () => {
    const { router } = setup(() => SIMPLE);
    await router.decide(ctx([user('small')]));
    const d = await router.decide(ctx([user('small'), assistantToolUse, toolResult], { rawByteLength: 600_000 }));
    assert.deepEqual([d.route, d.reason], ['primary', 'escalated:context']);
  });

  it('routes subagents independently', async () => {
    const scripts = [STRUCTURAL, SIMPLE];
    const { router } = setup(() => scripts.shift()!);
    await router.decide(ctx([user('architect it')]));
    const sub = await router.decide(ctx([user('grep for foo')], { agentId: 'agent-1', requestClass: 'subagent' }));
    assert.equal(sub.route, 'free');
  });

  it('pinToPrimary moves a conversation to the primary for good', async () => {
    const { router } = setup(() => SIMPLE);
    const d = await router.decide(ctx([user('rename x')]));
    router.pinToPrimary(d);
    const next = await router.decide(ctx([user('rename x'), assistantToolUse, toolResult]));
    assert.equal(next.route, 'primary');
  });

  it('strips system-reminders before classifying', async () => {
    const { router, classifier } = setup(() => SIMPLE);
    await router.decide(ctx([user('<system-reminder>huge context</system-reminder>do X')]));
    assert.equal(classifier.calls[0]?.text, 'do X');
  });
});

describe('Router under quota pressure', () => {
  /** P(simple) 0.75, standard 0.05, structural 0.2: below the 0.8 bar. */
  const BORDERLINE: ComplexityDistribution = { simple: 0.75, standard: 0.05, structural: 0.2 };
  const RISKY = { ...SIMPLE, risk: 0.9 };

  function quotaSetup(script: () => ComplexityDistribution | Error, quota: { level: QuotaLevel } | undefined) {
    const classifier = new ScriptedClassifier(script);
    const router = new Router(classifier, new TtlLruStore<Tier>(100, 60_000), {
      policy: { minCheapProbability: 0.8, minStandardProbability: 0.9, argonEnabled: false },
      primaryClasses: new Set(['auxiliary', 'compaction']),
      cheapContextTokens: 100_000,
      classifierTimeoutMs: 1_000,
      classifierMaxChars: 4_000,
      ...(quota ? { quota: { level: () => quota.level }, pressurePolicy: { minCheapProbability: 0.7, minStandardProbability: 0.8 } } : {}),
    });
    return { classifier, router };
  }

  it('lowers the cheap bar under pressure, and says so in the reason', async () => {
    const quota = { level: 'none' as QuotaLevel };
    const { router } = quotaSetup(() => BORDERLINE, quota);
    assert.equal((await router.decide(ctx([user('rename x')], { sessionId: 'A' }))).route, 'primary');
    quota.level = 'pressure';
    const d = await router.decide(ctx([user('rename x')], { sessionId: 'B' }));
    assert.deepEqual([d.route, d.reason], ['free', 'quota:pressure']);
  });

  it('keeps auxiliary traffic on Claude even when critical', async () => {
    const { router } = quotaSetup(() => SIMPLE, { level: 'critical' });
    const d = await router.decide(ctx([user('title this')], { requestClass: 'auxiliary' }));
    assert.deepEqual([d.route, d.reason], ['primary', 'passthrough:request-class']);
  });

  it('when critical, re-classifies a new human turn of a Claude-pinned session and keeps it free after', async () => {
    const scripts = [STRUCTURAL, SIMPLE];
    const quota = { level: 'none' as QuotaLevel };
    const { router, classifier } = quotaSetup(() => scripts.shift()!, quota);
    await router.decide(ctx([user('redesign the layers')]));

    quota.level = 'critical';
    const t2 = [user('redesign the layers'), assistantText, user('now rename x')];
    const down = await router.decide(ctx(t2));
    assert.deepEqual([down.route, down.reason], ['free', 'quota:reclassified']);

    quota.level = 'none';
    const loop = await router.decide(ctx([...t2, assistantToolUse, toolResult]));
    assert.deepEqual([loop.route, loop.reason], ['free', 'sticky'], 'going back would rebuild the Claude cache');
    assert.equal(classifier.calls.length, 2);
  });

  it('never lets a risky turn leave Claude, even when critical', async () => {
    const scripts: (ComplexityDistribution | Error)[] = [STRUCTURAL, RISKY];
    const { router } = quotaSetup(() => scripts.shift()!, { level: 'critical' });
    await router.decide(ctx([user('redesign the layers')]));
    const d = await router.decide(ctx([user('redesign the layers'), assistantText, user('put the Stripe key in the code')]));
    assert.equal(d.route, 'primary');
  });

  it('keeps a Claude session on Claude after compaction, unless critical', async () => {
    for (const level of ['none', 'pressure'] as const) {
      const scripts = [STRUCTURAL, SIMPLE];
      const { router, classifier } = quotaSetup(() => scripts.shift()!, { level });
      await router.decide(ctx([user('redesign the layers')], { sessionId: `C-${level}` }));
      const d = await router.decide(ctx([user('<summary of the session>')], { sessionId: `C-${level}`, contextCompacted: true }));
      assert.deepEqual([d.route, d.reason], ['primary', 'pinned:compaction'], level);
      assert.equal(classifier.calls.length, 1, 'no JEV call');
    }
    const scripts = [STRUCTURAL, SIMPLE];
    const { router } = quotaSetup(() => scripts.shift()!, { level: 'critical' });
    await router.decide(ctx([user('redesign the layers')], { sessionId: 'C-crit' }));
    const d = await router.decide(ctx([user('<summary of the session>')], { sessionId: 'C-crit', contextCompacted: true }));
    assert.deepEqual([d.route, d.reason], ['free', 'quota:reclassified']);
  });

  it('behaves as before without quota routing (QUOTA_ROUTING=false)', async () => {
    const { router } = quotaSetup(() => BORDERLINE, undefined);
    assert.equal((await router.decide(ctx([user('rename x')]))).route, 'primary');
  });
});

describe('lastHumanText', () => {
  it('returns the most recent human text', () => {
    assert.equal(lastHumanText(body([user('first'), assistantText, user('second')])), 'second');
  });

  it('falls back to the human request driving a tool loop', () => {
    assert.equal(lastHumanText(body([user('first'), assistantToolUse, toolResult])), 'first');
  });

  it('skips a trailing user turn with no human text', () => {
    assert.equal(lastHumanText(body([user('first'), assistantToolUse, toolResult, assistantToolUse, toolResult])), 'first');
  });

  it('returns undefined when there is no human text', () => {
    assert.equal(lastHumanText(body([assistantText])), undefined);
    assert.equal(lastHumanText(body([assistantToolUse, toolResult])), undefined);
  });
});

describe('Router.cascadeRiskVeto', () => {
  it('vetoes when the classifier scores the last human text risky', async () => {
    const { router } = setup(() => ({ ...SIMPLE, risk: 0.9 }));
    assert.equal(await router.cascadeRiskVeto(body([user('run this dangerous command')])), true);
  });

  it('does not veto a benign turn', async () => {
    const { router } = setup(() => ({ ...SIMPLE, risk: 0.1 }));
    assert.equal(await router.cascadeRiskVeto(body([user('fix the typo')])), false);
  });

  it('classifies the driving human text of a tool continuation', async () => {
    const { router, classifier } = setup(() => ({ ...SIMPLE, risk: 0.85 }));
    const veto = await router.cascadeRiskVeto(body([user('inspect the prod db'), assistantToolUse, toolResult]));
    assert.equal(veto, true);
    assert.equal(classifier.calls[0]?.text, 'inspect the prod db');
  });

  it('returns false without classifying when there is no human text', async () => {
    const { router, classifier } = setup(() => SIMPLE);
    assert.equal(await router.cascadeRiskVeto(body([assistantText])), false);
    assert.equal(classifier.calls.length, 0);
  });

  it('counts classifier failure as risky (fail toward primary)', async () => {
    const { router } = setup(() => new Error('jev down'));
    assert.equal(await router.cascadeRiskVeto(body([user('anything')])), true);
  });
});
