import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { parseJevRisk } from '../src/classifier/jev-classifier.js';
import { toDistribution } from '../src/domain/complexity.js';
import { selectRoute, selectTier, stickyRoute, stickyTier, primaryProfile } from '../src/domain/policy.js';

const options = { minCheapProbability: 0.8 } as const;

describe('selectRoute', () => {
  it('sends confidently simple work to the free tier', () => {
    assert.equal(selectRoute({ simple: 0.9, standard: 0.08, structural: 0.02 }, options), 'free');
  });

  it('routes non-vetoed structural work to the argon tier', () => {
    assert.equal(selectRoute({ simple: 0.05, standard: 0.15, structural: 0.8 }, options), 'argon');
  });

  it('routes structural work to primary when argon is disabled', () => {
    assert.equal(selectRoute({ simple: 0.05, standard: 0.15, structural: 0.8 }, { ...options, argonEnabled: false }), 'primary');
  });

  it('keeps hard risk vetoed work on primary', () => {
    assert.equal(selectRoute({ simple: 0.05, standard: 0.15, structural: 0.8, risk: 0.9 }, options), 'primary');
  });
});

describe('selectRoute risk veto', () => {
  it('keeps confidently simple but risky work on the primary', () => {
    const d = { simple: 0.96, standard: 0.03, structural: 0.01 };
    assert.equal(selectRoute({ ...d, risk: 0.94 }, options), 'primary');
    assert.equal(selectRoute({ ...d, risk: 0.15 }, options), 'free');
    assert.equal(selectRoute(d, options), 'free');
  });

  it('vetoes a simple-looking question that needs the repository to answer ("is everything ok here?")', () => {
    const answer = (inspection: number) => ({
      answers: {
        security_sensitive: { noul: 0.02 },
        destructive_or_production: { noul: 0.03 },
        requires_inspection: { noul: inspection },
      },
    });
    const d = { simple: 0.91, standard: 0.07, structural: 0.02 };
    assert.equal(selectRoute({ ...d, risk: parseJevRisk(answer(0.88)) }, options), 'primary');
    assert.equal(selectRoute({ ...d, risk: parseJevRisk(answer(0.1)) }, options), 'free', 'the facts are in the prompt');
  });
});

describe('selectTier', () => {
  const tiers = { ...options, minCheapProbability: 0.9, standardEnabled: true, minStandardProbability: 0.75 } as const;

  it('sends confident simple work to the free tier, the rest to argon or primary', () => {
    assert.equal(selectTier({ simple: 0.93, standard: 0.05, structural: 0.02 }, tiers), 'free');
    assert.equal(selectTier({ simple: 0.5, standard: 0.3, structural: 0.2 }, tiers), 'free'); // 0.5+0.3 >= 0.75 standard bar
    assert.equal(selectTier({ simple: 0.3, standard: 0.3, structural: 0.4 }, tiers), 'argon');
    assert.equal(selectTier({ simple: 0.3, standard: 0.3, structural: 0.4, risk: 0.9 }, tiers), 'primary');
  });

  it('lets the risk veto beat every tier', () => {
    assert.equal(selectTier({ simple: 0.5, standard: 0.45, structural: 0.05, risk: 0.6 }, tiers), 'primary');
    assert.equal(selectTier({ simple: 0.99, standard: 0.01, structural: 0, risk: 0.6 }, tiers), 'primary');
  });

  it('routes to primary when argon is disabled and turn is not free', () => {
    const off = { ...tiers, argonEnabled: false };
    assert.equal(selectTier({ simple: 0.3, standard: 0.3, structural: 0.4 }, off), 'primary');
  });
});

describe('stickyTier', () => {
  it('only moves up: free < argon < primary', () => {
    assert.equal(stickyTier('free', 'argon'), 'argon');
    assert.equal(stickyTier('argon', 'free'), 'argon');
    assert.equal(stickyTier('argon', 'primary'), 'primary');
    assert.equal(stickyTier('primary', 'free'), 'primary');
  });
});

describe('stickyRoute', () => {
  it('escalates but never de-escalates', () => {
    assert.equal(stickyRoute('free', 'primary'), 'primary');
    assert.equal(stickyRoute('primary', 'free'), 'primary');
    assert.equal(stickyRoute('free', 'free'), 'free');
    assert.equal(stickyRoute('argon', 'primary'), 'primary');
  });
});

describe('toDistribution', () => {
  it('normalizes and rejects an empty distribution', () => {
    assert.deepEqual(toDistribution(2, 1, 1), { simple: 0.5, standard: 0.25, structural: 0.25 });
    assert.throws(() => toDistribution(0, 0, 0));
  });
});

describe('primaryProfile policy and escalation', () => {
  const OPUS = 'claude-opus-5-5';
  const SONNET = 'claude-sonnet-5-5';

  it('matches base table first-match rows', () => {
    // 1. hardRiskVeto(d) && structural >= 0.6 -> opus xhigh
    assert.deepEqual(
      primaryProfile({ simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.9 }),
      { model: OPUS, effort: 'xhigh' }
    );
    // 2. hardRiskVeto(d) || structural >= 0.6 -> opus high
    assert.deepEqual(
      primaryProfile({ simple: 0.1, standard: 0.2, structural: 0.7 }),
      { model: OPUS, effort: 'high' }
    );
    assert.deepEqual(
      primaryProfile({ simple: 0.8, standard: 0.15, structural: 0.05, risk: 0.9 }),
      { model: OPUS, effort: 'high' }
    );
    // 3. structural >= 0.45 || requires_inspection >= 0.5 -> opus medium
    assert.deepEqual(
      primaryProfile({ simple: 0.2, standard: 0.3, structural: 0.5 }),
      { model: OPUS, effort: 'medium' }
    );
    assert.deepEqual(
      primaryProfile({
        simple: 0.8,
        standard: 0.15,
        structural: 0.05,
        riskScores: { security_sensitive: 0.1, destructive_or_production: 0.1, requires_inspection: 0.6 },
      }),
      { model: OPUS, effort: 'medium' }
    );
    // 4. structural >= 0.3 -> sonnet high
    assert.deepEqual(
      primaryProfile({ simple: 0.3, standard: 0.3, structural: 0.4 }),
      { model: SONNET, effort: 'high' }
    );
    // 5. simple >= 0.6 -> sonnet low
    assert.deepEqual(
      primaryProfile({ simple: 0.7, standard: 0.2, structural: 0.1 }),
      { model: SONNET, effort: 'low' }
    );
    // 6. else -> sonnet medium
    assert.deepEqual(
      primaryProfile({ simple: 0.4, standard: 0.5, structural: 0.1 }),
      { model: SONNET, effort: 'medium' }
    );
  });

  it('adjusts by bumping 1 step for gemini-failed and escalated-from-gemini', () => {
    const simple = { simple: 0.7, standard: 0.2, structural: 0.1 }; // base: sonnet low
    assert.deepEqual(primaryProfile(simple, 'gemini-failed'), { model: SONNET, effort: 'medium' });
    assert.deepEqual(primaryProfile(simple, 'escalated-from-gemini'), { model: SONNET, effort: 'medium' });

    const standard = { simple: 0.4, standard: 0.5, structural: 0.1 }; // base: sonnet medium
    assert.deepEqual(primaryProfile(standard, 'gemini-failed'), { model: SONNET, effort: 'high' });

    const lightStruct = { simple: 0.3, standard: 0.3, structural: 0.4 }; // base: sonnet high
    assert.deepEqual(primaryProfile(lightStruct, 'gemini-failed'), { model: SONNET, effort: 'xhigh' });

    const midStruct = { simple: 0.2, standard: 0.3, structural: 0.5 }; // base: opus medium
    assert.deepEqual(primaryProfile(midStruct, 'gemini-failed'), { model: OPUS, effort: 'high' });

    const highStruct = { simple: 0.1, standard: 0.2, structural: 0.7 }; // base: opus high
    assert.deepEqual(primaryProfile(highStruct, 'gemini-failed'), { model: OPUS, effort: 'xhigh' });

    const maxStruct = { simple: 0.1, standard: 0.2, structural: 0.7, risk: 0.9 }; // base: opus xhigh
    assert.deepEqual(primaryProfile(maxStruct, 'gemini-failed'), { model: OPUS, effort: 'xhigh' }); // capped
  });

  it('no bump for gemini-busy and gemini-unhealthy', () => {
    const d = { simple: 0.4, standard: 0.5, structural: 0.1 };
    assert.deepEqual(primaryProfile(d, 'gemini-busy'), { model: SONNET, effort: 'medium' });
    assert.deepEqual(primaryProfile(d, 'gemini-unhealthy'), { model: SONNET, effort: 'medium' });
  });

  it('context escalation gives opus low for simple >= 0.6, otherwise base', () => {
    assert.deepEqual(
      primaryProfile({ simple: 0.7, standard: 0.2, structural: 0.1 }, 'context'),
      { model: OPUS, effort: 'low' }
    );
    assert.deepEqual(
      primaryProfile({ simple: 0.4, standard: 0.5, structural: 0.1 }, 'context'),
      { model: SONNET, effort: 'medium' }
    );
  });
});
