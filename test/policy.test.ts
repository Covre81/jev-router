import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { parseJevRisk } from '../src/classifier/jev-classifier.js';
import { toDistribution } from '../src/domain/complexity.js';
import { selectRoute, selectTier, stickyRoute, stickyTier } from '../src/domain/policy.js';

const options = { minCheapProbability: 0.8, standardRoute: 'primary' } as const;

describe('selectRoute', () => {
  it('sends confidently simple work to the cheap provider', () => {
    assert.equal(selectRoute({ simple: 0.9, standard: 0.08, structural: 0.02 }, options), 'cheap');
  });

  it('keeps structural work on the primary', () => {
    assert.equal(selectRoute({ simple: 0.05, standard: 0.15, structural: 0.8 }, options), 'primary');
  });

  it('is not argmax: a weakly simple mode stays on the primary', () => {
    assert.equal(selectRoute({ simple: 0.45, standard: 0.3, structural: 0.25 }, options), 'primary');
  });

  it('can opt level-2 work into the cheap provider', () => {
    const d = { simple: 0.4, standard: 0.45, structural: 0.15 };
    assert.equal(selectRoute(d, options), 'primary');
    assert.equal(selectRoute(d, { ...options, standardRoute: 'cheap' }), 'cheap');
  });
});

describe('selectRoute risk veto', () => {
  it('keeps confidently simple but risky work on the primary', () => {
    const d = { simple: 0.96, standard: 0.03, structural: 0.01 };
    assert.equal(selectRoute({ ...d, risk: 0.94 }, options), 'primary');
    assert.equal(selectRoute({ ...d, risk: 0.15 }, options), 'cheap');
    assert.equal(selectRoute(d, options), 'cheap');
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
    assert.equal(selectRoute({ ...d, risk: parseJevRisk(answer(0.1)) }, options), 'cheap', 'the facts are in the prompt');
  });
});

describe('selectTier', () => {
  const tiers = { ...options, minCheapProbability: 0.9, standardEnabled: true, minStandardProbability: 0.75 } as const;

  it('sends confident simple work to the trivial tier, simple+standard mass to the standard tier, the rest to primary', () => {
    assert.equal(selectTier({ simple: 0.93, standard: 0.05, structural: 0.02 }, tiers), 'trivial');
    assert.equal(selectTier({ simple: 0.5, standard: 0.3, structural: 0.2 }, tiers), 'standard');
    assert.equal(selectTier({ simple: 0.3, standard: 0.3, structural: 0.4 }, tiers), 'primary');
  });

  it('lets the risk veto beat every tier, while escalating inspection-only risk to standard when enabled', () => {
    assert.equal(selectTier({ simple: 0.5, standard: 0.45, structural: 0.05, risk: 0.6 }, tiers), 'primary');
    assert.equal(selectTier({ simple: 0.99, standard: 0.01, structural: 0, risk: 0.6 }, tiers), 'primary');
    assert.equal(
      selectTier(
        {
          simple: 0.95,
          standard: 0,
          structural: 0.05,
          risk: 0.55,
          riskScores: { security_sensitive: 0.04, destructive_or_production: 0.03, requires_inspection: 0.55 },
        },
        tiers,
      ),
      'standard',
    );
    assert.equal(
      selectTier(
        {
          simple: 0.95,
          standard: 0,
          structural: 0.05,
          risk: 0.8,
          riskScores: { security_sensitive: 0.8, destructive_or_production: 0.03, requires_inspection: 0.1 },
        },
        tiers,
      ),
      'primary',
    );
  });

  it('behaves like before when the standard tier is off', () => {
    const off = { ...tiers, standardEnabled: false };
    assert.equal(selectTier({ simple: 0.5, standard: 0.45, structural: 0.05 }, off), 'primary');
    assert.equal(selectTier({ simple: 0.5, standard: 0.45, structural: 0.05 }, { ...off, standardRoute: 'cheap' }), 'trivial');
    assert.equal(selectRoute({ simple: 0.5, standard: 0.45, structural: 0.05 }, tiers), 'cheap', 'selectRoute still says cheap');
  });
});

describe('stickyTier', () => {
  it('only moves up: trivial < standard < primary', () => {
    assert.equal(stickyTier('trivial', 'standard'), 'standard');
    assert.equal(stickyTier('standard', 'trivial'), 'standard');
    assert.equal(stickyTier('standard', 'primary'), 'primary');
    assert.equal(stickyTier('primary', 'trivial'), 'primary');
  });

  it('is total and keeps gemini out of ranking', () => {
    assert.equal(stickyTier('gemini', 'trivial'), 'trivial');
    assert.equal(stickyTier('standard', 'gemini'), 'standard');
    assert.equal(stickyTier('gemini', 'primary'), 'primary');
    assert.equal(stickyTier('gemini', 'gemini'), 'primary');
    assert.equal(stickyTier('cursor', 'cursor'), 'primary');
  });
});

describe('stickyRoute', () => {
  it('escalates but never de-escalates', () => {
    assert.equal(stickyRoute('cheap', 'primary'), 'primary');
    assert.equal(stickyRoute('primary', 'cheap'), 'primary');
    assert.equal(stickyRoute('cheap', 'cheap'), 'cheap');
  });
});

describe('toDistribution', () => {
  it('normalizes and rejects an empty distribution', () => {
    assert.deepEqual(toDistribution(2, 1, 1), { simple: 0.5, standard: 0.25, structural: 0.25 });
    assert.throws(() => toDistribution(0, 0, 0));
  });
});
