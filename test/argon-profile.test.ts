import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import {
  ARGON_STRUCTURAL_HIGH,
  ARGON_STRUCTURAL_MED,
  bumpArgonProfile,
  hardRiskVeto,
  primaryProfile,
  selectArgonProfile,
  selectRoute,
  selectTier,
  type ArgonProfile,
} from '../src/domain/policy.js';

describe('Argon profile selection and bump ladder', () => {
  describe('selectArgonProfile threshold table', () => {
    it('returns null on hard risk veto (security_sensitive >= 0.5)', () => {
      const d = {
        simple: 0.2,
        standard: 0.3,
        structural: 0.5,
        riskScores: {
          security_sensitive: 0.6,
          destructive_or_production: 0.1,
          requires_inspection: 0.2,
        },
      };
      assert.equal(hardRiskVeto(d), true);
      assert.equal(selectArgonProfile(d), null);
    });

    it('returns null on hard risk veto (destructive_or_production >= 0.5)', () => {
      const d = {
        simple: 0.2,
        standard: 0.3,
        structural: 0.5,
        riskScores: {
          security_sensitive: 0.1,
          destructive_or_production: 0.7,
          requires_inspection: 0.2,
        },
      };
      assert.equal(hardRiskVeto(d), true);
      assert.equal(selectArgonProfile(d), null);
    });

    it('returns null on legacy risk >= 0.5 when riskScores is missing', () => {
      const d = {
        simple: 0.2,
        standard: 0.3,
        structural: 0.5,
        risk: 0.8,
      };
      assert.equal(hardRiskVeto(d), true);
      assert.equal(selectArgonProfile(d), null);
    });

    it('returns high when structural >= ARGON_STRUCTURAL_HIGH (0.45)', () => {
      const d = {
        simple: 0.2,
        standard: 0.35,
        structural: ARGON_STRUCTURAL_HIGH,
      };
      assert.equal(selectArgonProfile(d), 'high');

      const dAbove = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.6,
      };
      assert.equal(selectArgonProfile(dAbove), 'high');
    });

    it('returns high when requires_inspection >= 0.5 (even with low structural)', () => {
      const d = {
        simple: 0.7,
        standard: 0.2,
        structural: 0.1,
        riskScores: {
          security_sensitive: 0.1,
          destructive_or_production: 0.1,
          requires_inspection: 0.5,
        },
      };
      assert.equal(hardRiskVeto(d), false);
      assert.equal(selectArgonProfile(d), 'high');
    });

    it('returns medium when structural >= ARGON_STRUCTURAL_MED (0.3)', () => {
      const d = {
        simple: 0.4,
        standard: 0.3,
        structural: ARGON_STRUCTURAL_MED,
      };
      assert.equal(selectArgonProfile(d), 'medium');

      const dAbove = {
        simple: 0.3,
        standard: 0.35,
        structural: 0.35,
      };
      assert.equal(selectArgonProfile(dAbove), 'medium');
    });

    it('returns medium when standard >= 0.5 (even with low structural)', () => {
      const d = {
        simple: 0.3,
        standard: 0.5,
        structural: 0.2,
      };
      assert.equal(selectArgonProfile(d), 'medium');

      const dAbove = {
        simple: 0.2,
        standard: 0.6,
        structural: 0.2,
      };
      assert.equal(selectArgonProfile(dAbove), 'medium');
    });

    it('returns low when neither high nor medium thresholds are met', () => {
      const d = {
        simple: 0.6,
        standard: 0.25,
        structural: 0.15,
      };
      assert.equal(selectArgonProfile(d), 'low');
    });
  });

  describe('bumpArgonProfile step function', () => {
    it('bumps low to medium', () => {
      assert.equal(bumpArgonProfile('low'), 'medium');
    });

    it('bumps medium to high', () => {
      assert.equal(bumpArgonProfile('medium'), 'high');
    });

    it('returns null after high (exhaustion)', () => {
      assert.equal(bumpArgonProfile('high'), null);
    });
  });

  describe('bump ladder traversal and Claude fallback profile', () => {
    it('simulates full low -> medium -> high -> Claude escalation ladder', () => {
      const d = { simple: 0.6, standard: 0.25, structural: 0.15 };
      let profile: ArgonProfile | null = selectArgonProfile(d);
      assert.equal(profile, 'low');

      profile = bumpArgonProfile(profile);
      assert.equal(profile, 'medium');

      profile = bumpArgonProfile(profile);
      assert.equal(profile, 'high');

      profile = bumpArgonProfile(profile);
      assert.equal(profile, null);

      // On exhaustion, falls back to Claude using primaryProfile(d, 'gemini-failed')
      const claudeProfile = primaryProfile(d, 'gemini-failed');
      assert.deepEqual(claudeProfile, {
        model: 'claude-sonnet-5-5',
        effort: 'medium', // base sonnet: low bumped to medium on gemini-failed
      });
    });

    it('routes hard risk veto straight to Claude without Argon', () => {
      const d = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.6,
        riskScores: {
          security_sensitive: 0.8,
          destructive_or_production: 0.1,
          requires_inspection: 0.1,
        },
      };

      assert.equal(selectArgonProfile(d), null);
      assert.equal(selectTier(d), 'primary');
      assert.equal(selectRoute(d), 'primary');

      const claudeProfile = primaryProfile(d);
      assert.deepEqual(claudeProfile, {
        model: 'claude-opus-5-5',
        effort: 'xhigh',
      });
    });
  });
});
