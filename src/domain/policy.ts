import type { ComplexityDistribution } from './complexity.js';

export type Route = 'free' | 'gemini' | 'argon' | 'cursor' | 'primary' | 'cheap';

export type Tier = 'free' | 'gemini' | 'argon' | 'cursor' | 'primary' | 'trivial' | 'standard';

export interface PolicyOptions {
  /** Minimum P(simple) for free tier. */
  readonly minCheapProbability?: number;
  /** Minimum P(simple) + P(standard) for standard/free tier. */
  readonly minStandardProbability?: number;
  readonly argonEnabled?: boolean;
  readonly standardRoute?: Route;
  readonly standardEnabled?: boolean;
}

export interface RiskScoresSummary {
  readonly security_sensitive: number;
  readonly destructive_or_production: number;
  readonly requires_inspection: number;
}

export const MAX_CHEAP_RISK = 0.5;

export const ARGON_STRUCTURAL_HIGH = 0.45;
export const ARGON_STRUCTURAL_MED = 0.3;

export type ArgonProfile = 'low' | 'medium' | 'high';

/**
 * Hard risk veto: security or destructive signals veto every non-primary route.
 */
export function hardRiskVeto(
  d?: (ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary }) | undefined,
): boolean {
  if (!d) return false;
  const hardRisk = d.riskScores
    ? Math.max(d.riskScores.security_sensitive, d.riskScores.destructive_or_production)
    : (d.risk ?? 0);
  return hardRisk >= MAX_CHEAP_RISK;
}

/**
 * Select profile inside the Argon tier:
 * - hardRiskVeto(d) -> null (straight to Claude)
 * - structural >= 0.45 || requires_inspection >= 0.5 -> 'high'
 * - structural >= 0.3 || standard >= 0.5 -> 'medium'
 * - else 'low'
 */
export function selectArgonProfile(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary },
): ArgonProfile | null {
  if (hardRiskVeto(d)) return null;

  const requiresInspection = d.riskScores?.requires_inspection ?? 0;
  if (d.structural >= ARGON_STRUCTURAL_HIGH || requiresInspection >= 0.5) {
    return 'high';
  }
  if (d.structural >= ARGON_STRUCTURAL_MED || d.standard >= 0.5) {
    return 'medium';
  }
  return 'low';
}

export function bumpArgonProfile(profile: ArgonProfile): ArgonProfile | null {
  if (profile === 'low') return 'medium';
  if (profile === 'medium') return 'high';
  return null;
}

export const CURSOR_CODING_MAX_TEXT_ONLY = 0.5;
export const CURSOR_CODING_MIN_MASS = 0.5;

export function isCodingTurn(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary; readonly textOnly?: number },
  options?: { readonly cursorCodingMaxTextOnly?: number | undefined; readonly cursorCodingMinMass?: number | undefined },
): boolean {
  if (hardRiskVeto(d)) return false;
  const maxTextOnly = options?.cursorCodingMaxTextOnly ?? CURSOR_CODING_MAX_TEXT_ONLY;
  const minMass = options?.cursorCodingMinMass ?? CURSOR_CODING_MIN_MASS;
  const textOnly = d.textOnly ?? 1;
  const mass = d.standard + d.structural;
  return textOnly < maxTextOnly && mass >= minMass;
}

/**
 * Deterministic System-One tier selection.
 * Free (was trivial/standard) -> Argon -> Primary.
 */
export function selectTier(
  d?: (ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary }) | undefined,
  options: PolicyOptions = {},
): 'free' | 'argon' | 'primary' {
  if (!d) return 'primary';
  if (hardRiskVeto(d)) return 'primary';

  const inspectionVeto = (d.riskScores?.requires_inspection ?? (d.risk ?? 0)) >= MAX_CHEAP_RISK;
  const minFree = options.minCheapProbability ?? 0.9;
  if (!inspectionVeto && d.simple >= minFree) return 'free';

  const lowMass = d.simple + d.standard;
  const minStandard = options.minStandardProbability ?? 0.75;
  if (!inspectionVeto && lowMass >= minStandard && options.standardEnabled !== false) {
    return 'free';
  }

  if (options.argonEnabled !== false) {
    return 'argon';
  }

  return 'primary';
}

export function selectRoute(d: ComplexityDistribution & { readonly risk?: number }, options: PolicyOptions = {}): Route {
  return tierRoute(selectTier(d, options));
}

export const tierRoute = (tier: Tier): Route => {
  if (tier === 'primary') return 'primary';
  if (tier === 'cursor') return 'cursor';
  if (tier === 'argon') return 'argon';
  if (tier === 'gemini') return 'free';
  return 'free';
};

const TIER_RANK: Readonly<Record<Tier, number>> = {
  trivial: 0,
  standard: 0,
  free: 0,
  gemini: 0,
  argon: 1,
  cursor: 1,
  primary: 2,
};

export function stickyTier(current: Tier, proposed: Tier): Tier {
  if (current === proposed) return current;
  return TIER_RANK[proposed] > TIER_RANK[current] ? proposed : current;
}

export type Effort = 'low' | 'medium' | 'high' | 'xhigh' | 'max';
export interface PrimaryProfile {
  readonly model: string;
  readonly effort: Effort;
}

export type EscalationReason =
  | 'gemini-failed'
  | 'escalated-from-gemini'
  | 'gemini-busy'
  | 'gemini-unhealthy'
  | 'context';

export const BUMP_LADDER: readonly PrimaryProfile[] = [
  { model: 'claude-sonnet-5-5', effort: 'low' },
  { model: 'claude-sonnet-5-5', effort: 'medium' },
  { model: 'claude-sonnet-5-5', effort: 'high' },
  { model: 'claude-sonnet-5-5', effort: 'xhigh' },
  { model: 'claude-opus-5-5', effort: 'medium' },
  { model: 'claude-opus-5-5', effort: 'high' },
  { model: 'claude-opus-5-5', effort: 'xhigh' },
];

function bumpLadder(profile: PrimaryProfile): PrimaryProfile {
  const idx = BUMP_LADDER.findIndex((p) => p.model === profile.model && p.effort === profile.effort);
  if (idx === -1) return profile;
  return BUMP_LADDER[Math.min(idx + 1, BUMP_LADDER.length - 1)] ?? profile;
}

export function primaryProfile(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary },
  escalation?: EscalationReason,
): PrimaryProfile {
  let base: PrimaryProfile;
  if (hardRiskVeto(d) && d.structural >= 0.6) {
    base = { model: 'claude-opus-5-5', effort: 'xhigh' };
  } else if (hardRiskVeto(d) || d.structural >= 0.6) {
    base = { model: 'claude-opus-5-5', effort: 'high' };
  } else if (d.structural >= 0.45 || (d.riskScores?.requires_inspection ?? 0) >= 0.5) {
    base = { model: 'claude-opus-5-5', effort: 'medium' };
  } else if (d.structural >= 0.3) {
    base = { model: 'claude-sonnet-5-5', effort: 'high' };
  } else if (d.simple >= 0.6) {
    base = { model: 'claude-sonnet-5-5', effort: 'low' };
  } else {
    base = { model: 'claude-sonnet-5-5', effort: 'medium' };
  }

  if (escalation === 'gemini-failed' || escalation === 'escalated-from-gemini') {
    return bumpLadder(base);
  }
  if (escalation === 'context') {
    if (d.simple >= 0.6) {
      return { model: 'claude-opus-5-5', effort: 'low' };
    }
    return base;
  }
  return base;
}

export function stickyRoute(current: Route, proposed: Route): Route {
  return current === 'primary' || proposed === 'primary' ? 'primary' : proposed;
}

export interface GeminiPolicyOptions {
  readonly enabled: boolean;
  readonly minTextOnly: number;
  readonly pressureMinTextOnly: number;
  readonly replacesCheap?: boolean;
}

export function geminiEligible(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary; readonly textOnly?: number },
  proposedTier: Tier,
  opts: GeminiPolicyOptions,
  quotaLevel: 'none' | 'pressure' | 'critical'
): boolean {
  if (!opts.enabled) return false;
  if (opts.replacesCheap && (proposedTier === 'free' || proposedTier === 'gemini' || (proposedTier as string) === 'trivial' || (proposedTier as string) === 'standard')) return true;
  if (proposedTier !== 'primary') return false;
  if (hardRiskVeto(d)) return false;

  const bar = quotaLevel !== 'none' ? Math.min(opts.minTextOnly, opts.pressureMinTextOnly) : opts.minTextOnly;
  if (bar <= 0) return true;
  if (d.textOnly === undefined) return false;
  return d.textOnly >= bar;
}
