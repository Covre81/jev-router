import type { ComplexityDistribution } from './complexity.js';

/**
 * Where a conversation runs.
 *   cheap   — an OpenAI-compatible provider (Groq, OpenRouter, …), billed per token
 *   primary — Anthropic, the quota reserved for heavy lifting
 */
export type Route = 'cheap' | 'gemini' | 'cursor' | 'primary';

/**
 * Which model serves a conversation. `trivial` and `standard` are both the
 * cheap route (same OpenAI-compatible provider, different model); `primary`
 * is Anthropic. Ordered: a conversation's tier only moves up.
 */
export type Tier = 'trivial' | 'standard' | 'gemini' | 'cursor' | 'primary';

export interface PolicyOptions {
  /** Minimum P(simple) for the trivial tier (the small cheap model). */
  readonly minCheapProbability: number;
  /**
   * Legacy, used only while the standard tier is off: `cheap` sends level-2
   * mass to the trivial tier. Kept so an old .env keeps its behavior.
   */
  readonly standardRoute: Route;
  /** A standard-tier model is configured (CHEAP_MODEL_STANDARD is not `off`). */
  readonly standardEnabled?: boolean;
  /** Minimum P(simple) + P(standard) for the standard tier. */
  readonly minStandardProbability?: number;
}

/**
 * Deterministic System-One decision.
 *
 * Not argmax: a distribution like {simple: .45, standard: .30, structural: .25}
 * has "simple" as its mode, yet a 55% chance the cheap model is out of its
 * depth. We send work to the cheap provider only when the probability mass of
 * cheap-eligible levels clears the bar; everything else stays on the primary.
 */
export interface RiskScoresSummary {
  readonly security_sensitive: number;
  readonly destructive_or_production: number;
  readonly requires_inspection: number;
}

export function selectTier(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary },
  options: PolicyOptions,
): Tier {
  if (hardRiskVeto(d)) return 'primary';

  const inspectionVeto = (d.risk ?? 0) >= MAX_CHEAP_RISK;
  if (!inspectionVeto && d.simple >= options.minCheapProbability) return 'trivial';

  const lowMass = d.simple + d.standard;
  if (options.standardEnabled) {
    return lowMass >= (options.minStandardProbability ?? 1) ? 'standard' : 'primary';
  }
  if (inspectionVeto) return 'primary';
  return options.standardRoute === 'cheap' && lowMass >= options.minCheapProbability ? 'trivial' : 'primary';
}

/** The route of `selectTier`: both cheap tiers are the cheap route. */
export function selectRoute(d: ComplexityDistribution & { readonly risk?: number }, options: PolicyOptions): Route {
  return tierRoute(selectTier(d, options));
}

export const tierRoute = (tier: Tier): Route =>
  tier === 'primary' ? 'primary' : tier === 'gemini' ? 'gemini' : tier === 'cursor' ? 'cursor' : 'cheap';

const TIER_RANK: Readonly<Record<Exclude<Tier, 'gemini' | 'cursor'>, number>> = { trivial: 0, standard: 1, primary: 2 };

/**
 * Escalate-only, across tiers: trivial → standard → primary. Bouncing down
 * would throw away the cache of the model the conversation moved up to.
 */
export function stickyTier(current: Tier, proposed: Tier): Tier {
  if (current === 'gemini' && proposed === 'gemini') return 'primary';
  if (current === 'gemini' || current === 'cursor') return proposed === 'gemini' || proposed === 'cursor' ? current : proposed;
  if (proposed === 'gemini' || proposed === 'cursor') return current;
  return TIER_RANK[proposed] > TIER_RANK[current] ? proposed : current;
}

/**
 * A short request can be risky ("store the Stripe key in the code"): JEV calls
 * it simple, and it is, for the wrong reason. On scripts/eval-jev.ts the risk
 * Nouls split cleanly (simple prompts <= 0.15, risky >= 0.92), so 0.5 is not
 * a tuned knob.
 */
// ponytail: fixed bar, make it a PolicyOptions field if the eval ever puts cases near it
const MAX_CHEAP_RISK = 0.5;

/** Same hard-risk rule selectTier uses: security/destructive scores veto every non-primary route. */
export function hardRiskVeto(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary },
): boolean {
  const hardRisk = d.riskScores
    ? Math.max(d.riskScores.security_sensitive, d.riskScores.destructive_or_production)
    : (d.risk ?? 0);
  return hardRisk >= MAX_CHEAP_RISK;
}

/**
 * Escalate-only: once a conversation reaches the primary provider it stays
 * there. Bouncing back to the cheap model would throw away the Anthropic
 * prompt cache, which the next escalation would have to rebuild at full price.
 */
export function stickyRoute(current: Route, proposed: Route): Route {
  return current === 'primary' || proposed === 'primary' ? 'primary' : 'cheap';
}

export interface GeminiPolicyOptions {
  readonly enabled: boolean;
  readonly minTextOnly: number;
  readonly pressureMinTextOnly: number;
}

export function geminiEligible(
  d: ComplexityDistribution & { readonly risk?: number; readonly riskScores?: RiskScoresSummary; readonly textOnly?: number },
  proposedTier: Tier,
  opts: GeminiPolicyOptions,
  quotaLevel: 'none' | 'pressure' | 'critical'
): boolean {
  if (!opts.enabled) return false;
  if (proposedTier !== 'primary') return false;
  const hardRisk = d.riskScores
    ? Math.max(d.riskScores.security_sensitive, d.riskScores.destructive_or_production)
    : (d.risk ?? 0);
  if (hardRisk >= MAX_CHEAP_RISK) return false;

  const bar = quotaLevel !== 'none' ? Math.min(opts.minTextOnly, opts.pressureMinTextOnly) : opts.minTextOnly;
  if (bar <= 0) return true;
  if (d.textOnly === undefined) return false;
  return d.textOnly >= bar;
}

