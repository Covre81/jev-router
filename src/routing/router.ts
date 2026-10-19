import type { Classification, ComplexityClassifier } from '../classifier/classifier.js';
import type { ComplexityDistribution } from '../domain/complexity.js';
import { selectTier, stickyTier, tierRoute, geminiEligible, type PolicyOptions, type Route, type Tier } from '../domain/policy.js';
import {
  conversationFingerprint,
  estimateInputTokens,
  isFreshConversation,
  latestHumanText,
  hasImageDocumentOrToolChoice,
  type MessagesBody,
} from './messages-body.js';
import { needsReadFirstHint } from './read-first-hint.js';
import type { QuotaLevel } from '../quota.js';
import type { TtlLruStore } from './session-store.js';

export type RouteReason =
  | 'passthrough:request-class'
  | 'passthrough:no-session-state'
  | 'passthrough:classifier-failed'
  | 'sticky'
  | 'classified'
  | 'escalated'
  | 'escalated:standard'
  | 'escalated:context'
  | 'failover:cheap-unavailable'
  | 'failover:standard-unavailable'
  | 'skipped:cheap-unhealthy'
  | 'pinned:compaction'
  | 'quota:pressure'
  | 'quota:reclassified'
  | 'read-first:standard'
  | 'failover:primary-rate-limited'
  | 'gemini:text-only'
  | 'gemini:text-only:pressure'
  | 'skipped:gemini-unhealthy'
  | 'skipped:gemini-busy'
  | 'failover:gemini-unavailable';

export interface RouteDecision {
  readonly route: Route;
  /** Which cheap model serves a cheap route; absent on decisions built outside the router. */
  readonly tier?: Tier;
  readonly fallbackTier?: Tier;
  readonly previousTier?: Tier;
  readonly modelTransition?: boolean;
  readonly reason: RouteReason;
  readonly conversationKey: string | undefined;
  readonly distribution?: Classification;
  readonly classifierMs?: number;
  readonly classifierError?: string;
}

/** Facts about the request that live outside the body (Claude Code headers). */
export interface RequestContext {
  readonly body: MessagesBody;
  readonly rawByteLength: number;
  readonly sessionId: string | undefined;
  readonly agentId: string | undefined;
  readonly requestClass: string | undefined;
  readonly contextCompacted: boolean;
}

export interface RouterOptions {
  readonly policy: PolicyOptions;
  /** x-claude-code-request-class values that always use the primary provider. */
  readonly primaryClasses: ReadonlySet<string>;
  /** Input-token budget of the trivial-tier model (its context window minus headroom). */
  readonly cheapContextTokens: number;
  /** Input-token budget of the standard-tier model; defaults to the trivial one. */
  readonly standardContextTokens?: number;
  readonly classifierTimeoutMs: number;
  readonly classifierMaxChars: number;
  /** Claude quota as last seen on an Anthropic response; absent = quota routing off. */
  readonly quota?: { level(): QuotaLevel };
  /** Bars used from `pressure` on; each only lowers the normal one. */
  readonly pressurePolicy?: { readonly minCheapProbability: number; readonly minStandardProbability: number };
  readonly geminiPolicy?: import('../domain/policy.js').GeminiPolicyOptions;
  readonly geminiFromPrimary?: boolean;
  /** route first open-ended project-state turns to the standard tier */
  readonly readFirstStandard?: boolean;
}

const TIER_RANK: Readonly<Record<Exclude<Tier, 'gemini'>, number>> = { trivial: 0, standard: 1, primary: 2 };

/**
 * Multi-provider router: decides, per conversation, between a cheap
 * OpenAI-compatible provider and the primary Anthropic quota. It only decides —
 * base URLs, headers and wire formats belong to `src/providers`.
 *
 * Rules:
 *   - a fresh (or just-compacted) conversation is classified by JEV;
 *   - tool-result continuations reuse the conversation's route (no JEV call);
 *   - a new human turn on the cheap route is re-classified and may escalate;
 *   - primary is terminal, except under critical quota, where a new human
 *     turn may leave it (and a compaction no longer pins it there);
 *   - anything the cheap model cannot hold (context) goes primary;
 *   - JEV failure fails toward primary — losing savings, never correctness.
 */
export class Router {
  private readonly lastRouteWasGemini = new Set<string>();
  private readonly lastServedTier = new Map<string, Tier>();

  constructor(
    private readonly classifier: ComplexityClassifier,
    private readonly sessions: TtlLruStore<Tier>,
    private readonly options: RouterOptions,
  ) {}

  async decide(ctx: RequestContext): Promise<RouteDecision> {
    if (ctx.requestClass && this.options.primaryClasses.has(ctx.requestClass)) {
      return { route: 'primary', reason: 'passthrough:request-class', conversationKey: undefined };
    }

    const key = ctx.sessionId
      ? `${ctx.sessionId}:${ctx.agentId ?? 'main'}`
      : `fp:${conversationFingerprint(ctx.body)}`;
    const fresh = isFreshConversation(ctx.body) || ctx.contextCompacted;
    const prevServed: Tier | undefined = this.lastServedTier.get(key) ?? (!fresh ? this.sessions.get(key) : undefined);
    const inputTokens = estimateInputTokens(ctx.rawByteLength);
    const budget: Readonly<Record<Tier, number>> = {
      trivial: this.options.cheapContextTokens,
      standard: this.options.standardContextTokens ?? this.options.cheapContextTokens,
      primary: Infinity,
      gemini: Infinity,
    };

    const stampTransition = (dec: RouteDecision): RouteDecision => {
      const current = dec.tier ?? (dec.route === 'primary' ? 'primary' : dec.route === 'gemini' ? 'gemini' : 'trivial');
      this.lastServedTier.set(key, current);
      if (prevServed !== undefined && prevServed !== current) {
        return { ...dec, previousTier: prevServed, modelTransition: true };
      }
      return dec;
    };

    const resolve = (tier: Tier, reason: RouteReason, extra: Partial<RouteDecision> = {}): RouteDecision => {
      const escalatedByContext = inputTokens > budget[tier];
      const final: Tier = escalatedByContext ? 'primary' : tier;
      this.sessions.set(key, final);
      if (final === 'gemini') this.lastRouteWasGemini.add(key);
      else this.lastRouteWasGemini.delete(key);
      return stampTransition({
        ...extra,
        route: tierRoute(final),
        tier: final,
        reason: escalatedByContext ? 'escalated:context' : reason,
        conversationKey: key,
      });
    };
    const primary = (reason: RouteReason, extra: Partial<RouteDecision> = {}): RouteDecision => {
      this.lastRouteWasGemini.delete(key);
      return stampTransition({
        ...extra,
        route: 'primary',
        tier: 'primary',
        reason,
        conversationKey: key,
      });
    };

    const level = this.options.quota?.level() ?? 'none';
    const pressure = this.options.pressurePolicy;
    const policy: PolicyOptions =
      level === 'none' || !pressure
        ? this.options.policy
        : {
            ...this.options.policy,
            minCheapProbability: Math.min(this.options.policy.minCheapProbability, pressure.minCheapProbability),
            minStandardProbability: Math.min(this.options.policy.minStandardProbability ?? 1, pressure.minStandardProbability),
          };

    // A conversation we hold no state for (proxy restart, TTL expiry) has been
    // running on the primary: keep it there rather than switching mid-flight.
    const stored: Tier = this.sessions.get(key) ?? 'primary';
    // Compaction used to count as a fresh start, dropping long Claude sessions on the 20B.
    if (ctx.contextCompacted && stored === 'primary' && level !== 'critical') return resolve('primary', 'pinned:compaction');
    const sticky: Tier | undefined = (fresh || stored === 'gemini') ? undefined : stored;

    const humanText = latestHumanText(ctx.body);
    if (humanText === undefined) {
      if (
        (stored === 'gemini' || this.lastRouteWasGemini.has(key)) &&
        this.options.geminiPolicy?.enabled &&
        !hasImageDocumentOrToolChoice(ctx.body)
      ) {
        this.lastRouteWasGemini.add(key);
        const fallbackTier: Tier = stored === 'gemini' ? 'primary' : stored;
        return stampTransition({
          route: 'gemini',
          tier: 'gemini',
          fallbackTier,
          reason: 'sticky',
          conversationKey: key,
        });
      }
      if (stored === 'gemini') return primary('sticky');
      return sticky ? resolve(sticky, 'sticky') : primary('passthrough:no-session-state');
    }

    // Quota nearly gone: a new human turn on Claude may leave it. Tool loops never do.
    const reclassify = level === 'critical' && stored === 'primary' && (sticky === 'primary' || ctx.contextCompacted);
    const classifyForGemini = this.options.geminiPolicy?.enabled && this.options.geminiFromPrimary && sticky === 'primary' && !reclassify;

    if (sticky === 'primary' && !reclassify && !classifyForGemini) return resolve('primary', 'sticky');

    const started = performance.now();
    let distribution: Classification;
    try {
      distribution = await this.classifier.classify(
        {
          text: humanText.slice(0, this.options.classifierMaxChars),
          turnCount: ctx.body.messages.length,
          toolCount: Array.isArray(ctx.body['tools']) ? ctx.body['tools'].length : 0,
          estimatedInputTokens: inputTokens,
        },
        AbortSignal.timeout(this.options.classifierTimeoutMs),
      );
    } catch (err) {
      // An unscored human turn may be structural: fail toward primary even mid-conversation.
      const classifierError = err instanceof Error ? err.message : String(err);
      if (classifyForGemini) {
        return resolve('primary', 'sticky', { classifierError });
      }
      return resolve('primary', 'passthrough:classifier-failed', { classifierError });
    }
    const classifierMs = Math.round(performance.now() - started);

    // The risk veto runs inside selectTier, before any bar: no quota level moves a risky turn.
    const proposed = selectTier(distribution, policy);
    const extra = { distribution, classifierMs };
    
    const needsReadFirst = this.options.readFirstStandard === true && needsReadFirstHint(ctx.body);

    let isGeminiEligible = false;
    if (this.options.geminiPolicy && !hasImageDocumentOrToolChoice(ctx.body)) {
      const effectiveProposedForGemini = classifyForGemini ? 'primary' : proposed;
      isGeminiEligible = geminiEligible(distribution, effectiveProposedForGemini, this.options.geminiPolicy, level);
    }
    
    if (isGeminiEligible) {
      let nextStored: Tier;
      if (fresh || stored === 'gemini') {
        nextStored = 'gemini';
      } else if (stored === 'primary') {
        nextStored = 'primary';
      } else {
        nextStored = sticky!;
      }
      this.sessions.set(key, nextStored);
      this.lastRouteWasGemini.add(key);

      const gp = this.options.geminiPolicy!;
      const barLowered = level !== 'none' && gp.pressureMinTextOnly < gp.minTextOnly;
      const isPressure = barLowered && (distribution.textOnly ?? 0) < gp.minTextOnly;
      return stampTransition({
        ...extra,
        route: 'gemini',
        tier: 'gemini',
        fallbackTier: nextStored === 'gemini' ? 'primary' : nextStored,
        reason: isPressure ? 'gemini:text-only:pressure' : 'gemini:text-only',
        conversationKey: key,
      });
    }

    if (classifyForGemini) {
      return resolve('primary', 'sticky', extra);
    }

    if (reclassify) return resolve(proposed, proposed === 'primary' ? 'sticky' : 'quota:reclassified', extra);
    if (!sticky) {
      if (needsReadFirst && proposed === 'trivial') {
        return resolve('standard', 'read-first:standard', extra);
      }
      let lowered = false;
      if (policy !== this.options.policy) {
         const normalProposed = selectTier(distribution, this.options.policy);
         if (normalProposed !== 'gemini' && proposed !== 'gemini') {
            lowered = TIER_RANK[proposed] < TIER_RANK[normalProposed];
         }
      }
      return resolve(proposed, lowered ? 'quota:pressure' : 'classified', extra);
    }

    const next = stickyTier(sticky, proposed);
    const reason: RouteReason = next === sticky ? 'sticky' : next === 'standard' ? 'escalated:standard' : 'escalated';
    return resolve(next, reason, extra);
  }

  /** The cheap provider could not serve this conversation: keep it on the primary from now on. */
  pinToPrimary(decision: RouteDecision): void {
    if (decision.conversationKey) {
      this.sessions.set(decision.conversationKey, 'primary');
      this.lastRouteWasGemini.delete(decision.conversationKey);
      this.lastServedTier.set(decision.conversationKey, 'primary');
    }
  }

  /** The standard tier was unavailable for a read-first turn: serve it on trivial. */
  useTrivial(decision: RouteDecision): void {
    if (decision.conversationKey) {
      this.sessions.set(decision.conversationKey, 'trivial');
      this.lastRouteWasGemini.delete(decision.conversationKey);
      this.lastServedTier.set(decision.conversationKey, 'trivial');
    }
  }

  /**
   * A Gemini turn fell back: clear its tool loop and record who served it, so the next turn is no fake
   * transition. A conversation stored as gemini moves to that tier too, or its tool steps would try Gemini again.
   */
  clearGeminiLoop(decision: RouteDecision, served: Tier): void {
    const key = decision.conversationKey;
    if (key) {
      this.lastRouteWasGemini.delete(key);
      this.lastServedTier.set(key, served);
      if (this.sessions.get(key) === 'gemini') this.sessions.set(key, served);
    }
  }
}

