import type { Classification, ComplexityClassifier } from '../classifier/classifier.js';
import type { ComplexityDistribution } from '../domain/complexity.js';
import { selectTier, stickyTier, tierRoute, geminiEligible, hardRiskVeto, type PolicyOptions, type Route, type Tier } from '../domain/policy.js';
import {
  conversationFingerprint,
  estimateInputTokens,
  isFreshConversation,
  latestHumanText,
  hasImageDocumentOrToolChoice,
  type MessagesBody,
} from './messages-body.js';
import { needsReadFirstHint } from './read-first-hint.js';
import type { QuotaLevel, QuotaSnapshot } from '../quota.js';
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
  | 'failover:gemini-unavailable'
  | 'failover:cursor-unavailable'
  | 'skipped:cursor-unhealthy'
  | 'skipped:cursor-busy';

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
  /** Set when this decision serves a turn of a cascade fallback loop (Cursor/Gemini after a Claude 429). */
  readonly fallbackLoop?: 'cursor' | 'gemini';
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
  readonly quota?: { level(): QuotaLevel; current?(): QuotaSnapshot | undefined };
  /** Bars used from `pressure` on; each only lowers the normal one. */
  readonly pressurePolicy?: { readonly minCheapProbability: number; readonly minStandardProbability: number };
  readonly geminiPolicy?: import('../domain/policy.js').GeminiPolicyOptions;
  readonly geminiFromPrimary?: boolean;
  /** Session TTL: fallback-loop marks older than this are treated as absent. Defaults to the serve.ts session TTL (6h). */
  readonly sessionTtlMs?: number;
  /** route first open-ended project-state turns to the standard tier */
  readonly readFirstStandard?: boolean;
}

const TIER_RANK: Readonly<Record<Exclude<Tier, 'gemini' | 'cursor'>, number>> = { trivial: 0, standard: 1, primary: 2 };

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
  /** Cascade fallback loop: conversation key -> CLI tier + mark time. Entries older than the session TTL are pruned lazily on read/mark (no timers). */
  private readonly fallbackLoop = new Map<string, { readonly tier: 'cursor' | 'gemini'; markedAt: number }>();

  private fallbackLoopTtlMs(): number {
    return this.options.sessionTtlMs ?? 6 * 60 * 60 * 1000;
  }

  private pruneFallbackLoop(now = Date.now()): void {
    const ttl = this.fallbackLoopTtlMs();
    for (const [k, e] of this.fallbackLoop) {
      if (now - e.markedAt > ttl) this.fallbackLoop.delete(k);
    }
  }

  private fallbackLoopTier(key: string): 'cursor' | 'gemini' | undefined {
    const entry = this.fallbackLoop.get(key);
    if (!entry) return undefined;
    if (Date.now() - entry.markedAt > this.fallbackLoopTtlMs()) {
      this.fallbackLoop.delete(key);
      return undefined;
    }
    return entry.tier;
  }

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
      cursor: Infinity,
    };

    const stampTransition = (dec: RouteDecision): RouteDecision => {
      const current = dec.tier ?? (dec.route === 'primary' ? 'primary' : dec.route === 'gemini' ? 'gemini' : dec.route === 'cursor' ? 'cursor' : 'trivial');
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
    // A quota probe (max_tokens <= 1) is never rerouted: it must not be served by a loop CLI tier.
    const isProbe = typeof ctx.body.max_tokens === 'number' && ctx.body.max_tokens <= 1;
    const loopTier = this.fallbackLoopTier(key);
    const loopActive =
      loopTier && !hasImageDocumentOrToolChoice(ctx.body) && !isProbe ? loopTier : undefined;
    if (loopActive && humanText === undefined) {
      // Tool loop of a cascade-served turn: stay on that tier, never hit Claude first.
      return stampTransition({
        route: tierRoute(loopActive),
        tier: loopActive,
        reason: 'sticky',
        conversationKey: key,
        fallbackLoop: loopActive,
      });
    }
    // A new human turn in a loop is NOT decided here: it falls through to the
    // single classification below, so the risk veto runs before the loop can
    // hold it (see the loopActive handling after classification).
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

    if (sticky === 'primary' && !reclassify && !classifyForGemini && !loopActive) return resolve('primary', 'sticky');

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

    if (loopActive) {
      // Human turn held by a cascade loop, now classified (single call above).
      if (hardRiskVeto(distribution)) {
        // Risky turn leaves the loop for Claude; the loop must not serve it
        // via a CLI tier or advance it to cheap from a CLI fallback.
        this.fallbackLoop.delete(key);
        return resolve('primary', 'classified', extra);
      }
      // Hold the loop only while the gate has not cleared. Clear it only when
      // a snapshot exists AND (its window is not 5h OR its reset has passed);
      // no snapshot => stay on the loop tier.
      const snap = this.options.quota?.current?.();
      const gateClears =
        snap !== undefined &&
        (!snap.window.includes('5h') || (snap.resetAt !== undefined && snap.resetAt.getTime() <= Date.now()));
      if (!gateClears) {
        return stampTransition({
          ...extra,
          route: tierRoute(loopActive),
          tier: loopActive,
          reason: 'sticky',
          conversationKey: key,
          fallbackLoop: loopActive,
        });
      }
      // Window reset (or not a 5h window): Claude is retried; a fresh 429 re-enters the cascade.
      this.fallbackLoop.delete(key);
    }

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
        // gemini/cursor are transient tiers, not escalation ranks (mirrors policy.ts TIER_RANK).
        const ranked = (t: Tier): t is Exclude<Tier, 'gemini' | 'cursor'> => t !== 'gemini' && t !== 'cursor';
        if (ranked(proposed) && ranked(normalProposed)) {
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
      this.fallbackLoop.delete(decision.conversationKey);
      this.lastServedTier.set(decision.conversationKey, 'primary');
    }
  }

  /** A cascade tier (Cursor/Gemini) served a turn after Claude rate-limited: its tool loop sticks to it. */
  markFallbackLoop(decision: RouteDecision, served: 'cursor' | 'gemini'): void {
    const key = decision.conversationKey;
    if (!key) return;
    this.pruneFallbackLoop();
    this.fallbackLoop.set(key, { tier: served, markedAt: Date.now() });
    this.lastServedTier.set(key, served);
    if (served === 'gemini') this.lastRouteWasGemini.add(key);
    else this.lastRouteWasGemini.delete(key);
  }

  /** The fallback loop ended (another tier served): record who served, drop the loop mark. */
  clearFallbackLoop(decision: RouteDecision, served: Tier): void {
    const key = decision.conversationKey;
    if (key) {
      this.fallbackLoop.delete(key);
      this.lastRouteWasGemini.delete(key);
      this.lastServedTier.set(key, served);
      if (this.sessions.get(key) === 'gemini') this.sessions.set(key, served);
    }
  }

  /** The standard tier was unavailable for a read-first turn: serve it on trivial. */
  useTrivial(decision: RouteDecision): void {
    if (decision.conversationKey) {
      this.sessions.set(decision.conversationKey, 'trivial');
      this.lastRouteWasGemini.delete(decision.conversationKey);
      this.fallbackLoop.delete(decision.conversationKey);
      this.lastServedTier.set(decision.conversationKey, 'trivial');
    }
  }

  /**
   * A Gemini turn fell back: clear its tool loop and record who served it, so the next turn is no fake
   * transition. A conversation stored as gemini moves to that tier too, or its tool steps would try Gemini again.
   */
  clearGeminiLoop(decision: RouteDecision, served: Tier): void {
    this.clearFallbackLoop(decision, served);
  }
}

