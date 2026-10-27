import type { Classification, ComplexityClassifier } from '../classifier/classifier.js';
import type { ComplexityDistribution } from '../domain/complexity.js';
import {
  selectTier,
  stickyTier,
  tierRoute,
  geminiEligible,
  hardRiskVeto,
  isCodingTurn,
  selectArgonProfile,
  primaryProfile,
  raiseToFloor,
  type EscalationReason,
  type PolicyOptions,
  type PrimaryProfile,
  type Route,
  type Tier,
  type ArgonProfile,
} from '../domain/policy.js';
import {
  conversationFingerprint,
  estimateInputTokens,
  isFreshConversation,
  lastHumanText,
  latestHumanText,
  hasImageDocumentOrToolChoice,
  type MessagesBody,
} from './messages-body.js';
import type { QuotaLevel, QuotaSnapshot } from '../quota.js';
import type { TtlLruStore } from './session-store.js';

export type RouteReason =
  | 'passthrough:request-class'
  | 'passthrough:no-session-state'
  | 'passthrough:classifier-failed'
  | 'sticky'
  | 'classified'
  | 'escalated'
  | 'pinned:compaction'
  | 'profile:refused'
  | 'quota:pressure'
  | 'quota:reclassified'
  | 'escalated:context'
  | 'failover:primary-rate-limited'
  | 'gemini:text-only'
  | 'gemini:text-only:pressure'
  | 'skipped:gemini-unhealthy'
  | 'skipped:gemini-busy'
  | 'failover:gemini-unavailable'
  | 'failover:cursor-unavailable'
  | 'skipped:cursor-unhealthy'
  | 'skipped:cursor-busy'
  | 'free:flash'
  | 'argon:low'
  | 'argon:medium'
  | 'argon:high'
  | 'cursor:coding'
  | 'escalated:flash-failed'
  | 'escalated:argon-failed'
  | 'escalated:cursor-failed'
  | 'hop-timeout:flash'
  | 'hop-timeout:argon'
  | 'hop-timeout:cursor'
  | 'hop-timeout:claude'
  | `quota-substitute:${string}`;

export interface RouteDecision {
  readonly route: Route;
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
  readonly fallbackLoop?: 'cursor' | 'gemini' | 'argon' | 'free' | 'trivial';
  /** Claude model + effort for this main-loop turn; absent = forward the client's own. */
  readonly primaryProfile?: PrimaryProfile;
  /** JEV score behind a profile picked outside routing (compaction): telemetry only, the risk veto re-classifies. */
  readonly profileScore?: Classification;
  readonly escalationReason?: EscalationReason;
  readonly argonProfile?: ArgonProfile;
  readonly substitutedFrom?: string;
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
  readonly classifierTimeoutMs: number;
  readonly classifierMaxChars: number;
  readonly cheapContextTokens?: number;
  readonly standardContextTokens?: number;
  readonly cheapRoute?: boolean;
  /** Claude quota as last seen on an Anthropic response; absent = quota routing off. */
  readonly quota?: { level(): QuotaLevel; current?(): QuotaSnapshot | undefined };
  /** Bars used from `pressure` on; each only lowers the normal one. */
  readonly pressurePolicy?: { readonly minCheapProbability: number; readonly minStandardProbability: number };
  readonly geminiPolicy?: import('../domain/policy.js').GeminiPolicyOptions;
  readonly geminiFromPrimary?: boolean;
  /** Session TTL: fallback-loop marks older than this are treated as absent. Defaults to the serve.ts session TTL (6h). */
  readonly sessionTtlMs?: number;
  /** Let JEV pick Claude model + effort per session (PRIMARY_PROFILES). */
  readonly primaryProfiles?: boolean;
  readonly argonPolicy?: {
    readonly enabled: boolean;
    readonly modelLow: string;
    readonly modelMedium: string;
    readonly modelHigh: string;
  };
  readonly cursorCoding?: boolean;
  readonly cursorCodingMaxTextOnly?: number;
  readonly cursorCodingMinMass?: number;
  readonly cursorEnabled?: boolean;
}

export class Router {
  private readonly lastRouteWasGemini = new Set<string>();
  private readonly lastServedTier = new Map<string, Tier>();
  /** null = Anthropic refused this session's profile: the client default serves until a fresh start. */
  private readonly profiles = new Map<string, PrimaryProfile | null>();
  /** Cascade fallback loop: conversation key -> CLI tier + mark time. Entries older than the session TTL are pruned lazily on read/mark (no timers). */
  private readonly fallbackLoop = new Map<string, { readonly tier: 'cursor' | 'gemini' | 'argon' | 'free'; markedAt: number }>();

  private fallbackLoopTtlMs(): number {
    return this.options.sessionTtlMs ?? 6 * 60 * 60 * 1000;
  }

  private pruneFallbackLoop(now = Date.now()): void {
    const ttl = this.fallbackLoopTtlMs();
    for (const [k, e] of this.fallbackLoop) {
      if (now - e.markedAt > ttl) this.fallbackLoop.delete(k);
    }
  }

  private fallbackLoopTier(key: string): 'cursor' | 'gemini' | 'argon' | 'free' | undefined {
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
    const dec = await this.decideTier(ctx);
    if (this.options.primaryProfiles && ctx.requestClass === 'compaction' && ctx.sessionId) {
      const stored = this.profiles.get(`${ctx.sessionId}:${ctx.agentId ?? 'main'}`);
      return stored ? { ...dec, primaryProfile: stored } : dec;
    }
    const key = dec.conversationKey;
    if (!this.options.primaryProfiles || ctx.requestClass !== 'main' || !key) return dec;
    if (isFreshConversation(ctx.body) || (ctx.contextCompacted && this.profiles.get(key) !== null)) this.profiles.delete(key);
    let profile = this.profiles.get(key);
    let profileScore: Classification | undefined;
    const opening =
      isFreshConversation(ctx.body) ||
      ctx.contextCompacted ||
      dec.previousTier === 'free' ||
      dec.previousTier === 'gemini' ||
      dec.previousTier === 'argon' ||
      dec.previousTier === 'cursor';
    let escalation: EscalationReason | undefined;
    if (profile === undefined && dec.route === 'primary' && opening) {
      if (!dec.distribution) profileScore = await this.classifyForProfile(ctx.body, ctx.rawByteLength);
      const distribution = dec.distribution ?? profileScore;
      if (distribution) {
        escalation =
          dec.reason === 'escalated:context'
            ? 'context'
            : dec.previousTier === 'gemini'
              ? 'escalated-from-gemini'
              : undefined;
        profile = primaryProfile(distribution, escalation);
        this.profiles.set(key, profile);
      }
    }
    return profile
      ? {
          ...dec,
          ...(profileScore && { profileScore }),
          primaryProfile: profile,
          ...(escalation && { escalationReason: escalation }),
        }
      : dec;
  }

  async profileForFallback(
    dec: RouteDecision,
    ctxOrDistribution: RequestContext | MessagesBody | Classification | undefined,
    reason: EscalationReason,
    floor?: PrimaryProfile,
  ): Promise<RouteDecision> {
    if (!this.options.primaryProfiles) return dec;
    const key = dec.conversationKey;
    if (!key) return dec;

    if (
      ctxOrDistribution &&
      'requestClass' in ctxOrDistribution &&
      ctxOrDistribution.requestClass &&
      ctxOrDistribution.requestClass !== 'main'
    ) {
      return dec;
    }

    if (this.profiles.get(key) === null) return dec;

    const stored = dec.primaryProfile ?? this.profiles.get(key);
    if (stored) {
      const finalProfile = floor ? raiseToFloor(stored, floor) : stored;
      return { ...dec, primaryProfile: finalProfile, escalationReason: reason };
    }

    let distribution: Classification | undefined = dec.distribution;
    let profileScore: Classification | undefined;

    if (!distribution && ctxOrDistribution) {
      if ('simple' in ctxOrDistribution && typeof ctxOrDistribution.simple === 'number') {
        distribution = ctxOrDistribution as Classification;
      } else if ('body' in ctxOrDistribution && (ctxOrDistribution as RequestContext).body) {
        const rc = ctxOrDistribution as RequestContext;
        distribution = await this.classifyForProfile(rc.body, rc.rawByteLength);
        profileScore = distribution;
      } else if ('messages' in ctxOrDistribution && Array.isArray((ctxOrDistribution as MessagesBody).messages)) {
        distribution = await this.classifyForProfile(ctxOrDistribution as MessagesBody);
        profileScore = distribution;
      }
    }

    if (!distribution) {
      if (floor) {
        return { ...dec, primaryProfile: floor, escalationReason: reason };
      }
      return dec;
    }

    let profile = primaryProfile(distribution, reason);
    if (floor) {
      profile = raiseToFloor(profile, floor);
    }
    if (reason !== 'gemini-busy') {
      this.profiles.set(key, profile);
    }

    return {
      ...dec,
      ...(profileScore && { profileScore }),
      primaryProfile: profile,
      escalationReason: reason,
    };
  }

  private async classifyForProfile(body: MessagesBody, rawByteLength = 0): Promise<Classification | undefined> {
    const text = latestHumanText(body);
    if (text === undefined) return undefined;
    try {
      return await this.classifier.classify(
        {
          text: text.slice(0, this.options.classifierMaxChars),
          turnCount: body.messages.length,
          toolCount: Array.isArray(body['tools']) ? body['tools'].length : 0,
          estimatedInputTokens: estimateInputTokens(rawByteLength),
        },
        AbortSignal.timeout(this.options.classifierTimeoutMs),
      );
    } catch {
      return undefined;
    }
  }

  private async decideTier(ctx: RequestContext): Promise<RouteDecision> {
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
      trivial: this.options.cheapContextTokens ?? Infinity,
      standard: this.options.standardContextTokens ?? this.options.cheapContextTokens ?? Infinity,
      free: this.options.cheapContextTokens ?? Infinity,
      argon: Infinity,
      primary: Infinity,
      gemini: Infinity,
      cursor: Infinity,
    };

    const stampTransition = (dec: RouteDecision): RouteDecision => {
      const current = dec.tier ?? (dec.route === 'primary' ? 'primary' : dec.route === 'cursor' ? 'cursor' : dec.route === 'argon' ? 'argon' : 'free');
      this.lastServedTier.set(key, current);
      if (prevServed !== undefined && prevServed !== current) {
        return { ...dec, previousTier: prevServed, modelTransition: true };
      }
      return dec;
    };

    const resolve = (tier: Tier, reason: RouteReason, extra: Partial<RouteDecision> = {}): RouteDecision => {
      const escalatedByContext = inputTokens > (budget[tier] ?? Infinity);
      const final: Tier = escalatedByContext ? 'primary' : tier;
      this.sessions.set(key, final);
      if (final === 'gemini' || final === 'free') this.lastRouteWasGemini.add(key);
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
            minCheapProbability: Math.min(this.options.policy.minCheapProbability ?? 0.9, pressure.minCheapProbability),
            minStandardProbability: Math.min(this.options.policy.minStandardProbability ?? 0.75, pressure.minStandardProbability),
          };

    const stored: Tier = this.sessions.get(key) ?? 'primary';
    if (ctx.contextCompacted && stored === 'primary' && level !== 'critical') return resolve('primary', 'pinned:compaction');
    const sticky: Tier | undefined = (fresh || stored === 'gemini') ? undefined : stored;

    const humanText = latestHumanText(ctx.body);
    const isProbe = typeof ctx.body.max_tokens === 'number' && ctx.body.max_tokens <= 1;
    if (isProbe) return primary('classified');
    const loopTier = this.fallbackLoopTier(key);
    const loopActive =
      loopTier && !hasImageDocumentOrToolChoice(ctx.body) && !isProbe ? loopTier : undefined;
    if (loopActive && humanText === undefined) {
      return stampTransition({
        route: tierRoute(loopActive),
        tier: loopActive,
        reason: 'sticky',
        conversationKey: key,
        fallbackLoop: loopActive,
      });
    }

    if (humanText === undefined) {
      if (
        (stored === 'free' || stored === 'gemini' || this.lastRouteWasGemini.has(key)) &&
        !hasImageDocumentOrToolChoice(ctx.body)
      ) {
        return resolve('free', 'sticky', { fallbackTier: 'primary' });
      }
      if (stored === 'argon') {
        return resolve('argon', 'sticky', { fallbackTier: 'primary' });
      }
      if (stored === 'cursor') {
        return resolve('cursor', 'sticky', { fallbackTier: 'primary' });
      }
      if (stored === 'gemini') return primary('sticky');
      return sticky ? resolve(sticky, 'sticky') : primary('passthrough:no-session-state');
    }

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
      const classifierError = err instanceof Error ? err.message : String(err);
      if (classifyForGemini) {
        return resolve('primary', 'sticky', { classifierError });
      }
      return resolve('primary', 'passthrough:classifier-failed', { classifierError });
    }
    const classifierMs = Math.round(performance.now() - started);

    const scored = selectTier(distribution, policy);
    const extra = { distribution, classifierMs };

    if (loopActive) {
      if (hardRiskVeto(distribution)) {
        this.fallbackLoop.delete(key);
        return resolve('primary', 'classified', extra);
      }
      const snap = this.options.quota?.current?.();
      const gateClears =
        snap !== undefined &&
        !(
          (snap.utilization >= 1 || snap.status === 'rejected') &&
          (snap.resetAt === undefined || snap.resetAt.getTime() > Date.now())
        );
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
      this.fallbackLoop.delete(key);
    }

    let isGeminiEligible = false;
    // With Argon on, argon/primary-scored turns follow the Argon chain, not the legacy text-only gate.
    const argonOwnsTurn = this.options.argonPolicy?.enabled === true && (scored === 'argon' || scored === 'primary');
    if (this.options.geminiPolicy && !argonOwnsTurn && !hasImageDocumentOrToolChoice(ctx.body)) {
      const effectiveProposed = classifyForGemini ? 'primary' : scored;
      isGeminiEligible = geminiEligible(distribution, effectiveProposed, this.options.geminiPolicy, level);
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
      const reason: RouteReason = isPressure ? 'gemini:text-only:pressure' : 'gemini:text-only';
      return stampTransition({
        ...extra,
        route: 'free',
        tier: 'gemini',
        fallbackTier: nextStored === 'gemini' ? 'primary' : nextStored,
        reason,
        conversationKey: key,
      });
    }

    if (classifyForGemini) {
      return resolve('primary', 'sticky', extra);
    }

    if (reclassify) return resolve(scored, scored === 'primary' ? 'sticky' : 'quota:reclassified', extra);

    let isQuotaPressure = false;
    if (policy !== this.options.policy) {
      const normalScored = selectTier(distribution, this.options.policy);
      if (normalScored !== scored && scored === 'free') {
        isQuotaPressure = true;
      }
    }

    if (scored === 'free') {
      // Free tier turn
      return resolve('free', isQuotaPressure ? 'quota:pressure' : 'free:flash', { ...extra, fallbackTier: 'primary' });
    }

    if (scored === 'argon') {
      const codingTurn = isCodingTurn(distribution, {
        ...(this.options.cursorCodingMaxTextOnly !== undefined ? { cursorCodingMaxTextOnly: this.options.cursorCodingMaxTextOnly } : {}),
        ...(this.options.cursorCodingMinMass !== undefined ? { cursorCodingMinMass: this.options.cursorCodingMinMass } : {}),
      });
      const cursorActive = this.options.cursorCoding !== false && this.options.cursorEnabled && !hasImageDocumentOrToolChoice(ctx.body);
      if (codingTurn && cursorActive) {
        return resolve('cursor', 'cursor:coding', extra);
      }
      const argonProf = selectArgonProfile(distribution) ?? 'low';
      const argonReason: RouteReason = `argon:${argonProf}`;
      return resolve('argon', argonReason, { ...extra, argonProfile: argonProf });
    }

    if (!sticky) {
      return resolve(scored, 'classified', extra);
    }

    const next = stickyTier(sticky, scored);
    const reason: RouteReason = next === sticky ? 'sticky' : 'escalated';
    return resolve(next, reason, extra);
  }

  async cascadeRiskVeto(body: MessagesBody): Promise<boolean> {
    const text = lastHumanText(body);
    if (text === undefined) return false;
    try {
      const distribution = await this.classifier.classify(
        {
          text: text.slice(0, this.options.classifierMaxChars),
          turnCount: body.messages.length,
          toolCount: Array.isArray(body['tools']) ? body['tools'].length : 0,
          estimatedInputTokens: 0,
        },
        AbortSignal.timeout(this.options.classifierTimeoutMs),
      );
      return hardRiskVeto(distribution);
    } catch {
      return true;
    }
  }

  refusePrimaryProfile(decision: RouteDecision): void {
    if (decision.conversationKey) this.profiles.set(decision.conversationKey, null);
  }

  pinToPrimary(decision: RouteDecision): void {
    if (decision.conversationKey) {
      this.sessions.set(decision.conversationKey, 'primary');
      this.lastRouteWasGemini.delete(decision.conversationKey);
      this.fallbackLoop.delete(decision.conversationKey);
      this.lastServedTier.set(decision.conversationKey, 'primary');
    }
  }

  markFallbackLoop(decision: RouteDecision, served: 'cursor' | 'gemini' | 'argon' | 'free'): void {
    const key = decision.conversationKey;
    if (!key) return;
    this.pruneFallbackLoop();
    this.fallbackLoop.set(key, { tier: served, markedAt: Date.now() });
    this.lastServedTier.set(key, served);
    if (served === 'gemini' || served === 'free') this.lastRouteWasGemini.add(key);
    else this.lastRouteWasGemini.delete(key);
  }

  clearFallbackLoop(decision: RouteDecision, served: Tier): void {
    const key = decision.conversationKey;
    if (key) {
      this.fallbackLoop.delete(key);
      this.lastRouteWasGemini.delete(key);
      this.lastServedTier.set(key, served);
      if (this.sessions.get(key) === 'gemini' || this.sessions.get(key) === 'free') this.sessions.set(key, served);
    }
  }

  clearGeminiLoop(decision: RouteDecision, served: Tier): void {
    this.clearFallbackLoop(decision, served);
  }
}
