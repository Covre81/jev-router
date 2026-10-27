import { createHash, timingSafeEqual } from 'node:crypto';
import { Readable } from 'node:stream';
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from 'fastify';
import type { Dispatcher } from 'undici';
import type { BuildInfo } from '../build-info.js';
import type { Config } from '../config.js';
import {
  hardRiskVeto,
  selectArgonProfile,
  bumpArgonProfile,
  type ArgonProfile,
  type EscalationReason,
  type PrimaryProfile,
  type Route,
  type Tier,
} from '../domain/policy.js';
import type { Provider, ProviderRequest, ProviderResult } from '../providers/provider.js';
import type { QuotaStore } from '../quota.js';
import { buildTransitionHandoff, withTransitionHandoff } from '../context/transition-handoff.js';
import { hasImageDocumentOrToolChoice, latestHumanText, parseMessagesBody, type MessagesBody } from '../routing/messages-body.js';
import { withPrimaryProfile } from '../routing/primary-profile.js';
import type { RouteDecision, RouteReason, Router } from '../routing/router.js';
import { auditExchange, auditFailure, type ExchangeContext } from '../telemetry/audit.js';
import { noopTelemetry, type TelemetrySink } from '../telemetry/recorder.js';
import { forwardableHeaders, presentedCredential, single } from './headers.js';

export interface ServerDeps {
  readonly config: Config;
  readonly router: Router;
  readonly providers: {
    readonly primary: Provider;
    readonly gemini?: Provider | undefined;
    readonly cursor?: Provider | undefined;
  };
  readonly telemetry?: TelemetrySink;
  readonly build?: BuildInfo;
  readonly quota?: QuotaStore;
  readonly geminiBreaker?: import('../providers/breaker.js').CircuitBreaker;
  readonly argonBreaker?: import('../providers/breaker.js').CircuitBreaker;
  readonly cursorBreaker?: import('../providers/breaker.js').CircuitBreaker;
  readonly timeouts?: {
    readonly firstByteMs?: number;
    readonly idleMs?: number;
    readonly keepaliveIntervalMs?: number;
  };
}

/** Primary statuses that mean "no quota/capacity right now" (opt-in failover). */
const PRIMARY_RATE_LIMITED = new Set([429, 529]);

function anthropicError(type: string, message: string) {
  return { type: 'error', error: { type, message } };
}

function sameSecret(a: string, b: string): boolean {
  const x = Buffer.from(a);
  const y = Buffer.from(b);
  return x.length === y.length && timingSafeEqual(x, y);
}

let lastQuotaHeaderNamesStr: string | undefined;
let loggedNoQuotaHeaders = false;

export function rateLimitHeaderNames(headers: Record<string, string | string[]>): string[] {
  return Object.keys(headers)
    .filter((k) => k.toLowerCase().startsWith('anthropic-ratelimit'))
    .sort();
}

export function isQuotaProbe(body: MessagesBody | undefined): boolean {
  return typeof body?.max_tokens === 'number' && body.max_tokens <= 1;
}

export function failureCode(reason: string): string {
  if (reason.startsWith('HTTP ')) {
    const match = reason.match(/^HTTP \d{3}/);
    if (match) return match[0];
  }
  const low = reason.toLowerCase();
  if (low.includes('first-byte-timeout')) return 'first-byte-timeout';
  if (low.includes('idle-timeout')) return 'idle-timeout';
  if (low.includes('empty')) return 'empty';
  if (low.startsWith('exit')) return 'exit';
  if (low.includes('timeout') || low.includes('deadline')) return 'timeout';
  if (low.startsWith('network:')) return 'network';
  if (low.startsWith('not translatable')) return 'not translatable';
  if (low.startsWith('bad response')) return 'bad response';
  if (low === 'aborted') return 'aborted';
  return 'other';
}

class StreamKeepAlive {
  private timer: NodeJS.Timeout | undefined;
  private headersFlushed = false;

  constructor(
    private readonly reply: FastifyReply,
    private readonly intervalMs: number,
  ) {}

  start() {
    this.timer = setInterval(() => {
      this.sendPing();
    }, this.intervalMs);
  }

  sendPing() {
    if (this.reply.raw.destroyed || this.reply.raw.writableEnded) {
      this.stop();
      return;
    }
    if (!this.headersFlushed) {
      this.headersFlushed = true;
      this.reply.raw.writeHead(200, {
        'content-type': 'text/event-stream',
        'cache-control': 'no-cache',
        connection: 'keep-alive',
      });
    }
    this.reply.raw.write('event: ping\ndata: {"type": "ping"}\n\n');
  }

  stop() {
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  didSendHeaders(): boolean {
    return this.headersFlushed || this.reply.raw.headersSent;
  }
}

export function buildServer({
  config,
  router,
  providers,
  telemetry = noopTelemetry,
  build,
  quota,
  geminiBreaker,
  argonBreaker,
  cursorBreaker,
  timeouts,
}: ServerDeps): FastifyInstance {
  const startedAt = new Date().toISOString();

  const app = Fastify({
    logger: { level: config.logLevel },
    bodyLimit: config.bodyLimitBytes,
    exposeHeadRoutes: false,
  });

  const sendPrimary = async (req: ProviderRequest): Promise<ProviderResult> => {
    const result = await providers.primary.send(req);
    if (result.kind === 'response') {
      quota?.observe(result.headers);
      if (config.telemetry?.diagnostics && req.url.startsWith('/v1/messages')) {
        try {
          const names = rateLimitHeaderNames(result.headers);
          if (names.length > 0) {
            const namesStr = names.join(',');
            if (namesStr !== lastQuotaHeaderNamesStr) {
              lastQuotaHeaderNamesStr = namesStr;
              app.log.info({ quotaHeaderNames: names }, 'anthropic rate-limit headers seen');
            }
          } else if (result.status >= 200 && result.status < 300) {
            if (!loggedNoQuotaHeaders) {
              loggedNoQuotaHeaders = true;
              app.log.info('no anthropic-ratelimit headers on responses');
            }
          }
        } catch (err) {}
      }
    }
    return result;
  };

  app.removeAllContentTypeParsers();
  app.addContentTypeParser('*', { parseAs: 'buffer' }, (_req, body, done) => done(null, body));

  const proxyToken = config.primary.authMode === 'inject' ? config.proxyAuthToken : undefined;
  if (proxyToken) {
    app.addHook('onRequest', async (req, reply) => {
      if (req.url === '/healthz' || req.url.startsWith('/route/')) return;
      const presented = presentedCredential(req.headers);
      if (!presented || !sameSecret(presented, proxyToken)) {
        return reply.code(401).send(anthropicError('authentication_error', 'invalid proxy token'));
      }
    });
  }

  app.get('/healthz', async () => ({
    ok: true,
    sha: build?.sha ?? null,
    builtAt: build?.builtAt ?? null,
    startedAt,
    pid: process.pid,
    quota: quota?.current() ? { ...quota.current(), level: quota.level() } : null,
    gemini: config.gemini ? { model: config.gemini.model, breaker: geminiBreaker?.state, inFlight: geminiBreaker?.inFlight } : null,
    argon: config.argon.enabled ? { breaker: argonBreaker?.state, inFlight: argonBreaker?.inFlight } : null,
    cursor: config.cursor ? { model: config.cursor.model, breaker: cursorBreaker?.state, inFlight: cursorBreaker?.inFlight } : null,
  }));

  const lastRoutes = new Map<string, { route: string; reason: string; at: string }>();
  app.get<{ Params: { session: string } }>('/route/:session', async (req, reply) => {
    const last = lastRoutes.get(`${req.params.session}:main`);
    return last ?? reply.code(404).send({ error: 'no decision for this session' });
  });

  app.post('/v1/messages', handleMessages);
  app.all('*', async (req, reply) => relay(reply, await sendPrimary(providerRequest(req, reply))));

  async function handleMessages(req: FastifyRequest, reply: FastifyReply) {
    const startedAt = performance.now();
    const startedAtMs = Date.now();
    const base = providerRequest(req, reply);
    if (!base.body || !base.rawBody) return relay(reply, await sendPrimary(base));

    const isStreaming = (base.body as { stream?: boolean }).stream === true;
    const keepaliveInterval = timeouts?.keepaliveIntervalMs ?? config.timeouts?.keepaliveIntervalMs ?? 15_000;
    const keepAlive = isStreaming ? new StreamKeepAlive(reply, keepaliveInterval) : null;
    if (keepAlive) keepAlive.start();

    const context = {
      body: base.body,
      rawByteLength: base.rawBody.length,
      sessionId: single(req.headers, 'x-claude-code-session-id'),
      agentId: single(req.headers, 'x-claude-code-agent-id'),
      requestClass: single(req.headers, 'x-claude-code-request-class'),
      contextCompacted: single(req.headers, 'x-claude-code-context-compacted') !== undefined,
    };
    const decision = await router.decide(context);

    let activeBase = base;
    if (decision.modelTransition && decision.previousTier && decision.tier && decision.tier !== 'primary' && base.body) {
      try {
        const handoffBlock = await buildTransitionHandoff({
          body: base.body,
          previousTier: decision.previousTier,
          currentTier: decision.tier,
        });
        const handoffBody = withTransitionHandoff(base.body, handoffBlock);
        activeBase = {
          ...base,
          body: handoffBody,
          rawBody: Buffer.from(JSON.stringify(handoffBody)),
        };
        req.log.info(
          { conversation: decision.conversationKey, from: decision.previousTier, to: decision.tier },
          'model transition handoff injected',
        );
      } catch (err) {
        req.log.warn({ err, conversation: decision.conversationKey }, 'model transition handoff failed');
      }
    }

    let upstreamFailureState: { failure?: string; status?: number; rateLimitHeaders?: string } | undefined;

    const logHopFailure = (hop: 'flash' | 'argon' | 'cursor' | 'claude', rawReason: string, durationMs: number) => {
      const code = failureCode(rawReason);
      req.log.warn({ hop, reason: code, detail: rawReason.slice(0, 200), ms: Math.round(durationMs) }, 'hop failed');
    };

    const exchange = (route: Route, modelOverride?: string, effortOverride?: string): ExchangeContext => {
      let diagnostics: ExchangeContext['diagnostics'];
      if (config.telemetry?.diagnostics) {
        try {
          const body = base.body!;
          const stringify = (v: unknown) => (v !== undefined ? JSON.stringify(v) : undefined);
          const hash16 = (s: string | undefined) => (s ? createHash('sha256').update(s).digest('hex').slice(0, 16) : undefined);
          const sysStr = stringify(body.system);
          const toolsStr = stringify(body['tools']);
          const ccCompaction = single(req.headers, 'x-claude-code-compaction');
          const ccContextCompacted = single(req.headers, 'x-claude-code-context-compacted');
          const compaction = ccCompaction ?? (ccContextCompacted ? `context-compacted:${ccContextCompacted}` : undefined);

          diagnostics = {
            ccSessionId: single(req.headers, 'x-claude-code-session-id'),
            agentType: single(req.headers, 'x-claude-code-agent-type'),
            promptId: single(req.headers, 'x-claude-code-prompt-id'),
            compaction,
            systemHash: hash16(sysStr),
            toolsHash: hash16(toolsStr),
            systemChars: sysStr?.length,
            toolsChars: toolsStr?.length,
            maxTokens: typeof body.max_tokens === 'number' ? body.max_tokens : undefined,
            thinkingType: typeof body['thinking'] === 'object' && body['thinking'] ? (body['thinking'] as any).type : undefined,
            thinkingBudget: typeof body['thinking'] === 'object' && body['thinking'] ? (body['thinking'] as any).budget_tokens : undefined,
          };
        } catch (err) {
          req.log.warn({ err }, 'diagnostics capture failed');
        }
      }
      return {
        startedAt,
        startedAtMs,
        humanText: latestHumanText(base.body!),
        requestClass: single(req.headers, 'x-claude-code-request-class'),
        model:
          route === 'primary'
            ? (modelOverride ?? base.body!.model)
            : route === 'free' || route === 'gemini'
              ? (modelOverride ?? config.gemini?.model ?? 'gemini-3.8-flash-high')
              : route === 'argon'
                ? (modelOverride ?? config.argon.modelMedium)
                : (modelOverride ?? config.cursor?.model ?? 'auto'),
        requestedModel: base.body!.model,
        // The body's effort is Claude's; flash/cursor have none, argon passes its profile.
        effort:
          effortOverride ??
          (route === 'primary' ? (base.body!['output_config'] as { effort?: string } | undefined)?.effort : undefined),
        toolsOffered: Array.isArray(base.body!['tools']) ? base.body!['tools'].length : 0,
        quotaUtilization: quota?.current()?.utilization,
        diagnostics,
        upstream: upstreamFailureState,
      };
    };

    const tryCursor = async (r: ProviderRequest): Promise<ProviderResult | undefined> => {
      const provider = providers.cursor;
      const breaker = cursorBreaker;
      if (!provider || !breaker || breaker.state === 'open') return undefined;
      if (!breaker.acquire()) return undefined;
      const hopStart = performance.now();
      let outcome: 'success' | 'failure' | 'neutral' = 'failure';
      try {
        let res: ProviderResult;
        try {
          res = await provider.send(r);
        } catch (err) {
          const rawReason = err instanceof Error ? err.message : String(err);
          logHopFailure('cursor', rawReason, performance.now() - hopStart);
          upstreamFailureState = { failure: failureCode(rawReason) };
          return undefined;
        }
        if (res.kind === 'response') {
          outcome = 'success';
          return res;
        }
        const isRoutingMiss = res.reason === 'model asked for tools' || res.reason.startsWith('tool attempt:');
        const overBudget = res.reason === 'transcript budget exceeded';
        if (res.reason === 'aborted' || isRoutingMiss || overBudget) outcome = 'neutral';
        logHopFailure('cursor', res.reason, performance.now() - hopStart);
        upstreamFailureState = { failure: failureCode(res.reason) };
        const statusMatch = res.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return undefined;
      } finally {
        breaker.release(outcome);
      }
    };

    type FlashResult =
      | { kind: 'response'; result: ProviderResult }
      | { kind: 'skipped'; reason: 'skipped:gemini-busy' | 'skipped:gemini-unhealthy' }
      | { kind: 'failed'; reason: 'failover:gemini-unavailable' };

    const tryFlash = async (r: ProviderRequest): Promise<FlashResult> => {
      const provider = providers.gemini;
      const breaker = geminiBreaker;
      if (!provider || !breaker || breaker.state === 'open') {
        upstreamFailureState = { failure: 'skipped' };
        return { kind: 'skipped', reason: 'skipped:gemini-unhealthy' };
      }
      if (!breaker.acquire()) {
        upstreamFailureState = { failure: 'skipped' };
        return { kind: 'skipped', reason: 'skipped:gemini-busy' };
      }
      const hopStart = performance.now();
      let outcome: 'success' | 'failure' | 'neutral' = 'failure';
      try {
        let res: ProviderResult;
        try {
          res = await provider.send(r);
        } catch (err) {
          const rawReason = err instanceof Error ? err.message : String(err);
          logHopFailure('flash', rawReason, performance.now() - hopStart);
          upstreamFailureState = { failure: failureCode(rawReason) };
          return { kind: 'failed', reason: 'failover:gemini-unavailable' };
        }
        if (res.kind === 'response') {
          outcome = 'success';
          return { kind: 'response', result: res };
        }
        const isRoutingMiss = res.reason === 'model asked for tools' || res.reason.startsWith('tool attempt:');
        const overBudget = res.reason === 'transcript budget exceeded';
        if (res.reason === 'aborted' || isRoutingMiss || overBudget) outcome = 'neutral';
        logHopFailure('flash', res.reason, performance.now() - hopStart);
        upstreamFailureState = { failure: failureCode(res.reason) };
        const statusMatch = res.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return { kind: 'failed', reason: 'failover:gemini-unavailable' };
      } finally {
        breaker.release(outcome);
      }
    };

    const getArgonModel = (profile: ArgonProfile): string => {
      if (profile === 'low') return config.argon.modelLow;
      if (profile === 'high') return config.argon.modelHigh;
      return config.argon.modelMedium;
    };

    const tryArgon = async (
      profile: ArgonProfile,
      r: ProviderRequest,
    ): Promise<{ kind: 'response'; result: ProviderResult } | { kind: 'skipped' } | { kind: 'failed'; reason: string }> => {
      const provider = providers.gemini;
      const breaker = argonBreaker ?? geminiBreaker;
      if (!config.argon.enabled || !provider || !breaker || breaker.state === 'open') {
        return { kind: 'skipped' };
      }
      if (!breaker.acquire()) {
        return { kind: 'skipped' };
      }
      const hopStart = performance.now();
      const model = getArgonModel(profile);
      let outcome: 'success' | 'failure' | 'neutral' = 'failure';
      try {
        let res: ProviderResult;
        try {
          res = await provider.send({ ...r, modelOverride: model });
        } catch (err) {
          const rawReason = err instanceof Error ? err.message : String(err);
          logHopFailure('argon', rawReason, performance.now() - hopStart);
          upstreamFailureState = { failure: failureCode(rawReason) };
          return { kind: 'failed', reason: rawReason };
        }
        if (res.kind === 'response') {
          outcome = 'success';
          return { kind: 'response', result: res };
        }
        const isRoutingMiss = res.reason === 'model asked for tools' || res.reason.startsWith('tool attempt:');
        const overBudget = res.reason === 'transcript budget exceeded';
        if (res.reason === 'aborted' || isRoutingMiss || overBudget) outcome = 'neutral';
        logHopFailure('argon', res.reason, performance.now() - hopStart);
        upstreamFailureState = { failure: failureCode(res.reason) };
        const statusMatch = res.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return { kind: 'failed', reason: res.reason };
      } finally {
        breaker.release(outcome);
      }
    };

    const sendClaude = async (
      b: ProviderRequest,
      dec: RouteDecision,
    ): Promise<{ result: ProviderResult; dec: RouteDecision; served?: PrimaryProfile }> => {
      const profile = dec.primaryProfile;
      const profiled = profile && b.rawBody ? withPrimaryProfile(b.rawBody, profile) : undefined;
      if (!profile || !profiled) return { result: await sendPrimary(b), dec };
      const first = await sendPrimary({ ...b, rawBody: profiled });
      if (first.kind !== 'response' || first.status !== 400) return { result: first, dec, served: profile };
      const error = Buffer.concat(await first.body.toArray()).toString('utf8').slice(0, 500);
      const result = await sendPrimary(b);
      if (result.kind !== 'response' || result.status < 200 || result.status >= 300) return { result, dec };
      req.log.warn({ profile, error }, 'primary profile refused, cleared');
      router.refusePrimaryProfile(dec);
      const refused: RouteDecision = { ...dec, reason: 'profile:refused' };
      return { result, dec: refused };
    };

    const relayClaude = async (b: ProviderRequest, dec: RouteDecision) => {
      const c = await sendClaude(b, dec);
      return safeRelay(c.result, c.dec, exchange('primary', c.served?.model, c.served?.effort));
    };

    const safeRelay = async (
      result: ProviderResult,
      dec: RouteDecision,
      exch?: ExchangeContext,
    ) => {
      if (keepAlive) keepAlive.stop();

      const servedModel = exch?.model ?? (dec.route === 'primary' ? base.body!.model : config.gemini?.model ?? 'gemini-3.8-flash-high');
      const servedEffort = exch?.effort;
      req.log.info(
        { route: dec.route, tier: dec.tier ?? dec.route, model: servedModel, effort: servedEffort, reason: dec.reason },
        'routing decision',
      );

      if (keepAlive && keepAlive.didSendHeaders()) {
        if (result.kind === 'response') {
          for await (const chunk of result.body) {
            reply.raw.write(chunk);
          }
          reply.raw.end();
        } else {
          reply.raw.write(
            `event: error\ndata: {"type":"error","error":{"type":"api_error","message":"jev-router: ${result.reason}"}}\n\n`,
          );
          reply.raw.end();
        }
        return reply;
      }
      return relay(reply, result, dec, req, exch);
    };

    // Execution chain based on routed decision
    if (decision.route === 'cursor') {
      const cursorRes = hasImageDocumentOrToolChoice(base.body!) ? undefined : await tryCursor(activeBase);
      if (cursorRes) {
        router.markFallbackLoop(decision, 'cursor');
        return safeRelay(cursorRes, decision, exchange('cursor'));
      }

      // Cursor failed -> fall back to Argon (starting profile from selectArgonProfile)
      let curProfile: ArgonProfile | null = selectArgonProfile(decision.distribution ?? { simple: 0, standard: 1, structural: 0 }) ?? 'low';
      while (curProfile !== null) {
        const argonAttempt = await tryArgon(curProfile, activeBase);
        if (argonAttempt.kind === 'response') {
          router.markFallbackLoop(decision, 'argon');
          const argonModel = getArgonModel(curProfile);
          const routeDec: RouteDecision = { ...decision, route: 'argon', tier: 'argon', reason: 'escalated:cursor-failed' };
          return safeRelay(argonAttempt.result, routeDec, exchange('argon', argonModel, curProfile));
        }
        if (argonAttempt.kind === 'skipped') {
          break; // skip without bump to Claude
        }
        curProfile = bumpArgonProfile(curProfile);
      }

      // Argon exhausted or skipped -> Claude
      router.clearFallbackLoop(decision, 'primary');
      const routeDec: RouteDecision = await router.profileForFallback(
        { ...decision, route: 'primary', tier: 'primary', reason: 'escalated:argon-failed' },
        context,
        'gemini-failed',
      );
      return relayClaude(base, routeDec);
    }

    if (decision.route === 'free') {
      const flashRes = hasImageDocumentOrToolChoice(base.body!)
        ? { kind: 'skipped' as const, reason: 'skipped:gemini-unhealthy' as const }
        : await tryFlash(activeBase);
      if (flashRes.kind === 'response') {
        router.markFallbackLoop(decision, 'free');
        return safeRelay(flashRes.result, decision, exchange('free', config.gemini?.model));
      }

      const flashFailureReason = flashRes.reason;
      let argonFailed = false;
      if (config.argon.enabled && flashRes.kind !== 'skipped') {
        let curProfile: ArgonProfile | null = selectArgonProfile(decision.distribution ?? { simple: 1, standard: 0, structural: 0 }) ?? 'low';
        while (curProfile !== null) {
          const argonAttempt = await tryArgon(curProfile, activeBase);
          if (argonAttempt.kind === 'response') {
            router.markFallbackLoop(decision, 'argon');
            const argonModel = getArgonModel(curProfile);
            const routeDec: RouteDecision = { ...decision, route: 'argon', tier: 'argon', reason: 'escalated:flash-failed' };
            return safeRelay(argonAttempt.result, routeDec, exchange('argon', argonModel, curProfile));
          }
          if (argonAttempt.kind === 'skipped') {
            break;
          }
          argonFailed = true;
          curProfile = bumpArgonProfile(curProfile);
        }
      }

      // Claude fallback
      if (flashFailureReason !== 'skipped:gemini-busy') {
        router.clearGeminiLoop(decision, 'primary');
      }
      const fallbackReason = argonFailed ? 'escalated:argon-failed' : flashFailureReason;
      const escalationKind: EscalationReason = flashFailureReason === 'skipped:gemini-busy' ? 'gemini-busy' : 'gemini-failed';
      const routeDec: RouteDecision = await router.profileForFallback(
        { ...decision, route: 'primary', tier: 'primary', reason: fallbackReason },
        context,
        escalationKind,
      );
      return relayClaude(base, routeDec);
    }

    if (decision.route === 'argon') {
      let curProfile: ArgonProfile | null = decision.argonProfile ?? selectArgonProfile(decision.distribution ?? { simple: 0, standard: 1, structural: 0 }) ?? 'low';
      let isFirst = true;
      while (curProfile !== null) {
        const argonAttempt = await tryArgon(curProfile, activeBase);
        if (argonAttempt.kind === 'response') {
          router.markFallbackLoop(decision, 'argon');
          const argonModel = getArgonModel(curProfile);
          const reason: RouteReason = isFirst ? (`argon:${curProfile}` as RouteReason) : 'escalated:argon-failed';
          const routeDec: RouteDecision = { ...decision, route: 'argon', tier: 'argon', reason };
          return safeRelay(argonAttempt.result, routeDec, exchange('argon', argonModel, curProfile));
        }
        if (argonAttempt.kind === 'skipped') {
          break;
        }
        isFirst = false;
        curProfile = bumpArgonProfile(curProfile);
      }

      router.clearFallbackLoop(decision, 'primary');
      const routeDec: RouteDecision = await router.profileForFallback(
        { ...decision, route: 'primary', tier: 'primary', reason: 'escalated:argon-failed' },
        context,
        'gemini-failed',
      );
      return relayClaude(base, routeDec);
    }

    // Default: primary (Claude)
    const claude = await sendClaude(activeBase, decision);
    const primary = claude.result;
    const primaryDecision = claude.dec;

    if (
      primary.kind === 'unavailable' ||
      !config.router.failoverOnPrimaryRateLimit ||
      !PRIMARY_RATE_LIMITED.has(primary.status)
    ) {
      return safeRelay(primary, primaryDecision, exchange('primary', claude.served?.model, claude.served?.effort));
    }

    if (isQuotaProbe(base.body)) {
      const quotaHeaderNames = rateLimitHeaderNames(primary.headers);
      req.log.info({ status: primary.status, quotaHeaderNames }, 'primary rate-limited probe passed through');
      upstreamFailureState = { status: primary.status, rateLimitHeaders: quotaHeaderNames.join(',') };
      return safeRelay(primary, primaryDecision, exchange('primary', claude.served?.model, claude.served?.effort));
    }

    const quotaHeaderNames = rateLimitHeaderNames(primary.headers);
    req.log.warn({ status: primary.status, quotaHeaderNames, conversation: decision.conversationKey }, 'primary rate-limited, failing over (cascade Cursor -> Gemini)');

    const errorBody = Buffer.concat(await primary.body.toArray());
    const skipCli = hasImageDocumentOrToolChoice(base.body!);
    const primaryFailure = { status: primary.status, rateLimitHeaders: quotaHeaderNames.join(',') };

    const vetoed = decision.distribution ? hardRiskVeto(decision.distribution) : await router.cascadeRiskVeto(base.body!);
    if (vetoed) {
      upstreamFailureState = primaryFailure;
      const original: ProviderResult = { ...primary, body: Readable.from([errorBody]) };
      return safeRelay(original, decision, exchange('primary'));
    }

    if (!skipCli) {
      const cursorRes = await tryCursor(activeBase);
      if (cursorRes) {
        router.markFallbackLoop(decision, 'cursor');
        upstreamFailureState = primaryFailure;
        return safeRelay(
          cursorRes,
          { ...decision, route: 'cursor', tier: 'cursor', reason: 'failover:primary-rate-limited', fallbackLoop: 'cursor' },
          exchange('cursor'),
        );
      }

      const flashRes = await tryFlash(activeBase);
      if (flashRes.kind === 'response') {
        router.markFallbackLoop(decision, 'free');
        upstreamFailureState = primaryFailure;
        return safeRelay(
          flashRes.result,
          {
            ...decision,
            route: 'free',
            tier: 'free',
            reason: providers.cursor ? 'failover:cursor-unavailable' : 'failover:primary-rate-limited',
            fallbackLoop: 'free',
          },
          exchange('free', config.gemini?.model),
        );
      }
    }

    // All cascade tiers failed: relay original 429/529
    const original: ProviderResult = { ...primary, body: Readable.from([errorBody]) };
    return safeRelay(original, decision, exchange('primary'));
  }

  function providerRequest(req: FastifyRequest, reply: FastifyReply): ProviderRequest {
    const controller = new AbortController();
    reply.raw.once('close', () => {
      if (!reply.raw.writableFinished) controller.abort();
    });
    const rawBody = Buffer.isBuffer(req.body) && req.body.length > 0 ? req.body : undefined;
    const body: MessagesBody | undefined = rawBody && req.url.startsWith('/v1/messages') ? parseMessagesBody(rawBody) : undefined;
    return {
      method: req.method as Dispatcher.HttpMethod,
      url: req.url,
      headers: forwardableHeaders(req.headers),
      rawBody,
      body,
      signal: controller.signal,
    };
  }

  function relay(
    reply: FastifyReply,
    result: ProviderResult,
    decision?: RouteDecision,
    req?: FastifyRequest,
    exchange?: ExchangeContext,
  ) {
    const audited = decision && exchange;
    if (result.kind === 'unavailable') {
      reply.log.warn({ route: decision?.route, reason: result.reason }, 'upstream unavailable');
      if (audited) auditFailure(telemetry, decision, exchange, result.reason);
      return reply.code(502).send(anthropicError('api_error', `jev-router: ${result.reason}`));
    }
    if (decision) {
      reply.header('x-jev-route', `${decision.route}; reason=${decision.reason}`);
      const key = decision.conversationKey;
      if (key?.endsWith(':main')) {
        lastRoutes.delete(key);
        lastRoutes.set(key, { route: decision.route, reason: decision.reason, at: new Date().toISOString() });
        if (lastRoutes.size > 500) lastRoutes.delete(lastRoutes.keys().next().value!);
      }
    }
    reply.code(result.status).headers(result.headers);
    const body = audited
      ? auditExchange(telemetry, reply.raw, result, decision, exchange, (err) =>
          reply.log.warn({ err }, 'telemetry failed'),
        )
      : result.body;
    return reply.send(body);
  }

  app.setErrorHandler((err, req, reply) => {
    if (reply.raw.destroyed) return;
    if (reply.raw.headersSent) {
      try {
        reply.raw.write(`event: error\ndata: {"type":"error","error":{"type":"api_error","message":"jev-router: upstream error"}}\n\n`);
        reply.raw.end();
      } catch {}
      return;
    }
    const status = (err as unknown as { statusCode?: number }).statusCode;
    if (status !== undefined && status >= 400 && status < 500) {
      const message = (err as unknown as { message?: string }).message || 'request error';
      return reply.code(status).send(anthropicError(status === 413 ? 'request_too_large' : 'invalid_request_error', message));
    }
    req.log.error({ err }, 'request failed');
    return reply.code(502).send(anthropicError('api_error', 'jev-router: upstream unreachable'));
  });

  return app;
}
