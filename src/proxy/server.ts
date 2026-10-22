import { createHash, timingSafeEqual } from 'node:crypto';
import { Readable } from 'node:stream';
import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from 'fastify';
import type { Dispatcher } from 'undici';
import type { BuildInfo } from '../build-info.js';
import type { Config } from '../config.js';
import { hardRiskVeto, type PrimaryProfile, type Route, type Tier } from '../domain/policy.js';
import type { CheapState } from '../providers/cheap-health.js';
import type { Provider, ProviderRequest, ProviderResult } from '../providers/provider.js';
import type { QuotaStore } from '../quota.js';
import { buildTransitionHandoff, withTransitionHandoff } from '../context/transition-handoff.js';
import { hasImageDocumentOrToolChoice, latestHumanText, parseMessagesBody, type MessagesBody } from '../routing/messages-body.js';
import { withPrimaryProfile } from '../routing/primary-profile.js';
import { needsReadFirstHint, withReadFirstHint, needsReadFirstFollowup, READ_FIRST_FOLLOWUP } from '../routing/read-first-hint.js';
import type { RouteDecision, RouteReason, Router } from '../routing/router.js';
import { auditExchange, auditFailure, type ExchangeContext } from '../telemetry/audit.js';
import { noopTelemetry, type TelemetrySink } from '../telemetry/recorder.js';
import { forwardableHeaders, presentedCredential, single } from './headers.js';

export interface ServerDeps {
  readonly config: Config;
  readonly router: Router;
  /** `standard` serves the standard tier; without it that tier falls back to `cheap`. */
  readonly providers: Readonly<Record<'cheap' | 'primary', Provider>> & { readonly standard?: Provider | undefined; readonly gemini?: Provider | undefined; readonly cursor?: Provider | undefined };
  readonly telemetry?: TelemetrySink;
  /** Live cheap-provider health; absent = never checked ('unknown'). */
  readonly cheapHealth?: { readonly state: CheapState };
  /** The build this process runs; absent when running from src. */
  readonly build?: BuildInfo;
  /** Fed from every Anthropic response's rate-limit headers; absent = quota routing off. */
  readonly quota?: QuotaStore;
  readonly geminiBreaker?: import('../providers/breaker.js').CircuitBreaker;
  readonly cursorBreaker?: import('../providers/breaker.js').CircuitBreaker;
}

/** Primary statuses that mean "no quota/capacity right now" (opt-in failover to cheap). */
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
  if (low.includes('timeout') || low.includes('deadline')) return 'timeout';
  if (low.startsWith('network:')) return 'network';
  if (low.startsWith('not translatable')) return 'not translatable';
  if (low.startsWith('bad response')) return 'bad response';
  if (low === 'aborted') return 'aborted';
  return 'other';
}

export function buildServer({
  config,
  router,
  providers,
  telemetry = noopTelemetry,
  cheapHealth,
  build,
  quota,
  geminiBreaker,
  cursorBreaker,
}: ServerDeps): FastifyInstance {
  const startedAt = new Date().toISOString();

  const app = Fastify({
    logger: { level: config.logLevel },
    bodyLimit: config.bodyLimitBytes,
    exposeHeadRoutes: false,
  });

  /** Every Anthropic answer carries the quota: read it on the way through, whatever the path. */
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

  // Keep every body as raw bytes: primary-bound requests are forwarded byte-for-byte.
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

  // Which build answers: a merge that was built but never reloaded shows up here and in the status line.
  app.get('/healthz', async () => ({
    ok: true,
    sha: build?.sha ?? null,
    builtAt: build?.builtAt ?? null,
    startedAt,
    pid: process.pid,
    cheap: cheapHealth?.state ?? 'unknown',
    quota: quota?.current() ? { ...quota.current(), level: quota.level() } : null,
    gemini: config.gemini ? { model: config.gemini.model, breaker: geminiBreaker?.state, inFlight: geminiBreaker?.inFlight } : null,
    cursor: config.cursor ? { model: config.cursor.model, breaker: cursorBreaker?.state, inFlight: cursorBreaker?.inFlight } : null,
  }));
  // Why the main loop's last answer came from where it did: the session-telemetry mod shows it on a model switch.
  const lastRoutes = new Map<string, { route: string; reason: string; at: string }>();
  app.get<{ Params: { session: string } }>('/route/:session', async (req, reply) => {
    const last = lastRoutes.get(`${req.params.session}:main`);
    return last ?? reply.code(404).send({ error: 'no decision for this session' });
  });
  app.post('/v1/messages', handleMessages);
  // count_tokens, /v1/models, HEAD /api/hello, … are Anthropic concerns: pass through.
  app.all('*', async (req, reply) => relay(reply, await sendPrimary(providerRequest(req, reply))));

  async function handleMessages(req: FastifyRequest, reply: FastifyReply) {
    const startedAt = performance.now();
    const startedAtMs = Date.now();
    const base = providerRequest(req, reply);
    // Unparseable → let Anthropic return its canonical validation error.
    if (!base.body || !base.rawBody) return relay(reply, await sendPrimary(base));

    const decision = await router.decide({
      body: base.body,
      rawByteLength: base.rawBody.length,
      sessionId: single(req.headers, 'x-claude-code-session-id'),
      agentId: single(req.headers, 'x-claude-code-agent-id'),
      requestClass: single(req.headers, 'x-claude-code-request-class'),
      contextCompacted: single(req.headers, 'x-claude-code-context-compacted') !== undefined,
    });

    const standard = decision.tier === 'standard' && providers.standard !== undefined;
    const cheapProvider = standard ? providers.standard! : providers.cheap;
    const cheapModel = standard ? (config.cheapStandard?.model ?? config.cheap.model) : config.cheap.model;

    let activeBase = base;
    // Never on Claude: it already reads the whole transcript, and a changed system prompt rewrites its prompt cache.
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

    let cheapReq = { ...activeBase };
    if (decision.route === 'cheap' && config.cheapReadFirstHint && activeBase.body && needsReadFirstHint(activeBase.body)) {
      const hintedBody = withReadFirstHint(activeBase.body);
      cheapReq = { ...cheapReq, body: hintedBody, rawBody: Buffer.from(JSON.stringify(hintedBody)) };
      req.log.info({ conversation: decision.conversationKey }, 'read-first hint added');
    } else if (decision.route === 'cheap' && config.cheapReadFirstHint && activeBase.body && needsReadFirstFollowup(activeBase.body)) {
      const hintedBody = withReadFirstHint(activeBase.body, READ_FIRST_FOLLOWUP);
      cheapReq = { ...cheapReq, body: hintedBody, rawBody: Buffer.from(JSON.stringify(hintedBody)) };
      req.log.info({ conversation: decision.conversationKey }, 'read-first follow-up added');
    }

    let upstreamFailureState: { failure?: string; status?: number; rateLimitHeaders?: string } | undefined;

    const doSendCheapWithRetry = async (provider: Provider, r: ProviderRequest): Promise<ProviderResult> => {
      const first = await provider.send(r);
      if (first.kind === 'response') return first;
      
      const fc = failureCode(first.reason);
      const isRetryable = config.cheapRetry.enabled && !r.signal.aborted && (fc.startsWith('HTTP 5') || fc === 'network' || fc === 'timeout');
      
      if (!isRetryable) {
        upstreamFailureState = { failure: fc };
        const statusMatch = first.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return first;
      }
      
      req.log.info({ reason: fc }, 'cheap provider failed, retrying once');
      
      try {
        await new Promise<void>((resolve, reject) => {
          const timer = setTimeout(resolve, config.cheapRetry.delayMs);
          const abort = () => { clearTimeout(timer); reject(new Error('aborted')); };
          if (r.signal.aborted) return abort();
          r.signal.addEventListener('abort', abort, { once: true });
        });
      } catch (err) {
        upstreamFailureState = { failure: fc };
        const statusMatch = first.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return first;
      }
      
      const second = await provider.send({ ...r, headersTimeoutMs: config.cheapRetry.headersTimeoutMs });
      if (second.kind === 'response') {
        upstreamFailureState = { failure: `retried: ${fc}` };
        return second;
      }
      
      const finalFc = failureCode(second.reason);
      upstreamFailureState = { failure: `retried: ${finalFc}` };
      const statusMatch = second.reason.match(/^HTTP (\d+)/);
      if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
      return second;
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
            effort: effortOverride ?? (typeof body['output_config'] === 'object' && body['output_config'] ? (body['output_config'] as any).effort : undefined),
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
        model: route === 'primary' ? (modelOverride ?? base.body!.model) : route === 'gemini' ? config.gemini?.model : route === 'cursor' ? config.cursor?.model : (modelOverride ?? cheapModel),
        requestedModel: base.body!.model,
        toolsOffered: Array.isArray(base.body!['tools']) ? base.body!['tools'].length : 0,
        quotaUtilization: quota?.current()?.utilization,
        diagnostics,
        upstream: upstreamFailureState,
      };
    };

    type CliTier = 'cursor' | 'gemini';
    /** One breaker-gated attempt on a CLI tier; undefined = unavailable/skipped, caller advances the cascade. */
    const tryCliTier = async (tier: CliTier, r: ProviderRequest): Promise<ProviderResult | undefined> => {
      const provider = tier === 'cursor' ? providers.cursor : providers.gemini;
      const breaker = tier === 'cursor' ? cursorBreaker : geminiBreaker;
      if (!provider || !breaker || breaker.state === 'open') return undefined;
      if (!breaker.acquire()) return undefined;
      let outcome: 'success' | 'failure' | 'neutral' = 'failure';
      try {
        let res: ProviderResult;
        try {
          res = await provider.send(r);
        } catch (err) {
          // A throwing provider is just unavailable: advance the cascade.
          // Log only the failure code, never raw reason/error text.
          const failure = failureCode(err instanceof Error ? err.message : String(err));
          req.log.warn({ failure, conversation: decision.conversationKey }, `${tier} provider unavailable`);
          upstreamFailureState = { failure };
          return undefined;
        }
        if (res.kind === 'response') {
          outcome = 'success';
          return res;
        }
        const isRoutingMiss = res.reason === 'model asked for tools' || res.reason.startsWith('tool attempt:');
        // Over-budget is a property of the request, not provider health: it must not open the breaker.
        const overBudget = res.reason === 'transcript budget exceeded';
        if (res.reason === 'aborted' || isRoutingMiss || overBudget) outcome = 'neutral';
        req.log.warn({ failure: failureCode(res.reason), conversation: decision.conversationKey }, `${tier} provider unavailable`);
        upstreamFailureState = { failure: failureCode(res.reason) };
        const statusMatch = res.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return undefined;
      } finally {
        breaker.release(outcome);
      }
    };

    /**
     * Cascade steps after Cursor: Gemini, then cheap; whichever serves holds the fallback loop.
     * The reason names the tier that failed just before. Returns whether a reply was sent:
     * never return the relay itself, a FastifyReply is thenable and awaits to undefined.
     */
    const serveGeminiOrCheap = async (
      beforeGemini: RouteReason,
      sendCheap: () => Promise<ProviderResult>,
      onServed: () => void = () => {},
    ): Promise<boolean> => {
      if (!hasImageDocumentOrToolChoice(base.body!)) {
        const geminiRes = await tryCliTier('gemini', activeBase);
        if (geminiRes) {
          onServed();
          router.markFallbackLoop(decision, 'gemini');
          const routeDec: RouteDecision = { ...decision, route: 'gemini', tier: 'gemini', reason: beforeGemini, fallbackLoop: 'gemini' };
          relay(reply, geminiRes, routeDec, req, exchange('gemini'));
          return true;
        }
      }
      const cheapRes = await sendCheap();
      if (cheapRes.kind !== 'response') return false;
      onServed();
      // Hold the loop, never demote the session: the quota gate hands the next human turn back to Claude.
      router.markFallbackLoop(decision, 'trivial');
      // Gemini counts once the tier is wired in (attempted, or skipped by breaker/image guard).
      const reason: RouteReason = providers.gemini ? 'failover:gemini-unavailable' : beforeGemini;
      const routeDec: RouteDecision = { ...decision, route: 'cheap', tier: 'trivial', reason, fallbackLoop: 'trivial' };
      relay(reply, cheapRes, routeDec, req, exchange('cheap'));
      return true;
    };

    /**
     * Every Claude send carries the session's primary profile. A 400 on the rewritten body
     * resends the original; the session drops profiles only when that original is accepted
     * (a 400 on both is the request's own fault, e.g. prompt too long).
     */
    // served is the profile that went out; exchange() snapshots failure state, so build it at relay time.
    const sendClaude = async (b: ProviderRequest, dec: RouteDecision): Promise<{ result: ProviderResult; dec: RouteDecision; served?: PrimaryProfile }> => {
      const profile = dec.primaryProfile;
      const profiled = profile && b.rawBody ? withPrimaryProfile(b.rawBody, profile) : undefined;
      if (!profile || !profiled) return { result: await sendPrimary(b), dec };
      const first = await sendPrimary({ ...b, rawBody: profiled });
      if (first.kind !== 'response' || first.status !== 400) return { result: first, dec, served: profile };
      const error = Buffer.concat(await first.body.toArray()).toString('utf8').slice(0, 500);
      const result = await sendPrimary(b);
      if (result.kind === 'response' && result.status === 400) return { result, dec };
      req.log.warn({ profile, error }, 'primary profile refused, cleared');
      router.refusePrimaryProfile(dec);
      const refused: RouteDecision = { ...dec, reason: 'profile:refused' };
      return { result, dec: refused };
    };
    const relayClaude = async (b: ProviderRequest, dec: RouteDecision) => {
      const c = await sendClaude(b, dec);
      return relay(reply, c.result, c.dec, req, exchange('primary', c.served?.model, c.served?.effort));
    };

    if (decision.route === 'cursor') {
      const cursorRes = hasImageDocumentOrToolChoice(base.body!) ? undefined : await tryCliTier('cursor', activeBase);
      if (cursorRes) return relay(reply, cursorRes, decision, req, exchange('cursor'));

      // Cascade advance from a fallback-loop continuation: Cursor -> Gemini -> cheap -> primary (one-off, never pinned).
      if (await serveGeminiOrCheap('failover:cursor-unavailable', () => doSendCheapWithRetry(cheapProvider, cheapReq))) return reply;
      router.clearFallbackLoop(decision, 'primary');
      const routeDec: RouteDecision = { ...decision, route: 'primary', tier: 'primary', reason: 'failover:cheap-unavailable' };
      return relayClaude(base, routeDec);
    }

    if (decision.route === 'gemini') {
      const fallbackToPrimary = async (reason: RouteReason) => {
        router.clearGeminiLoop(decision, 'primary');
        const routeDec: RouteDecision = { ...decision, route: 'primary', reason };
        // base, not activeBase: the Gemini handoff in the system prompt would rewrite Claude's cache.
        return relayClaude(base, routeDec);
      };

      const doFallback = async (reason: RouteReason) => {
        if (decision.fallbackLoop) {
          // Cascade continuation: Gemini -> cheap -> primary one-off (never pinToPrimary from a cascade continuation).
          const resCheap = await doSendCheapWithRetry(cheapProvider, cheapReq);
          if (resCheap.kind === 'response') {
            router.markFallbackLoop(decision, 'trivial');
            const routeDec: RouteDecision = { ...decision, route: 'cheap', tier: 'trivial', reason, fallbackLoop: 'trivial' };
            return relay(reply, resCheap, routeDec, req, exchange('cheap'));
          }
          router.clearFallbackLoop(decision, 'primary');
          const routeDec: RouteDecision = { ...decision, route: 'primary', tier: 'primary', reason: 'failover:cheap-unavailable' };
          return relayClaude(base, routeDec);
        }

        if (decision.fallbackTier !== 'trivial' && decision.fallbackTier !== 'standard') {
          return fallbackToPrimary(reason);
        }

        if (cheapHealth?.state === 'down') {
          return fallbackToPrimary(reason);
        }

        router.clearGeminiLoop(decision, decision.fallbackTier);
        const fbProvider = (decision.fallbackTier === 'standard' && providers.standard) ? providers.standard : providers.cheap;
        const resFb = await doSendCheapWithRetry(fbProvider, { ...cheapReq, onStreamFailure: () => router.pinToPrimary(decision) });
        if (resFb.kind === 'response') {
          const routeDec: RouteDecision = { ...decision, route: 'cheap', reason };
          return relay(reply, resFb, routeDec, req, exchange('cheap'));
        }

        router.pinToPrimary(decision);
        return fallbackToPrimary('failover:cheap-unavailable');
      };

      if (!providers.gemini || !geminiBreaker || geminiBreaker.state === 'open') {
        upstreamFailureState = { failure: 'skipped' };
        return doFallback('skipped:gemini-unhealthy');
      }

      if (!geminiBreaker.acquire()) {
        upstreamFailureState = { failure: 'skipped' };
        return doFallback('skipped:gemini-busy');
      }

      let outcome: 'success' | 'failure' | 'neutral' = 'failure';
      try {
        const res = await providers.gemini.send(activeBase);
        if (res.kind === 'response') {
          outcome = 'success';
          return relay(reply, res, decision, req, exchange('gemini'));
        }

        const isRoutingMiss = res.reason === 'model asked for tools' || res.reason.startsWith('tool attempt:');
        if (res.reason === 'aborted' || isRoutingMiss) {
          outcome = 'neutral';
        }

        req.log.warn({ reason: res.reason, conversation: decision.conversationKey }, 'gemini provider unavailable');
        upstreamFailureState = { failure: failureCode(res.reason) };
        const statusMatch = res.reason.match(/^HTTP (\d+)/);
        if (statusMatch && statusMatch[1]) upstreamFailureState.status = parseInt(statusMatch[1], 10);
        return await doFallback('failover:gemini-unavailable');
      } finally {
        geminiBreaker.release(outcome);
      }
    }

    // A provider known to be down costs a failed attempt per turn: skip it, and do not
    // pin the conversation, so it returns to cheap once the provider is back.
    if (decision.route === 'cheap' && cheapHealth?.state === 'down') {
      const skipped: RouteDecision = { ...decision, route: 'primary', reason: 'skipped:cheap-unhealthy' };
      upstreamFailureState = { failure: 'skipped' };
      return relayClaude(base, skipped);
    }

    if (decision.route === 'cheap') {
      const cheap = await doSendCheapWithRetry(cheapProvider, { ...cheapReq, onStreamFailure: () => router.pinToPrimary(decision) });
      if (cheap.kind === 'response') return relay(reply, cheap, decision, req, exchange('cheap'));

      if (decision.reason === 'read-first:standard' && providers.cheap) {
        const trivial = await doSendCheapWithRetry(providers.cheap, { ...cheapReq, onStreamFailure: () => router.pinToPrimary(decision) });
        if (trivial.kind === 'response') {
          router.useTrivial(decision);
          req.log.info({ conversation: decision.conversationKey }, 'standard tier unavailable, read-first turn served by trivial tier');
          const trivialDecision: RouteDecision = { ...decision, tier: 'trivial' as Tier, reason: 'failover:standard-unavailable' };
          return relay(reply, trivial, trivialDecision, req, exchange('cheap', config.cheap.model));
        }
      }

      router.pinToPrimary(decision);
      const failover: RouteDecision = { ...decision, route: 'primary', reason: 'failover:cheap-unavailable' };
      req.log.warn({ reason: cheap.reason, conversation: decision.conversationKey }, 'cheap provider unavailable');
      return relayClaude(base, failover);
    }

    const claude = await sendClaude(activeBase, decision);
    const primary = claude.result;
    const primaryDecision = claude.dec;
    if (
      primary.kind === 'unavailable' ||
      !config.router.failoverOnPrimaryRateLimit ||
      !PRIMARY_RATE_LIMITED.has(primary.status)
    ) {
      return relay(reply, primary, primaryDecision, req, exchange('primary', claude.served?.model, claude.served?.effort));
    }

    if (isQuotaProbe(base.body)) {
      const quotaHeaderNames = rateLimitHeaderNames(primary.headers);
      req.log.info({ status: primary.status, quotaHeaderNames }, 'primary rate-limited probe passed through');
      upstreamFailureState = { status: primary.status, rateLimitHeaders: quotaHeaderNames.join(',') };
      return relay(reply, primary, primaryDecision, req, exchange('primary', claude.served?.model, claude.served?.effort));
    }

    const quotaHeaderNames = rateLimitHeaderNames(primary.headers);
    req.log.warn({ status: primary.status, quotaHeaderNames, conversation: decision.conversationKey }, 'primary rate-limited, failing over (cascade Cursor -> Gemini -> cheap)');

    // Cascade failover: hold the (small) error body so it can still be relayed
    // verbatim if no cascade tier can take the request either. One-shot: these
    // attempts never re-enter this tail.
    const errorBody = Buffer.concat(await primary.body.toArray());
    const skipCli = hasImageDocumentOrToolChoice(base.body!);
    const primaryFailure = { status: primary.status, rateLimitHeaders: quotaHeaderNames.join(',') };

    // A hard risk veto wins over the cascade: a risky turn must not be
    // served by Cursor/Gemini/cheap just because Claude rate-limited.
    // Relay the original Anthropic error verbatim (all-tiers-down path).
    // Unclassified turns (sticky primary, tool continuations) are classified
    // here, on their last human text, before they can leave Claude.
    const vetoed = decision.distribution ? hardRiskVeto(decision.distribution) : await router.cascadeRiskVeto(base.body!);
    if (vetoed) {
      upstreamFailureState = primaryFailure;
      const original: ProviderResult = { ...primary, body: Readable.from([errorBody]) };
      return relay(reply, original, decision, req, exchange('primary'));
    }

    if (!skipCli) {
      const cursorRes = await tryCliTier('cursor', activeBase);
      if (cursorRes) {
        router.markFallbackLoop(decision, 'cursor');
        upstreamFailureState = primaryFailure;
        return relay(
          reply,
          cursorRes,
          { ...decision, route: 'cursor', tier: 'cursor', reason: 'failover:primary-rate-limited', fallbackLoop: 'cursor' },
          req,
          exchange('cursor'),
        );
      }
    }

    // Without a Cursor tier, Gemini/cheap keep the pre-cascade reason.
    const served = await serveGeminiOrCheap(
      providers.cursor ? 'failover:cursor-unavailable' : 'failover:primary-rate-limited',
      () => providers.cheap.send(cheapReq),
      () => { upstreamFailureState = primaryFailure; },
    );
    if (served) return reply;
    const original: ProviderResult = { ...primary, body: Readable.from([errorBody]) };
    return relay(reply, original, decision, req, exchange('primary'));
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
      // The 502 body reaches the client only; without this the cause is lost (2026-10-10: 79 bare 502s).
      reply.log.warn({ route: decision?.route, reason: result.reason }, 'upstream unavailable');
      if (audited) auditFailure(telemetry, decision, exchange, result.reason);
      return reply.code(502).send(anthropicError('api_error', `jev-router: ${result.reason}`));
    }
    if (decision) {
      req?.log.info({ route: decision }, 'route decision');
      reply.header('x-jev-route', `${decision.route}; reason=${decision.reason}`);
      const key = decision.conversationKey;
      if (key?.endsWith(':main')) {
        lastRoutes.delete(key); // re-insert at the end: the Map's order is then oldest-first
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
    if (reply.raw.destroyed) return; // client went away mid-request
    // Fastify's own 4xx (body too large, bad content type) are the client's problem, not the upstream's.
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
