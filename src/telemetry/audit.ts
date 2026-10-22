import { createHash } from 'node:crypto';
import type { Readable } from 'node:stream';
import type { ServerResponse } from 'node:http';
import type { RouteDecision } from '../routing/router.js';
import type { TelemetrySink } from './recorder.js';
import type { JevDecision, Outcome } from './schema.js';
import { meterAnthropicBody, type MeteredBody } from './usage-meter.js';

/** What the server knows about an exchange before the upstream answers. */
export interface ExchangeContext {
  readonly startedAt: number;
  readonly startedAtMs: number;
  readonly humanText: string | undefined;
  readonly requestClass: string | undefined;
  readonly model: string | undefined;
  readonly requestedModel: string | undefined;
  /** Size of `body.tools`: a cheap answer that used none of them is an inspection miss. */
  readonly toolsOffered: number;
  /** Binding Claude quota window when the exchange was routed. */
  readonly quotaUtilization?: number | undefined;
  readonly diagnostics?: {
    ccSessionId?: string | undefined;
    agentType?: string | undefined;
    promptId?: string | undefined;
    compaction?: string | undefined;
    systemHash?: string | undefined;
    toolsHash?: string | undefined;
    systemChars?: number | undefined;
    toolsChars?: number | undefined;
    maxTokens?: number | undefined;
    thinkingType?: string | undefined;
    thinkingBudget?: number | undefined;
    effort?: string | undefined;
  } | undefined;
  readonly upstream?: {
    status?: number;
    rateLimitHeaders?: string;
    failure?: string;
  } | undefined;
}

export interface UpstreamOutcome {
  readonly status: number;
  readonly headers: Readonly<Record<string, string | string[]>>;
  readonly body: Readable;
}

/** Upper bound on waiting for a compressed body's usage to finish decoding. */
const SETTLE_TIMEOUT_MS = 5_000;

const header = (h: UpstreamOutcome['headers'], name: string): string | undefined => {
  const v = h[name];
  return Array.isArray(v) ? v[0] : v;
};

export function promptHash(text: string | undefined): string | null {
  return text === undefined ? null : createHash('sha256').update(text).digest('hex');
}

export function isAcknowledgement(text: string): boolean {
  if (!text) return false;
  if (text.includes('?')) return false;
  const words = text.trim().split(/\s+/);
  if (words.length > 12 || words.length === 0 || words[0] === '') return false;

  if (text.includes('/') || text.includes('\\')) return false;
  for (const w of words) {
    if (/\.[a-zA-Z]{1,4}(?:[^a-zA-Z]|$)/.test(w)) return false;
  }

  const normalized = text.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
  const tokens = normalized.split(/[^a-z0-9]+/).filter(Boolean);

  const containsPhrase = (phrase: string) => {
    const pTokens = phrase.split(' ');
    for (let i = 0; i <= tokens.length - pTokens.length; i++) {
      let match = true;
      for (let j = 0; j < pTokens.length; j++) {
        if (tokens[i + j] !== pTokens[j]) {
          match = false;
          break;
        }
      }
      if (match) return true;
    }
    return false;
  };

  const requestCues = [
    'now', 'agora', 'please', 'por favor', 'can you', 'could you', 'fix', 'add',
    'create', 'run', 'read', 'write', 'change', 'implement', 'update', 'remove',
    'delete', 'make', 'check', 'review', 'explain', 'show me', 'faz', 'faca',
    'corrige', 'corrija', 'cria', 'crie', 'roda', 'rode', 'leia', 'escreva',
    'mude', 'implemente', 'atualize', 'remova', 'verifique', 'revise', 'explique',
    'mostre', 'vamos', 'preciso', 'quero'
  ];

  for (const cue of requestCues) {
    if (containsPhrase(cue)) return false;
  }

  const approvalWords = [
    'ok', 'okay', 'thanks', 'thank you', 'valeu', 'obrigado', 'obrigada',
    'top', 'otimo', 'perfeito', 'great', 'nice', 'legal', 'show', 'beleza',
    'massa', 'boa'
  ];

  for (const word of approvalWords) {
    if (containsPhrase(word)) return true;
  }

  return false;
}

export function jevDecisionOf(decision: RouteDecision): JevDecision | null {
  const d = decision.distribution ?? decision.profileScore;
  if (!d) return null;
  return {
    simple: d.simple,
    standard: d.standard,
    structural: d.structural,
    pSimple: d.simple,
    pComplex: d.structural,
    classifierMs: decision.classifierMs ?? null,
    tokensIn: d.usage?.inputTokens ?? null,
    tokensOut: d.usage?.outputTokens ?? null,
    risk: d.risk ?? null,
    model: d.model ?? null,
    riskScores: d.riskScores ?? null,
    textOnly: d.textOnly ?? null,
  };
}

/**
 * Wraps the upstream body so its usage can be read, and records one row once
 * the client connection closes. Recording is deferred off the response path:
 * nothing here can delay or fail the bytes the client receives.
 */
export function auditExchange(
  sink: TelemetrySink,
  res: ServerResponse,
  upstream: UpstreamOutcome,
  decision: RouteDecision,
  ctx: ExchangeContext,
  onError: (err: unknown) => void,
): Readable {
  let metered: MeteredBody;
  try {
    metered = meterAnthropicBody(
      upstream.body,
      header(upstream.headers, 'content-type') ?? '',
      header(upstream.headers, 'content-encoding'),
    );
  } catch (err) {
    onError(err);
    return upstream.body;
  }

  res.once('close', () => {
    const latencyMs = Math.round(performance.now() - ctx.startedAt);
    const delivered = res.writableFinished;
    const timeout = new Promise<void>((resolve) => setTimeout(resolve, SETTLE_TIMEOUT_MS).unref());
    void Promise.race([metered.settled, timeout]).then(() => {
      try {
        const outcome: Outcome = !delivered
          ? 'client_abort'
          : upstream.status >= 400
            ? 'http_error'
            : metered.sawErrorEvent()
              ? 'stream_error'
              : 'ok';
        record(sink, decision, ctx, {
          status: upstream.status,
          outcome,
          latencyMs,
          usage: metered.usage(),
          toolCalls: metered.toolCalls(),
          endsWithQuestion: metered.endsWithQuestion(),
        });
      } catch (err) {
        onError(err);
      }
    });
  });
  return metered.body;
}

/** Longest upstream failure reason kept: an HTTP error can carry a whole response body. */
const MAX_FAILURE_CHARS = 200;

/**
 * For exchanges that never got an upstream body (jev-router answered 502 itself).
 * `reason` is why the provider was unavailable. On the cheap route the retry path's
 * own record ("retried: network") wins, being the more precise; on any other route
 * that record belongs to an earlier, failed-over attempt and would misattribute it.
 */
export function auditFailure(sink: TelemetrySink, decision: RouteDecision, ctx: ExchangeContext, reason: string): void {
  const failure = (decision.route === 'cheap' ? ctx.upstream?.failure : undefined) ?? reason.slice(0, MAX_FAILURE_CHARS);
  record(sink, decision, { ...ctx, upstream: { ...ctx.upstream, failure } }, {
    status: 502,
    outcome: 'proxy_error',
    latencyMs: Math.round(performance.now() - ctx.startedAt),
    usage: undefined,
    toolCalls: null,
    endsWithQuestion: false,
  });
}

function record(
  sink: TelemetrySink,
  decision: RouteDecision,
  ctx: ExchangeContext,
  result: {
    status: number;
    outcome: Outcome;
    latencyMs: number;
    usage: ReturnType<MeteredBody['usage']>;
    toolCalls: number | null;
    endsWithQuestion: boolean;
  },
): void {
  const cheap = decision.route === 'cheap';
  sink.record({
    createdAt: new Date(ctx.startedAtMs),
    sessionId: decision.conversationKey ?? null,
    humanPromptHash: promptHash(ctx.humanText),
    jevDecision: jevDecisionOf(decision),
    finalProvider: decision.route === 'gemini' ? 'gemini' : decision.route === 'cursor' ? 'cursor' : cheap ? 'openai' : 'anthropic',
    model: ctx.model ?? null,
    requestedModel: ctx.requestedModel ?? null,
    routeReason: decision.reason,
    requestClass: ctx.requestClass ?? null,
    httpStatus: result.status,
    outcome: result.outcome,
    tokensIn: result.usage?.tokensIn ?? null,
    tokensOut: result.usage?.tokensOut ?? null,
    cacheReadTokens: result.usage?.cacheReadTokens ?? null,
    cacheWriteTokens: result.usage?.cacheWriteTokens ?? null,
    latencyMs: result.latencyMs,
    fallbackTriggered: decision.reason.startsWith('failover:'),
    toolsOffered: ctx.toolsOffered,
    toolCalls: result.toolCalls,
    inspectionMiss: cheap ? result.outcome === 'ok' && ctx.toolsOffered > 0 && result.toolCalls === 0 && ctx.humanText !== undefined && !isAcknowledgement(ctx.humanText) && !result.endsWithQuestion : null,
    quotaUtilization: ctx.quotaUtilization ?? null,
    ccSessionId: ctx.diagnostics?.ccSessionId ?? null,
    agentType: ctx.diagnostics?.agentType ?? null,
    promptId: ctx.diagnostics?.promptId ?? null,
    compaction: ctx.diagnostics?.compaction ?? null,
    systemHash: ctx.diagnostics?.systemHash ?? null,
    toolsHash: ctx.diagnostics?.toolsHash ?? null,
    systemChars: ctx.diagnostics?.systemChars ?? null,
    toolsChars: ctx.diagnostics?.toolsChars ?? null,
    maxTokens: ctx.diagnostics?.maxTokens ?? null,
    thinkingType: ctx.diagnostics?.thinkingType ?? null,
    thinkingBudget: ctx.diagnostics?.thinkingBudget ?? null,
    effort: ctx.diagnostics?.effort ?? null,
    cacheWrite5mTokens: result.usage?.cacheWrite5mTokens ?? null,
    cacheWrite1hTokens: result.usage?.cacheWrite1hTokens ?? null,
    upstreamStatus: ctx.upstream?.status ?? null,
    upstreamRatelimitHeaders: ctx.upstream?.rateLimitHeaders ?? null,
    upstreamFailure: ctx.upstream?.failure ?? null,
  });
}
