import { computeNetCost, type CostRow, type NetCost } from './cost.js';
import type { TelemetryDb } from './db.js';
import { pricingFromEnv, type Pricing } from './pricing.js';
import type { FinalProvider } from './schema.js';
import { latestQuota } from './statusline.js';

export interface ProviderStats {
  readonly requests: number;
  readonly ok: number;
  readonly tokensIn: number;
  readonly tokensOut: number;
  readonly cacheReadTokens: number;
  readonly avgLatencyMs: number | null;
}

export interface RouterStats {
  readonly since: Date | null;
  readonly firstAt: Date | null;
  readonly lastAt: Date | null;
  readonly total: number;
  readonly byProvider: Readonly<Record<FinalProvider, ProviderStats>>;
  /** Cheap-route requests per model: the trivial tier (20B) and the standard tier (Gemma). */
  readonly cheapByModel: readonly { readonly model: string; readonly requests: number; readonly ok: number }[];
  /** Requests JEV actually scored (the rest were sticky, auxiliary or classifier failures). */
  readonly classified: number;
  /** Cheap provider failed before answering; the request went to Anthropic. */
  readonly fallbacks: number;
  /** Cheap provider broke mid-stream; Claude Code retries, the conversation is pinned to Anthropic. */
  readonly cheapStreamErrors: number;
  /** Cheap answers, delivered ok, that called none of the tools offered: likely made up without looking. */
  readonly inspectionMisses: number;
  /** Turns that left Claude because of the quota: quota:* reasons and the 429/529 failover. */
  readonly quotaRouted: number;
  readonly probeRateLimited: number;
  /** Latest persisted Claude quota reading (any age). */
  readonly lastQuota: { readonly utilization: number; readonly window: string; readonly createdAt: Date } | null;
  /** Human prompts sent more than once (same sha256), and the extra sends. */
  readonly repeatedPrompts: number;
  readonly repeatedSends: number;
  /**
   * Estimate of Anthropic tokens the System One layer kept off the primary
   * quota: in + out tokens of requests the cheap provider served successfully.
   */
  readonly estimatedTokensSaved: number;
  /** Dollar view: actual spend versus the all-Anthropic baseline, cache-miss penalty included. */
  readonly cost: NetCost;
  readonly cheapPriceUnset: boolean;
  readonly jevPriceUnset: boolean;
  readonly geminiFallbacks: number;
  readonly geminiSkipped: number;
  readonly cursorFallbacks: number;
}

export interface StatsOptions {
  readonly since?: Date;
  readonly pricing?: Pricing;
}

const EMPTY: ProviderStats = { requests: 0, ok: 0, tokensIn: 0, tokensOut: 0, cacheReadTokens: 0, avgLatencyMs: null };

export function computeStats(db: TelemetryDb, options: StatsOptions = {}): RouterStats {
  const { since } = options;
  const pricing = options.pricing ?? pricingFromEnv({});
  const params = { since: since?.getTime() ?? null };
  const inWindow = 'WHERE (@since IS NULL OR created_at >= @since)';
  const all = <T>(sql: string) => db.prepare(sql).all(params) as T[];
  const get = <T>(sql: string) => db.prepare(sql).get(params) as T;

  const rows = all<{ provider: FinalProvider } & ProviderStats>(`
    SELECT final_provider AS provider, count(*) AS requests,
      sum(CASE WHEN outcome = 'ok' THEN 1 ELSE 0 END) AS ok,
      coalesce(sum(tokens_in), 0) AS tokensIn, coalesce(sum(tokens_out), 0) AS tokensOut,
      coalesce(sum(cache_read_tokens), 0) AS cacheReadTokens, avg(latency_ms) AS avgLatencyMs
    FROM router_logs ${inWindow} GROUP BY final_provider`);

  const byProvider: Record<FinalProvider, ProviderStats> = { anthropic: EMPTY, openai: EMPTY, gemini: EMPTY, cursor: EMPTY };
  for (const r of rows) {
    byProvider[r.provider] = {
      requests: r.requests,
      ok: r.ok,
      tokensIn: r.tokensIn,
      tokensOut: r.tokensOut,
      cacheReadTokens: r.cacheReadTokens,
      avgLatencyMs: r.avgLatencyMs === null ? null : Math.round(r.avgLatencyMs),
    };
  }

  const summary = get<{
    total: number;
    firstAt: number | null;
    lastAt: number | null;
    classified: number;
    fallbacks: number;
    saved: number;
    geminiFallbacks: number;
    geminiSkipped: number;
    cursorFallbacks: number;
  }>(`
    SELECT count(*) AS total, min(created_at) AS firstAt, max(created_at) AS lastAt,
      coalesce(sum(CASE WHEN jev_decision IS NOT NULL THEN 1 ELSE 0 END), 0) AS classified,
      coalesce(sum(CASE WHEN fallback_triggered = 1 AND route_reason != 'failover:gemini-unavailable' THEN 1 ELSE 0 END), 0) AS fallbacks,
      coalesce(sum(CASE WHEN final_provider IN ('openai', 'gemini', 'cursor') AND outcome = 'ok'
        THEN coalesce(tokens_in, 0) + coalesce(tokens_out, 0) ELSE 0 END), 0) AS saved,
      coalesce(sum(CASE WHEN route_reason = 'failover:gemini-unavailable' THEN 1 ELSE 0 END), 0) AS geminiFallbacks,
      coalesce(sum(CASE WHEN route_reason IN ('skipped:gemini-unhealthy', 'skipped:gemini-busy') THEN 1 ELSE 0 END), 0) AS geminiSkipped,
      coalesce(sum(CASE WHEN route_reason = 'failover:cursor-unavailable' THEN 1 ELSE 0 END), 0) AS cursorFallbacks
    FROM router_logs ${inWindow}`);

  const cheapStreamErrors = get<{ n: number }>(
    `SELECT count(*) AS n FROM router_logs ${inWindow} AND final_provider = 'openai' AND outcome = 'stream_error'`,
  ).n;

  const cheapByModel = all<{ model: string; requests: number; ok: number }>(`
    SELECT coalesce(model, '?') AS model, count(*) AS requests, sum(CASE WHEN outcome = 'ok' THEN 1 ELSE 0 END) AS ok
    FROM router_logs ${inWindow} AND final_provider = 'openai' GROUP BY 1 ORDER BY requests DESC, model`).map((r) => ({
    ...r,
  }));

  const quotaRouted = get<{ n: number }>(
    `SELECT count(*) AS n FROM router_logs ${inWindow}
     AND (route_reason LIKE 'quota:%' OR (route_reason = 'failover:primary-rate-limited' AND coalesce(max_tokens, 2) > 1))`,
  ).n;

  const probeRateLimited = get<{ n: number }>(
    `SELECT count(*) AS n FROM router_logs ${inWindow}
     AND ((coalesce(max_tokens, 2) <= 1 AND http_status IN (429, 529)) OR (route_reason = 'failover:primary-rate-limited' AND coalesce(max_tokens, 2) <= 1))`,
  ).n;
  const lastQuota = latestQuota(db, { maxAgeMs: Infinity }) ?? null;

  const inspectionMisses = get<{ n: number }>(
    `SELECT coalesce(sum(inspection_miss), 0) AS n FROM router_logs ${inWindow} AND human_prompt_hash IS NOT NULL`,
  ).n;

  const repeats = all<{ sends: number }>(`
    SELECT count(*) AS sends FROM router_logs ${inWindow} AND human_prompt_hash IS NOT NULL
    GROUP BY human_prompt_hash HAVING count(*) > 1`);

  // With --since, a session whose earlier turns fall outside the window starts
  // fresh: a cheap → Anthropic transition across the boundary is not penalized.
  const costRows = all<CostRow>(`
    SELECT id, session_id AS sessionId, final_provider AS finalProvider, model, requested_model AS requestedModel,
      outcome, tokens_in AS tokensIn, tokens_out AS tokensOut, cache_read_tokens AS cacheReadTokens,
      cache_write_tokens AS cacheWriteTokens,
      json_extract(jev_decision, '$.tokensIn') AS jevTokensIn, json_extract(jev_decision, '$.tokensOut') AS jevTokensOut,
      cache_write_5m_tokens AS cacheWrite5mTokens, cache_write_1h_tokens AS cacheWrite1hTokens
    FROM router_logs ${inWindow} ORDER BY id`);

  return {
    since: since ?? null,
    firstAt: summary.firstAt === null ? null : new Date(summary.firstAt),
    lastAt: summary.lastAt === null ? null : new Date(summary.lastAt),
    total: summary.total,
    byProvider,
    cheapByModel,
    classified: summary.classified,
    fallbacks: summary.fallbacks,
    cheapStreamErrors,
    inspectionMisses,
    quotaRouted,
    probeRateLimited,
    lastQuota,
    repeatedPrompts: repeats.length,
    repeatedSends: repeats.reduce((n, r) => n + r.sends - 1, 0),
    estimatedTokensSaved: summary.saved,
    cost: computeNetCost(costRows, pricing),
    cheapPriceUnset: pricing.cheapPriceUnset,
    jevPriceUnset: pricing.jevPriceUnset,
    geminiFallbacks: summary.geminiFallbacks,
    geminiSkipped: summary.geminiSkipped,
    cursorFallbacks: summary.cursorFallbacks,
  };
}

/** `24h`, `7d`, `30m` → the instant that far back from `now`. */
export function parseSince(span: string, now = Date.now()): Date {
  const m = /^(\d+)\s*([mhd])$/.exec(span.trim());
  if (!m) throw new Error(`invalid --since "${span}": use <n>m, <n>h or <n>d`);
  const unit = { m: 60_000, h: 3_600_000, d: 86_400_000 }[m[2] as 'm' | 'h' | 'd'];
  return new Date(now - Number(m[1]) * unit);
}

const fmt = (n: number) => n.toLocaleString('en-US');
const pct = (part: number, whole: number) => (whole === 0 ? '—' : `${((part / whole) * 100).toFixed(1)}%`);

function table(rows: readonly (readonly string[])[]): string {
  const widths = rows[0]!.map((_, c) => Math.max(...rows.map((r) => r[c]!.length)));
  const line = (r: readonly string[]) =>
    r.map((cell, c) => (c === 0 ? cell.padEnd(widths[c]!) : cell.padStart(widths[c]!))).join('  ');
  const rule = widths.map((w) => '─'.repeat(w)).join('  ');
  return [line(rows[0]!), rule, ...rows.slice(1).map(line)].join('\n');
}

export function renderStats(s: RouterStats): string {
  const a = s.byProvider.anthropic;
  const o = s.byProvider.openai;
  const g = s.byProvider.gemini;
  const cu = s.byProvider.cursor;
  const window = s.since ? `since ${s.since.toISOString()}` : 'all time';
  const span = s.firstAt && s.lastAt ? `${s.firstAt.toISOString()} → ${s.lastAt.toISOString()}` : 'no data';

  const overview = table([
    ['jev-router stats', window],
    ['Requests', fmt(s.total)],
    ['Diverted from Anthropic (cheap route)', `${fmt(o.requests)} (${pct(o.requests, s.total)})`],
    ['  served OK by cheap provider', fmt(o.ok)],
    ['  cheap stream errors (retried on Anthropic)', fmt(s.cheapStreamErrors)],
    ['Diverted to Gemini (subscription)', `${fmt(g.requests)} (${pct(g.requests, s.total)})`],
    ['  served OK by Gemini', fmt(g.ok)],
    ['  Gemini skipped (breaker/busy)', fmt(s.geminiSkipped)],
    ['Diverted to Cursor (subscription)', `${fmt(cu.requests)} (${pct(cu.requests, s.total)})`],
    ['  served OK by Cursor', fmt(cu.ok)],
    ['Cheap answers without tool use (inspection miss)', fmt(s.inspectionMisses)],
    ['Sent cheap by quota pressure', fmt(s.quotaRouted)],
    ['Rate-limited session probes', fmt(s.probeRateLimited)],
    [
      'Claude quota (last seen)',
      s.lastQuota ? `${Math.round(s.lastQuota.utilization * 100)}% ${s.lastQuota.window}` : '—',
    ],
    ['Fallbacks cheap → Anthropic', fmt(s.fallbacks)],
    ['Fallbacks Gemini → Claude', fmt(s.geminiFallbacks)],
    ['Fallbacks Cursor → next tier', fmt(s.cursorFallbacks)],
    ['Scored by JEV', fmt(s.classified)],
    ['Repeated prompts (extra sends)', `${fmt(s.repeatedPrompts)} (${fmt(s.repeatedSends)})`],
    ['Estimated Anthropic tokens saved', fmt(s.estimatedTokensSaved)],
  ]);

  const providers = table([
    ['Provider', 'Requests', 'OK', 'Tokens in', 'Tokens out', 'Cache read', 'Avg latency'],
    ...(
      [
        ['anthropic', a],
        ['openai (cheap)', o],
        ['gemini (subscription)', g],
        ['cursor (subscription)', cu],
      ] as const
    ).map(([name, p]) => [
      name,
      fmt(p.requests),
      fmt(p.ok),
      fmt(p.tokensIn),
      fmt(p.tokensOut),
      fmt(p.cacheReadTokens),
      p.avgLatencyMs === null ? '—' : `${fmt(p.avgLatencyMs)} ms`,
    ]),
  ]);

  const models =
    s.cheapByModel.length > 0
      ? table([['Cheap model', 'Requests', 'OK'], ...s.cheapByModel.map((m) => [m.model, fmt(m.requests), fmt(m.ok)])])
      : undefined;

  const c = s.cost;
  const money = (n: number) => `${n < 0 ? '-' : ''}$${Math.abs(n).toFixed(4)}`;
  const verdict = c.netUsd >= 0 ? 'PROFIT' : 'LOSS';
  const costs = table([
    ['Net cost (USD, API list prices)', ''],
    ['All-Anthropic baseline (warm cache)', money(c.baselineUsd)],
    ['Actual: Anthropic', money(c.anthropicUsd)],
    ['Actual: cheap provider', money(c.cheapUsd)],
    ['Gross savings (cheap requests served OK)', money(c.grossSavingsUsd)],
    [`Cache-miss penalty (${fmt(c.transitions)} cheap → Anthropic returns)`, money(-c.cachePenaltyUsd)],
    ['Failed cheap attempts (paid, then redone)', money(-c.failedCheapUsd)],
    ['Classifier (JEV) calls', money(-c.jevUsd)],
    [`NET (${verdict})`, money(c.netUsd)],
  ]);
  const warnings = [
    ...(s.cheapPriceUnset
      ? ['! Cheap provider priced at $0: set CHEAP_PRICE_INPUT_PER_MTOK / CHEAP_PRICE_OUTPUT_PER_MTOK (leave unset only for local models).']
      : []),
    ...(s.jevPriceUnset && s.classified > 0
      ? ['! JEV priced at $0: set JEV_PRICE_INPUT_PER_MTOK / JEV_PRICE_OUTPUT_PER_MTOK from your TypeSafe plan.']
      : []),
    ...(c.unpricedModels.length > 0 ? [`! No list price for ${c.unpricedModels.join(', ')}: priced at $0, set PRIMARY_PRICE_*.`] : []),
    ...(c.rowsWithoutUsage > 0 ? [`! ${fmt(c.rowsWithoutUsage)} requests carried no token usage and count as $0.`] : []),
  ];

  return [
    overview,
    '',
    providers,
    ...(models ? ['', models] : []),
    '',
    costs,
    ...(warnings.length > 0 ? ['', ...warnings] : []),
    '',
    `Window: ${span}`,
    'Tokens saved = tokens in + out of requests the cheap provider served OK. Estimate:',
    "the cheap model's tokenizer differs from Claude's, and on Anthropic part of that",
    'input would have been a cache read. Rows without usage count as 0.',
    'Dollars are API-equivalent: on a claude.ai subscription they measure quota, not a bill.',
  ].join('\n');
}


export function computeStatsByClass(db: TelemetryDb, pricing: Pricing, options: StatsOptions = {}) {
  const { since } = options;
  const params = { since: since?.getTime() ?? null };
  const inWindow = 'WHERE (@since IS NULL OR created_at >= @since)';
  const rows = db.prepare(
    'SELECT final_provider AS finalProvider, coalesce(request_class, \'?\') AS requestClass, coalesce(requested_model, \'?\') AS requestedModel, ' +
    '  count(*) AS requests, sum(coalesce(tokens_in, 0)) AS tokensIn, sum(coalesce(cache_read_tokens, 0)) AS cacheRead, ' +
    '  sum(coalesce(cache_write_tokens, 0)) AS cacheWrite, sum(coalesce(tokens_out, 0)) AS tokensOut, ' +
    '  sum(cache_write_5m_tokens) AS cacheWrite5m, sum(cache_write_1h_tokens) AS cacheWrite1h ' +
    'FROM router_logs ' + inWindow + ' ' +
    'GROUP BY final_provider, request_class, requested_model ' +
    'ORDER BY final_provider, request_class, requested_model'
  ).all(params) as any[];

  return rows.map(r => {
    let usd = 0;
    if (r.finalProvider === 'anthropic') {
      const p = pricing.primary(r.requestedModel !== '?' ? r.requestedModel : null);
      if (p) {
        let writeCost = r.cacheWrite * p.input * pricing.cacheWriteMultiplier;
        if (r.cacheWrite5m != null || r.cacheWrite1h != null) {
           writeCost = ((r.cacheWrite5m ?? 0) * 1.25 + (r.cacheWrite1h ?? 0) * 2) * p.input;
        }
        usd = (Math.max(0, r.tokensIn - r.cacheRead - r.cacheWrite) * p.input + writeCost + r.cacheRead * p.cacheRead + r.tokensOut * p.output) / 1000000;
      }
    }
    return { ...r, usd };
  });
}

export function renderStatsByClass(stats: any[]) {
  if (stats.length === 0) return 'No data';
  return table([
    ['Provider', 'Class', 'Requested model', 'Requests', 'Tokens in', 'Cache read', 'Cache write', 'Tokens out', 'USD proxy'],
    ...stats.map(s => [s.finalProvider, s.requestClass, s.requestedModel, fmt(s.requests), fmt(s.tokensIn), fmt(s.cacheRead), fmt(s.cacheWrite), fmt(s.tokensOut), '$' + s.usd.toFixed(4)])
  ]);
}

export function computeCacheMisses(db: TelemetryDb, pricing: Pricing, options: StatsOptions & { minWrite?: number }) {
  const { since } = options;
  const minWrite = options.minWrite ?? 150000;
  const params = { since: since?.getTime() ?? null, minWrite };
  
  const oversized = db.prepare(
    'SELECT id, tokens_in as tokensIn, cache_read_tokens as cacheRead, cache_write_tokens as cacheWrite ' +
    'FROM router_logs WHERE (@since IS NULL OR created_at >= @since) AND tokens_in > 1000000'
  ).all({ since: params.since }) as any[];
  
  const rows = db.prepare(
    'SELECT id, created_at as createdAt, coalesce(cc_session_id, session_id) as session, request_class as requestClass, requested_model as requestedModel, ' +
    '       tokens_in as tokensIn, cache_write_tokens as cacheWrite, cache_write_5m_tokens as cacheWrite5m, cache_write_1h_tokens as cacheWrite1h, ' +
    '       cc_session_id as ccSessionId, session_id as sessionId, compaction, system_hash as systemHash, tools_hash as toolsHash, ' +
    '       final_provider as finalProvider, route_reason as routeReason, max_tokens as maxTokens ' +
    'FROM router_logs ' +
    'WHERE (@since IS NULL OR created_at >= @since) AND coalesce(max_tokens, 2) > 1 ' +
    'ORDER BY created_at, id'
  ).all({ since: params.since }) as any[];

  const lastAny = new Map<string, any>();
  const lastAnthropic = new Map<string, any>();
  const misses = [];
  const summary: Record<string, { count: number, tokens: number }> = {};
  
  for (const row of rows) {
    const sKey = row.session + '|' + row.requestClass;
    const prevAny = row.session ? lastAny.get(sKey) : undefined;
    const prevAnthropic = row.session ? lastAnthropic.get(sKey) : undefined;
    
    if (row.finalProvider === 'anthropic' && row.cacheWrite >= minWrite) {
      let cause = 'unknown';
      if (row.compaction || prevAnthropic?.compaction || row.requestClass === 'compaction' || prevAnthropic?.requestClass === 'compaction') {
        cause = 'compaction';
      } else if (row.routeReason?.startsWith('failover:') || row.routeReason?.startsWith('skipped:')) {
        cause = 'failover-from-cheap';
      } else if (prevAny && prevAny.finalProvider !== 'anthropic') {
        cause = 'return-from-cheap';
      } else if (!prevAny) {
        cause = 'first-turn-of-session';
      } else if (prevAnthropic && row.requestedModel !== prevAnthropic.requestedModel) {
        cause = 'model-switch';
      } else if (prevAnthropic && row.systemHash && prevAnthropic.systemHash && row.systemHash !== prevAnthropic.systemHash) {
        cause = 'system-changed';
      } else if (prevAnthropic && row.toolsHash && prevAnthropic.toolsHash && row.toolsHash !== prevAnthropic.toolsHash) {
        cause = 'tools-changed';
      } else if (prevAnthropic) {
        const gap = row.createdAt - prevAnthropic.createdAt;
        if (gap > 3600000) cause = 'gap>1h';
        else if (gap > 300000) cause = 'gap5-60m';
      }
      
      misses.push({ ...row, cause });
      if (!summary[cause]) summary[cause] = { count: 0, tokens: 0 };
      const s = summary[cause]!;
      s.count++;
      s.tokens += row.cacheWrite;
    }
    
    if (row.session) {
      lastAny.set(sKey, row);
      if (row.finalProvider === 'anthropic') {
        lastAnthropic.set(sKey, row);
      }
    }
  }
  
  return { misses, summary, oversized };
}

export function renderCacheMisses(stats: any) {
  const lines = [];
  const tRows = [['Time', 'Session', 'Class', 'Model', 'Tokens in', 'Cache write', '5m/1h', 'Cause']];
  for (const m of stats.misses) {
    const time = new Date(m.createdAt).toLocaleTimeString();
    const split = (m.cacheWrite5m != null || m.cacheWrite1h != null) ? (m.cacheWrite5m ?? 0) + '/' + (m.cacheWrite1h ?? 0) : '?';
    tRows.push([time, (m.session ?? '').slice(0, 8), m.requestClass ?? '?', m.requestedModel ?? '?', fmt(m.tokensIn), fmt(m.cacheWrite), split, m.cause]);
  }
  if (tRows.length > 1) lines.push(table(tRows), '');
  
  const sumRows = [['Cause', 'Count', 'Cache write tokens']];
  for (const [c, s] of Object.entries(stats.summary) as any) {
    sumRows.push([c, fmt(s.count), fmt(s.tokens)]);
  }
  if (sumRows.length > 1) lines.push(table(sumRows), '');
  
  if (stats.oversized && stats.oversized.length > 0) {
    lines.push('Oversized rows (tokens_in > 1,000,000): ' + stats.oversized.length);
    for (const o of stats.oversized) {
      lines.push('  id=' + o.id + ', tokens_in=' + fmt(o.tokensIn) + ', cache_read=' + fmt(o.cacheRead) + ', cache_write=' + fmt(o.cacheWrite) + ', fresh=' + fmt(o.tokensIn - (o.cacheRead || 0) - (o.cacheWrite || 0)));
    }
  } else {
    lines.push('Oversized rows (tokens_in > 1,000,000): 0');
  }
  return lines.join('\n');
}

export function computeDailyStats(db: TelemetryDb, pricing: Pricing, options: StatsOptions = {}) {
  const { since } = options;
  const params = { since: since?.getTime() ?? null };
  const rows = db.prepare(
    'SELECT created_at as createdAt, prompt_id as promptId, final_provider as finalProvider, ' +
    '       tokens_in as tokensIn, cache_read_tokens as cacheRead, cache_write_tokens as cacheWrite, ' +
    '       tokens_out as tokensOut, cache_write_5m_tokens as cacheWrite5m, cache_write_1h_tokens as cacheWrite1h, ' +
    '       requested_model as requestedModel ' +
    'FROM router_logs WHERE (@since IS NULL OR created_at >= @since)'
  ).all(params) as any[];

  const byDay = new Map<string, any>();
  const taskTotals = new Map<string, any>();
  
  for (const r of rows) {
    const day = new Date(r.createdAt).toLocaleDateString();
    if (!byDay.has(day)) byDay.set(day, { requests: 0, tokensIn: 0, usd: 0, tasks: new Set() });
    const d = byDay.get(day);
    d.requests++;
    
    let usd = 0;
    if (r.finalProvider === 'anthropic') {
      const p = pricing.primary(r.requestedModel);
      if (p) {
        let writeCost = (r.cacheWrite || 0) * p.input * pricing.cacheWriteMultiplier;
        if (r.cacheWrite5m != null || r.cacheWrite1h != null) {
           writeCost = ((r.cacheWrite5m ?? 0) * 1.25 + (r.cacheWrite1h ?? 0) * 2) * p.input;
        }
        usd = (Math.max(0, (r.tokensIn || 0) - (r.cacheRead || 0) - (r.cacheWrite || 0)) * p.input + writeCost + (r.cacheRead || 0) * p.cacheRead + (r.tokensOut || 0) * p.output) / 1000000;
      }
      d.tokensIn += (r.tokensIn || 0);
      d.usd += usd;
    }
    
    if (r.promptId) {
      d.tasks.add(r.promptId);
      if (!taskTotals.has(r.promptId)) taskTotals.set(r.promptId, { usd: 0, requests: 0 });
      const t = taskTotals.get(r.promptId);
      t.requests++;
      if (r.finalProvider === 'anthropic') {
        t.usd += usd;
      }
    }
  }
  
  const costs = Array.from(taskTotals.values()).map((t: any) => t.usd).sort((a: number, b: number) => a - b);
  const median = costs.length > 0 ? costs[Math.floor(costs.length / 2)] : 0;
  const p90 = costs.length > 0 ? costs[Math.floor(costs.length * 0.9)] : 0;
  
  const top10 = Array.from(taskTotals.entries())
    .sort((a, b) => b[1].usd - a[1].usd)
    .slice(0, 10)
    .map(([id, t]) => ({ promptId: id, requests: t.requests, usd: t.usd }));

  const daily = Array.from(byDay.entries()).map(([day, s]) => ({ day, requests: s.requests, tasks: s.tasks.size, tokensIn: s.tokensIn, usd: s.usd }));
  
  return { daily, tasks: taskTotals.size, median, p90, top10 };
}

export function renderDailyStats(stats: any) {
  const lines = [];
  const dRows = [['Day', 'Requests', 'Tasks', 'Anthropic tokens', 'USD proxy']];
  for (const d of stats.daily) dRows.push([d.day, fmt(d.requests), fmt(d.tasks), fmt(d.tokensIn), '$' + d.usd.toFixed(4)]);
  if (dRows.length > 1) lines.push(table(dRows), '');
  else lines.push('No daily data', '');
  
  lines.push('Tasks overall: ' + fmt(stats.tasks));
  lines.push('Median task USD: $' + stats.median.toFixed(4));
  lines.push('p90 task USD: $' + stats.p90.toFixed(4));
  lines.push('');
  
  const tRows = [['Prompt ID (task)', 'Requests', 'USD']];
  for (const t of stats.top10) tRows.push([t.promptId.slice(0, 8), fmt(t.requests), '$' + t.usd.toFixed(4)]);
  if (tRows.length > 1) {
    lines.push('Top 10 expensive tasks:');
    lines.push(table(tRows));
  }
  
  return lines.join('\n');
}
