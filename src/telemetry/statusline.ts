import type { TelemetryDb } from './db.js';
import type { FinalProvider, Outcome } from './schema.js';

/** What the Claude Code status line needs: is the router up, and where did this session's last turn go. */
export interface StatusLineData {
  readonly healthy: boolean;
  /** The build on disk is newer than the one running: someone forgot `jev-router reload`. */
  readonly staleBuild?: boolean;
  /** The router's health check sees the cheap provider down: cheap turns go to Claude. */
  readonly cheapDown?: boolean;
  readonly last?: LastRoute;
  /** Today's routed turns (see cheapShare), and how many the cheap model answered. */
  readonly today?: { readonly cheap: number; readonly total: number };
  /** Context of the session's last main-agent request (input + cache), in tokens. */
  readonly context?: number;
  /** Subagents of the session running now (SubagentStart/Stop hooks, see agents.ts) and how long the oldest has run. */
  readonly agents?: { readonly count: number; readonly longestMs: number };
  /** Binding Claude quota window, as last persisted by the router. */
  readonly quota?: { readonly utilization: number; readonly window: string };
}

/** From here the router lowers its cheap bars (QUOTA_PRESSURE default), so the line warns too. */
export const QUOTA_WARN = 0.8;

/** Latest persisted quota reading; older than `maxAgeMs` is stale and not shown. */
export function latestQuota(
  db: TelemetryDb,
  { now = Date.now(), maxAgeMs = 6 * 3_600_000 } = {},
): { utilization: number; window: string; createdAt: Date } | undefined {
  const row = db
    .prepare(`SELECT utilization, window, created_at AS createdAt FROM quota_observations ORDER BY id DESC LIMIT 1`)
    .get() as { utilization: number; window: string; createdAt: number } | undefined;
  if (!row || now - row.createdAt > maxAgeMs) return undefined;
  return { utilization: row.utilization, window: row.window, createdAt: new Date(row.createdAt) };
}

/**
 * Past this the session is where the money goes: in the 7 days to 2026-10-06,
 * requests above 200k carried 92% of main-session tokens. Every tool call
 * re-reads the whole context, so /compact or /clear pays back at once.
 */
export const CONTEXT_WARN_TOKENS = 200_000;

export interface LastRoute {
  readonly provider: FinalProvider;
  readonly reason: string;
  readonly outcome: Outcome;
  /** P(simple) from JEV; undefined when JEV was not asked (sticky turn, auxiliary request). */
  readonly pSimple?: number;
  /** Model that served the turn, when it is not the one Claude Code asked for (cheap or gemini route). */
  readonly model?: string;
}

interface LastRow {
  provider: FinalProvider;
  reason: string;
  outcome: Outcome;
  jev: string | null;
  model: string | null;
  requested: string | null;
}

/**
 * Last exchange of a Claude Code session. Telemetry keys conversations as
 * `<session id>:<agent>`, so every agent of the session counts. Without a
 * session id (or before its first turn) there is nothing to show.
 */
export function lastRoute(db: TelemetryDb, sessionId: string | undefined): LastRoute | undefined {
  if (!sessionId) return undefined;
  const row = db
    .prepare(
      `SELECT final_provider AS provider, route_reason AS reason, outcome, jev_decision AS jev,
              model, requested_model AS requested
       FROM router_logs WHERE substr(session_id, 1, length(@prefix)) = @prefix ORDER BY id DESC LIMIT 1`,
    )
    .get({ prefix: `${sessionId}:` }) as LastRow | undefined;
  if (!row) return undefined;
  const pSimple = row.jev ? (JSON.parse(row.jev) as { pSimple?: number }).pSimple : undefined;
  const served = row.model && row.model !== row.requested ? row.model : undefined;
  return {
    provider: row.provider,
    reason: row.reason,
    outcome: row.outcome,
    ...(pSimple === undefined ? {} : { pSimple }),
    ...(served === undefined ? {} : { model: served }),
  };
}

/** Context size of the session's latest main-agent request that reported usage. */
export function sessionContext(db: TelemetryDb, sessionId: string | undefined): number | undefined {
  if (!sessionId) return undefined;
  const row = db
    .prepare(`SELECT tokens_in AS tokens FROM router_logs WHERE session_id = @key AND tokens_in IS NOT NULL ORDER BY id DESC LIMIT 1`)
    .get({ key: `${sessionId}:main` }) as { tokens: number } | undefined;
  return row?.tokens;
}

/**
 * Routed turns since `since` and how many of them the cheap provider answered.
 * Counting requests buried the routing in traffic it never decides: auxiliary
 * requests always go primary, and every tool call of a primary session is one
 * more request.
 *
 * A routed turn is a main-agent request carrying typed text (audit.ts hashes
 * any trailing user text, auxiliary requests included, so the request class is
 * what singles out the human), or any request JEV scored, which adds subagent
 * conversations. Without CLAUDE_CODE_GATEWAY_HINT_HEADERS=1 there is no request
 * class and only JEV decisions count.
 */
export function cheapShare(db: TelemetryDb, since: Date): { cheap: number; total: number } {
  return db
    .prepare(
      `SELECT count(*) AS total, coalesce(sum(CASE WHEN final_provider = 'openai' AND outcome = 'ok' THEN 1 ELSE 0 END), 0) AS cheap
       FROM router_logs
       WHERE created_at >= @since AND ((request_class = 'main' AND human_prompt_hash IS NOT NULL) OR jev_decision IS NOT NULL)`,
    )
    .get({ since: since.getTime() }) as { cheap: number; total: number };
}

/** How much the line asks for attention: offline is critical, anything the line flags is a warning. */
export type StatusLevel = 'ok' | 'warn' | 'critical';

/**
 * The line plus its level, for `statusline --json`: a status line that colours
 * by level reads this field instead of parsing ✗ and ⚠ out of the text.
 */
export function statusLineReport(data: StatusLineData): { text: string; level: StatusLevel } {
  return { text: renderStatusLine(data), level: statusLevel(data) };
}

function statusLevel({ healthy, staleBuild, cheapDown, last, context, quota }: StatusLineData): StatusLevel {
  if (!healthy) return 'critical';
  const flagged =
    staleBuild ||
    cheapDown ||
    (last !== undefined && last.outcome !== 'ok') ||
    (context !== undefined && context >= CONTEXT_WARN_TOKENS) ||
    (quota !== undefined && quota.utilization >= QUOTA_WARN);
  return flagged ? 'warn' : 'ok';
}

/** 45s, 12m, 1h05: short enough for the line, precise enough to spot a stuck agent. */
export function elapsed(ms: number): string {
  const s = Math.max(0, Math.floor(ms / 1000));
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h${String(Math.floor((s % 3600) / 60)).padStart(2, '0')}`;
}

/** One line, plain text: Claude Code prints the first line of stdout under the prompt. */
export function renderStatusLine({ healthy, staleBuild, cheapDown, last, agents, today, context, quota }: StatusLineData): string {
  if (!healthy) return 'jev-router ✗ offline';
  const parts = ['jev-router ✓'];
  if (staleBuild) parts.push('⚠ build velho');
  if (cheapDown) parts.push('cheap ✗');
  if (last) {
    const where = last.provider === 'openai' ? 'cheap' : last.provider === 'gemini' ? 'gemini' : last.provider === 'cursor' ? 'cursor' : 'claude';
    const why = last.pSimple !== undefined ? `JEV ${last.pSimple.toFixed(2)}` : last.reason;
    parts.push(`last: ${where}${last.model ? ` ${last.model}` : ''} (${why})${last.outcome === 'ok' ? '' : ` ${last.outcome}`}`);
  }
  if (agents) parts.push(`🤖 ${agents.count} ⏳ ${elapsed(agents.longestMs)}`);
  if (context !== undefined) {
    const k = `ctx ${Math.round(context / 1000)}k`;
    parts.push(context >= CONTEXT_WARN_TOKENS ? `⚠ ${k} → /compact or /clear` : k);
  }
  if (today && today.total > 0) parts.push(`today ${today.cheap}/${today.total} cheap`);
  if (quota) {
    const q = `cota ${Math.round(quota.utilization * 100)}% ${quota.window}`;
    parts.push(quota.utilization >= QUOTA_WARN ? `⚠ ${q}` : q);
  }
  return parts.join(' · ');
}
