import { existsSync } from 'node:fs';
import { runningAgents } from './agents.js';
import { readHookInput } from './hook-input.js';
import { isStaleBuild, type BuildInfo } from './build-info.js';
import { openTelemetryDb } from './telemetry/db.js';
import {
  cheapShare,
  lastRoute,
  latestQuota,
  renderStatusLine,
  sessionContext,
  type StatusLineData,
} from './telemetry/statusline.js';

const HEALTH_TIMEOUT_MS = 500;
const STDIN_TIMEOUT_MS = 300;

export interface StatusLineOptions {
  readonly dbPath: string;
  readonly healthUrl: string;
  readonly sessionId?: string;
  /** dist/build-info.json of this checkout, compared with what /healthz says is running. */
  readonly localBuild?: BuildInfo;
  /**
   * Echo the router's last Claude quota reading (`cota 42% 5h`). Off by default:
   * Claude Code's own status line JSON already carries the plan's 5h/7d usage.
   */
  readonly showQuota?: boolean;
  /** Running-subagent registry the hooks keep (agents.ts); defaults to ~/.jev-router/agents. */
  readonly agentsDir?: string;
}

/** The part of /healthz the status line reads; older routers answer only `{ok:true}`. */
interface Health {
  readonly sha?: string | null;
  readonly builtAt?: string | null;
  readonly cheap?: string;
}

/**
 * Builds the Claude Code status line. Claude Code runs it after each message;
 * it must answer fast and never fail, so every error degrades to less
 * information instead of an exception.
 */
export async function statusLine(options: StatusLineOptions): Promise<string> {
  return renderStatusLine(await statusLineData(options));
}

/** What the line is built from; `statusline --json` reports it with its level (statusLineReport). */
export async function statusLineData({ dbPath, healthUrl, sessionId, localBuild, showQuota = false, agentsDir }: StatusLineOptions): Promise<StatusLineData> {
  const health = await probeHealth(healthUrl);
  const healthy = health !== undefined;
  const agents = runningAgents(sessionId, agentsDir);
  let data: StatusLineData = {
    healthy,
    ...(isStaleBuild(localBuild, { sha: health?.sha ?? null, builtAt: health?.builtAt ?? null }) ? { staleBuild: true } : {}),
    ...(health?.cheap === 'down' ? { cheapDown: true } : {}),
    ...(agents ? { agents: { count: agents.count, longestMs: Date.now() - agents.oldestStartMs } } : {}),
  };
  if (healthy && existsSync(dbPath)) {
    try {
      const db = openTelemetryDb(dbPath, { readonly: true });
      try {
        const midnight = new Date();
        midnight.setHours(0, 0, 0, 0);
        const last = lastRoute(db, sessionId);
        const context = sessionContext(db, sessionId);
        let quota: ReturnType<typeof latestQuota>;
        try {
          if (showQuota) quota = latestQuota(db);
        } catch {
          // Database not migrated yet (old router still running): no quota to show.
        }
        data = { ...data, ...(last ? { last } : {}), ...(quota ? { quota } : {}), ...(context === undefined ? {} : { context }), today: cheapShare(db, midnight) };
      } finally {
        db.close();
      }
    } catch {
      // A locked or half-migrated database only costs the details.
    }
  }
  return data;
}

/** The /healthz body when the router answers ok; undefined when it is down. */
async function probeHealth(url: string): Promise<Health | undefined> {
  try {
    const res = await fetch(url, { signal: AbortSignal.timeout(HEALTH_TIMEOUT_MS) });
    if (!res.ok) return undefined;
    try {
      return (await res.json()) as Health;
    } catch {
      return {};
    }
  } catch {
    return undefined;
  }
}

/** Claude Code and Antigravity/agy pipe the session as JSON on stdin; anything else (a TTY, garbage, silence) means no session. */
export async function readSessionId(stdin: NodeJS.ReadStream = process.stdin): Promise<string | undefined> {
  const input = await readHookInput(stdin, STDIN_TIMEOUT_MS);
  const id = input?.['session_id'] ?? input?.['conversationId'];
  return typeof id === 'string' && id.length > 0 ? id : undefined;
}
