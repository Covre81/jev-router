import { mkdirSync, readdirSync, rmSync, statSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { jevHome } from './paths.js';

/** One empty file per running subagent: `<dir>/<session id>/<agent id>`. */
export function defaultAgentsDir(env: NodeJS.ProcessEnv = process.env): string {
  return join(jevHome(env), 'agents');
}

/**
 * A session folder untouched this long belongs to a Claude Code that died and
 * was never resumed: no live session goes a week without starting or stopping an agent.
 */
export const STALE_SESSION_MS = 7 * 24 * 3_600_000;

/** Ids become path segments: anything else (`..`, separators) is ignored. */
const SAFE_ID = /^[\w-]+$/;
const safe = (id: unknown): id is string => typeof id === 'string' && SAFE_ID.test(id);

/** Windows: Defender or the indexer can hold a fresh file for a moment (EBUSY/EPERM). */
const remove = (path: string) => rmSync(path, { recursive: true, force: true, maxRetries: 3 });

/** The fields of a Claude Code or Antigravity/agy hook payload the registry reads. */
export interface AgentHookEvent {
  readonly hook_event_name?: unknown;
  readonly hookEventName?: unknown;
  readonly session_id?: unknown;
  readonly conversationId?: unknown;
  readonly agent_id?: unknown;
  readonly agentId?: unknown;
  readonly source?: unknown;
}

/**
 * Applies one Claude Code or Antigravity hook event: SubagentStart adds the agent (again on a
 * resume, which restarts its clock), SubagentStop removes it, SessionEnd drops
 * the session. SessionStart sweeps what a crash left behind: a resumed session's
 * subagents died with the process that ran them, and folders of sessions never
 * resumed go after STALE_SESSION_MS. Other events are ignored.
 *
 * ponytail: /resume of a session still open in another terminal wipes that
 * terminal's live count too; it fixes itself at their next Start.
 */
export function recordAgentEvent(event: AgentHookEvent, dir = defaultAgentsDir(), now = Date.now()): void {
  const sessionId = event.session_id ?? event.conversationId;
  if (!safe(sessionId)) return;
  const session = join(dir, sessionId);
  const eventName = event.hook_event_name ?? event.hookEventName;
  const agentId = event.agent_id ?? event.agentId;
  switch (eventName) {
    case 'SessionStart':
      if (event.source === 'resume') remove(session);
      sweepStale(dir, now);
      return;
    case 'SessionEnd':
      remove(session);
      return;
    case 'SubagentStart':
      if (!safe(agentId)) return;
      mkdirSync(session, { recursive: true });
      writeFileSync(join(session, agentId), '');
      return;
    case 'SubagentStop':
      if (safe(agentId)) remove(join(session, agentId));
      return;
  }
}

function sweepStale(dir: string, now: number): void {
  let sessions: string[];
  try {
    sessions = readdirSync(dir);
  } catch {
    return; // no agent ever started
  }
  for (const name of sessions) {
    const path = join(dir, name);
    try {
      if (now - statSync(path).mtimeMs > STALE_SESSION_MS) remove(path);
    } catch {
      // Gone already, or locked: the next SessionStart tries again.
    }
  }
}

/**
 * Subagents of the session started and not yet stopped, and when the oldest
 * started: its file is written at SubagentStart, so its mtime is the start.
 */
export function runningAgents(sessionId: string | undefined, dir = defaultAgentsDir()): { count: number; oldestStartMs: number } | undefined {
  if (!safe(sessionId)) return undefined;
  try {
    const session = join(dir, sessionId);
    const starts = readdirSync(session).map((agent) => statSync(join(session, agent)).mtimeMs);
    return starts.length ? { count: starts.length, oldestStartMs: Math.min(...starts) } : undefined;
  } catch {
    return undefined;
  }
}
