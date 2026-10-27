import { spawn, type ChildProcess } from 'node:child_process';
import { killTree } from '../cli-kill.js';
import { existsSync, readdirSync } from 'node:fs';
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { parseQuotaError } from '../quota-state.js';

export interface CursorRunnerOptions {
  /** Cursor agent root: `%LOCALAPPDATA%\cursor-agent` (holds `versions\<date>\node.exe` + `index.js`). */
  readonly bin: string;
  readonly model: string;
  readonly timeoutMs: number;
  readonly firstByteTimeoutMs?: number | undefined;
  readonly idleTimeoutMs?: number | undefined;
  readonly signal?: AbortSignal | undefined;
}

export type CursorResult =
  | { readonly ok: true; readonly text: string; readonly usage: { readonly inputTokens: number; readonly outputTokens: number } }
  | { readonly ok: false; readonly reason: string; readonly quota?: { readonly resetAt: number } | undefined };

export type SpawnCursor = (bin: string, args: string[], options: any) => ChildProcess;

/** Latest `versions\<YYYY.MM.DD-*>` layout, spawned like agent.ps1 does (no shell, no .cmd shim). */
export function resolveCursorLaunch(binRoot: string): { nodeExe: string; indexJs: string } | undefined {
  try {
    const versionsDir = join(binRoot, 'versions');
    const candidates = readdirSync(versionsDir, { withFileTypes: true })
      .filter((d) => d.isDirectory() && /^\d{4}\.\d{2}\.\d{2}-/.test(d.name))
      .map((d) => d.name)
      .sort()
      .reverse();
    for (const name of candidates) {
      const nodeExe = join(versionsDir, name, 'node.exe');
      const indexJs = join(versionsDir, name, 'index.js');
      if (existsSync(nodeExe) && existsSync(indexJs)) return { nodeExe, indexJs };
    }
  } catch {
    // missing layout: unavailable, never crash
  }
  return undefined;
}

function textOfAssistant(ev: any): string {
  const content = ev?.message?.content ?? ev?.content;
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    return content
      .filter((b: any) => b && (b.type === 'text' || typeof b.text === 'string'))
      .map((b: any) => (typeof b.text === 'string' ? b.text : ''))
      .join('');
  }
  return typeof ev?.text === 'string' ? ev.text : '';
}

export function toolKind(ev: any): string {
  const tc = ev?.tool_call ?? ev?.toolCall;
  if (tc && typeof tc === 'object') {
    const key = Object.keys(tc)[0];
    if (key) return key.replace(/ToolCall$/, '').replace(/([a-z])([A-Z])/g, '$1 $2').toLowerCase();
  }
  for (const v of [ev?.name, ev?.tool_name, ev?.toolName, ev?.kind]) if (typeof v === 'string' && v) return v;
  return 'unknown';
}

const isToolStarted = (ev: any): boolean =>
  ev?.type === 'tool_call' && ev?.subtype !== 'completed' && ev?.status !== 'completed';

function stderrTail(s: string): string {
  const trimmed = s.trim().replace(/\s+/g, ' ');
  return trimmed.slice(-300);
}

export async function runCursorAgent(
  prompt: string,
  opts: CursorRunnerOptions,
  spawnFn: SpawnCursor = spawn,
): Promise<CursorResult> {
  if (opts.signal?.aborted) return { ok: false, reason: 'aborted' };
  const launch = resolveCursorLaunch(opts.bin);
  if (!launch) return { ok: false, reason: 'cursor agent not found' };

  const workspaceDir = await mkdtemp(join(tmpdir(), 'jev-cursor-'));
  try {
    await writeFile(join(workspaceDir, 'prompt.md'), prompt, 'utf8');

    const args = [
      launch.indexJs,
      '-p',
      '--mode', 'ask',
      '--trust',
      '--workspace', workspaceDir,
      '--output-format', 'stream-json',
      '--model', opts.model,
      'Read the file prompt.md in the workspace and follow it exactly. Output only your answer.',
    ];
    // Subscription login must win and lives in the real ~/.cursor: keep HOME/USERPROFILE,
    // strip API keys so the child can never bill an API account instead.
    // MSYSTEM/EXEPATH (inherited from Git Bash) make Cursor pick a bash executor while it always
    // builds hook commands in PowerShell syntax on Windows: every hook then fails with
    // "eval: syntax error near unexpected token '&'".
    const STRIPPED_ENV = new Set(['CURSOR_API_KEY', 'ANTHROPIC_API_KEY', 'TYPESAFE_API_KEY', 'PROXY_AUTH_TOKEN', 'CHEAP_API_KEY', 'MSYSTEM', 'EXEPATH']);
    const childEnv: NodeJS.ProcessEnv = { ...process.env };
    for (const key of Object.keys(childEnv)) {
      if (STRIPPED_ENV.has(key.toUpperCase())) delete childEnv[key];
    }

    const child = spawnFn(launch.nodeExe, args, {
      cwd: workspaceDir,
      env: childEnv,
      windowsHide: true,
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    child.stdout?.on('error', () => {});
    child.stderr?.on('error', () => {});

    let settled = false;
    let toolAttempted: string | false = false;
    let stdoutBuffer = '';
    let stderrText = '';
    let segmentText = '';
    let aggregateText = '';
    let finalResult: any;
    let firstByteReceived = false;

    return await new Promise<CursorResult>((resolve) => {
      let firstByteTimer: NodeJS.Timeout | undefined;
      let idleTimer: NodeJS.Timeout | undefined;
      let overallTimer: NodeJS.Timeout | undefined;

      const settle = (r: CursorResult) => {
        if (settled) return;
        settled = true;
        if (firstByteTimer) clearTimeout(firstByteTimer);
        if (idleTimer) clearTimeout(idleTimer);
        if (overallTimer) clearTimeout(overallTimer);
        opts.signal?.removeEventListener('abort', abortHandler);
        resolve(r);
      };
      const abortHandler = () => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'aborted' });
      };

      const firstByteTimeoutMs = opts.firstByteTimeoutMs ?? 90_000;
      const idleTimeoutMs = opts.idleTimeoutMs ?? 60_000;

      firstByteTimer = setTimeout(() => {
        if (child.pid !== undefined) killTree(child.pid);
        const detail = stderrTail(stderrText);
        settle({ ok: false, reason: detail ? `first-byte-timeout: ${detail}` : 'first-byte-timeout' });
      }, firstByteTimeoutMs);

      overallTimer = setTimeout(() => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'timeout' });
      }, opts.timeoutMs);

      if (opts.signal?.aborted) return settle({ ok: false, reason: 'aborted' });
      opts.signal?.addEventListener('abort', abortHandler);

      const handleEvent = (ev: any) => {
        if (!ev || typeof ev !== 'object') return;
        if (isToolStarted(ev)) {
          // The prompt is delivered as workspace/prompt.md (argv is capped on Windows): reading it is expected.
          if (isPromptFileRead(ev) || isHarmlessCursorTool(ev)) return;
          toolAttempted = toolKind(ev);
          if (child.pid !== undefined) killTree(child.pid);
          return;
        }
        if (ev.type === 'assistant') {
          const text = textOfAssistant(ev);
          if (!text) return;
          // Segmented assistants carry timestamp_ms; the final aggregated one does not (would double the text).
          if (typeof ev.timestamp_ms === 'number') segmentText += text;
          else aggregateText = text;
        } else if (ev.type === 'result') {
          finalResult = ev;
        }
      };

      child.stdout!.on('data', (chunk) => {
        if (!firstByteReceived) {
          firstByteReceived = true;
          if (firstByteTimer) {
            clearTimeout(firstByteTimer);
            firstByteTimer = undefined;
          }
        }
        if (idleTimer) clearTimeout(idleTimer);
        idleTimer = setTimeout(() => {
          if (child.pid !== undefined) killTree(child.pid);
          const detail = stderrTail(stderrText);
          settle({ ok: false, reason: detail ? `idle-timeout: ${detail}` : 'idle-timeout' });
        }, idleTimeoutMs);

        stdoutBuffer += chunk.toString('utf8');
        const lines = stdoutBuffer.split('\n');
        stdoutBuffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try { handleEvent(JSON.parse(line)); } catch {}
        }
      });
      child.stderr!.on('data', (chunk) => {
        stderrText = (stderrText + chunk.toString('utf8')).slice(-1000);
      });
      child.on('error', (err) => settle({ ok: false, reason: `spawn failed: ${err.message}` }));
      child.on('close', (code) => {
        if (stdoutBuffer.trim()) {
          try { handleEvent(JSON.parse(stdoutBuffer)); } catch {}
        }
        if (toolAttempted) return settle({ ok: false, reason: `tool attempt: ${toolAttempted}` });
        if (opts.signal?.aborted) return settle({ ok: false, reason: 'aborted' });
        if (code !== 0) {
          const detail = stderrTail(stderrText);
          const errCandidate = [stderrText, detail].filter(Boolean).join(' ');
          const quotaParsed = parseQuotaError(errCandidate);
          if (quotaParsed) {
            return settle({ ok: false, reason: 'quota', quota: { resetAt: quotaParsed.resetAt } });
          }
          return settle({ ok: false, reason: detail ? `exit ${code}: ${detail}` : `exit ${code}` });
        }
        if (!finalResult) return settle({ ok: false, reason: 'no result event' });
        if (finalResult.is_error || finalResult.subtype !== 'success') {
          const resultStr = typeof finalResult.result === 'string' ? finalResult.result : '';
          const quotaParsed = parseQuotaError(resultStr || stderrText);
          if (quotaParsed) {
            return settle({ ok: false, reason: 'quota', quota: { resetAt: quotaParsed.resetAt } });
          }
          const detail = resultStr ? `: ${resultStr.slice(0, 200)}` : '';
          return settle({ ok: false, reason: `result error${detail}` });
        }
        const text = (typeof finalResult.result === 'string' && finalResult.result.trim() ? finalResult.result : segmentText || aggregateText).trim();
        if (!text) return settle({ ok: false, reason: 'empty' });

        settle({
          ok: true,
          text,
          usage: {
            inputTokens: finalResult.usage?.inputTokens ?? 0,
            outputTokens: finalResult.usage?.outputTokens ?? 0,
          },
        });
      });
    });
  } finally {
    try { await rm(workspaceDir, { recursive: true, force: true }); } catch {}
  }
}


/**
 * Harmless Cursor agent tools allowed during execution because they have no side
 * effects outside Cursor's own session state (such as internal planning/todo lists
 * and discovery of available MCP tools without executing them).
 *
 * Allowed tools:
 * - "update todos" (updateTodosToolCall): updates internal session todo state
 * - "todo write" (todoWriteToolCall): writes session todo item
 * - "read todos" (readTodosToolCall): reads session todo state
 * - "todo read" (todoReadToolCall): reads session todo item
 * - "get mcp tools" (getMcpToolsToolCall): lists available MCP tools without calling them
 * - "list mcp tools" (listMcpToolsToolCall): lists available MCP tools without calling them
 *
 * All other tools remain strictly blocked (shell/terminal execution, file edit/write,
 * arbitrary file reads, actual MCP tool invocation, web search/fetch, etc.).
 */
const harmlessToolsSet = new Set<string>([
  'update todos',
  'todo write',
  'read todos',
  'todo read',
  'get mcp tools',
  'list mcp tools',
]);

export const CURSOR_HARMLESS_TOOLS: ReadonlySet<string> = harmlessToolsSet;

/**
 * True when a tool call has no external side effects outside Cursor's internal session
 * (todo management, listing available MCP tools).
 * Matches on the normalised toolKind (exact match, case-insensitive), never substring.
 */
export function isHarmlessCursorTool(ev: unknown): boolean {
  if (!ev || typeof ev !== 'object') return false;
  const raw = toolKind(ev as any);
  const normalized = raw.replace(/[_-]+/g, ' ').replace(/\s+/g, ' ').trim().toLowerCase();
  return CURSOR_HARMLESS_TOOLS.has(normalized);
}

/** True when a tool event only reads the delivered prompt.md (the temp-file prompt channel). */
export function isPromptFileRead(ev: unknown): boolean {
  if (!ev || typeof ev !== 'object') return false;
  if (!/read/i.test(toolKind(ev as any))) return false;
  return /prompt\.md/.test(JSON.stringify(ev));
}
