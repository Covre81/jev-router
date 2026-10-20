import { spawn, type ChildProcess } from 'node:child_process';
import { killTree } from '../cli-kill.js';
import { existsSync, readdirSync } from 'node:fs';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export interface CursorRunnerOptions {
  /** Cursor agent root: `%LOCALAPPDATA%\cursor-agent` (holds `versions\<date>\node.exe` + `index.js`). */
  readonly bin: string;
  readonly model: string;
  readonly timeoutMs: number;
  readonly signal?: AbortSignal;
}

export type CursorResult =
  | { readonly ok: true; readonly text: string; readonly usage: { readonly inputTokens: number; readonly outputTokens: number } }
  | { readonly ok: false; readonly reason: string };

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

function toolKind(ev: any): string {
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

export async function runCursorAgent(
  prompt: string,
  opts: CursorRunnerOptions,
  spawnFn: SpawnCursor = spawn,
): Promise<CursorResult> {
  if (opts.signal?.aborted) return { ok: false, reason: 'aborted' };
  const launch = resolveCursorLaunch(opts.bin);
  if (!launch) return { ok: false, reason: 'cursor agent not found' };

  const cwd = await mkdtemp(join(tmpdir(), 'jev-cursor-'));
  try {
    const args = [
      launch.indexJs,
      '-p',
      '--mode', 'ask',
      '--trust',
      '--output-format', 'stream-json',
      '--model', opts.model,
      prompt,
    ];
    // Subscription login must win and lives in the real ~/.cursor: keep HOME/USERPROFILE,
    // strip API keys so the child can never bill an API account instead.
    const STRIPPED_ENV = new Set(['CURSOR_API_KEY', 'ANTHROPIC_API_KEY', 'TYPESAFE_API_KEY', 'PROXY_AUTH_TOKEN', 'CHEAP_API_KEY']);
    const childEnv: NodeJS.ProcessEnv = { ...process.env };
    for (const key of Object.keys(childEnv)) {
      if (STRIPPED_ENV.has(key.toUpperCase())) delete childEnv[key];
    }

    const child = spawnFn(launch.nodeExe, args, {
      cwd,
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

    return await new Promise<CursorResult>((resolve) => {
      const settle = (r: CursorResult) => {
        if (settled) return;
        settled = true;
        clearTimeout(timeout);
        opts.signal?.removeEventListener('abort', abortHandler);
        resolve(r);
      };
      const abortHandler = () => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'aborted' });
      };
      const timeout = setTimeout(() => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'timeout' });
      }, opts.timeoutMs);
      if (opts.signal?.aborted) return settle({ ok: false, reason: 'aborted' });
      opts.signal?.addEventListener('abort', abortHandler);

      const handleEvent = (ev: any) => {
        if (!ev || typeof ev !== 'object') return;
        if (isToolStarted(ev)) {
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
        stdoutBuffer += chunk.toString('utf8');
        const lines = stdoutBuffer.split('\n');
        stdoutBuffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try { handleEvent(JSON.parse(line)); } catch {}
        }
      });
      child.stderr!.on('data', (chunk) => {
        stderrText = (stderrText + chunk.toString('utf8')).slice(0, 500);
      });
      child.on('error', (err) => settle({ ok: false, reason: `spawn failed: ${err.message}` }));
      child.on('close', (code) => {
        if (stdoutBuffer.trim()) {
          try { handleEvent(JSON.parse(stdoutBuffer)); } catch {}
        }
        if (toolAttempted) return settle({ ok: false, reason: `tool attempt: ${toolAttempted}` });
        if (opts.signal?.aborted) return settle({ ok: false, reason: 'aborted' });
        if (code !== 0) {
          const detail = stderrText.trim().replace(/\s+/g, ' ');
          return settle({ ok: false, reason: detail ? `exit ${code}: ${detail}` : `exit ${code}` });
        }
        if (!finalResult) return settle({ ok: false, reason: 'no result event' });
        if (finalResult.is_error || finalResult.subtype !== 'success') {
          const detail = typeof finalResult.result === 'string' && finalResult.result ? `: ${finalResult.result.slice(0, 200)}` : '';
          return settle({ ok: false, reason: `result error${detail}` });
        }
        const text = (typeof finalResult.result === 'string' && finalResult.result.trim() ? finalResult.result : segmentText || aggregateText).trim();
        if (!text) return settle({ ok: false, reason: 'empty response' });
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
    try { await rm(cwd, { recursive: true, force: true }); } catch {}
  }
}