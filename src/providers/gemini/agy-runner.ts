import { spawn, type ChildProcess } from 'node:child_process';
import { killTree } from '../cli-kill.js';
import { mkdtemp, rm, readdir } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

export interface AgyRunnerOptions {
  readonly bin: string;
  readonly model: string;
  readonly timeoutMs: number;
  readonly firstByteTimeoutMs?: number | undefined;
  readonly idleTimeoutMs?: number | undefined;
  readonly home: string;
  readonly signal?: AbortSignal | undefined;
}

export type AgyResult =
  | { readonly ok: true; readonly text: string; readonly usage: { readonly inputTokens: number; readonly outputTokens: number } }
  | { readonly ok: false; readonly reason: string };

export type SpawnAgy = (bin: string, args: string[], options: any) => ChildProcess;

export async function runAgy(
  prompt: string,
  opts: AgyRunnerOptions,
  spawnFn: SpawnAgy = spawn
): Promise<AgyResult> {
  const cwd = await mkdtemp(join(tmpdir(), 'agy-'));
  let conversationId: string | undefined;

  try {
    const printTimeout = Math.ceil(opts.timeoutMs / 1000) + 5;
    const args = [
      '--model', opts.model,
      '--disable-slash-commands',
      '--input-format', 'stream-json',
      '--output-format', 'stream-json',
      '--print-timeout', `${printTimeout}s`,
      '--print='
    ];

    const childEnv: NodeJS.ProcessEnv = { ...process.env, USERPROFILE: opts.home, HOME: opts.home };
    const stripKeys = [
      'GEMINI_API_KEY', 'GOOGLE_API_KEY', 'GOOGLE_GENAI_API_KEY',
      'GOOGLE_GENAI_USE_VERTEXAI', 'GOOGLE_APPLICATION_CREDENTIALS', 'GOOGLE_CLOUD_PROJECT'
    ];
    for (const key of Object.keys(childEnv)) {
      if (stripKeys.includes(key.toUpperCase())) {
        delete childEnv[key];
      }
    }

    const child = spawnFn(opts.bin, args, {
      cwd,
      env: childEnv,
      windowsHide: true,
      stdio: ['pipe', 'pipe', 'pipe']
    });

    child.stdin?.on('error', () => {});
    child.stdout?.on('error', () => {});
    child.stderr?.on('error', () => {});

    let isSettled = false;
    let finalResult: any = undefined;
    let toolAttempted: string | false = false;
    let stdoutBuffer = '';
    let textDelta = '';
    let firstByteReceived = false;

    return await new Promise<AgyResult>((resolve) => {
      let firstByteTimer: NodeJS.Timeout | undefined;
      let idleTimer: NodeJS.Timeout | undefined;
      let overallTimer: NodeJS.Timeout | undefined;

      const abortHandler = () => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'aborted' });
      };

      const settle = (result: AgyResult) => {
        if (isSettled) return;
        isSettled = true;
        if (firstByteTimer) clearTimeout(firstByteTimer);
        if (idleTimer) clearTimeout(idleTimer);
        if (overallTimer) clearTimeout(overallTimer);
        opts.signal?.removeEventListener('abort', abortHandler);
        resolve(result);
      };

      const firstByteTimeoutMs = opts.firstByteTimeoutMs ?? 60_000;
      const idleTimeoutMs = opts.idleTimeoutMs ?? 60_000;

      firstByteTimer = setTimeout(() => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'first-byte-timeout' });
      }, firstByteTimeoutMs);

      overallTimer = setTimeout(() => {
        if (child.pid !== undefined) killTree(child.pid);
        settle({ ok: false, reason: 'timeout' });
      }, opts.timeoutMs);

      if (opts.signal?.aborted) return settle({ ok: false, reason: 'aborted' });
      opts.signal?.addEventListener('abort', abortHandler);

      try {
        child.stdin!.write(JSON.stringify({
          event: 'user',
          message: { role: 'user', content: prompt }
        }) + '\n');
        child.stdin!.end();
      } catch (err) {
        // Handle EPIPE etc if write fails immediately
      }

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
          settle({ ok: false, reason: 'idle-timeout' });
        }, idleTimeoutMs);

        stdoutBuffer += chunk.toString('utf8');
        const lines = stdoutBuffer.split('\n');
        stdoutBuffer = lines.pop() || '';
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const ev = JSON.parse(line);
            if (ev.conversation_id && typeof ev.conversation_id === 'string' && /^[A-Za-z0-9-]+$/.test(ev.conversation_id)) {
              conversationId = ev.conversation_id;
            }
            if (ev.event === 'step_update' && ev.step_update) {
              if (ev.step_update.step_type === 'tool') {
                toolAttempted = ev.step_update.tool_name || 'unknown';
                if (child.pid !== undefined) killTree(child.pid);
              } else if (ev.step_update.step_type === 'agent_response' && ev.step_update.text_delta) {
                textDelta += ev.step_update.text_delta;
              }
            } else if (ev.event === 'result' && ev.result) {
              finalResult = ev.result;
            }
          } catch {}
        }
      });

      child.on('error', (err) => {
        settle({ ok: false, reason: `spawn failed: ${err.message}` });
      });

      child.on('close', (code) => {
        if (stdoutBuffer.trim()) {
           try {
              const ev = JSON.parse(stdoutBuffer);
              if (ev.event === 'result' && ev.result) finalResult = ev.result;
           } catch {}
        }

        if (toolAttempted) {
          return settle({ ok: false, reason: `tool attempt: ${toolAttempted}` });
        }
        if (opts.signal?.aborted) {
          return settle({ ok: false, reason: 'aborted' });
        }
        if (code !== 0) {
          const reasonMsg = (finalResult && finalResult.status === 'ERROR' && finalResult.error) ? finalResult.error : '';
          if (finalResult && finalResult.status === 'ERROR') {
             return settle({ ok: false, reason: `status ERROR: ${reasonMsg}` });
          }
          return settle({ ok: false, reason: `exit ${code}` });
        }
        if (!finalResult) {
          return settle({ ok: false, reason: 'no result event' });
        }
        if (finalResult.status !== 'SUCCESS') {
          return settle({ ok: false, reason: `status ${finalResult.status}: ${finalResult.error || ''}`.trim() });
        }
        
        const responseText = (finalResult.response || textDelta).trim();
        if (!responseText) {
          return settle({ ok: false, reason: 'empty' });
        }
        
        settle({
          ok: true,
          text: finalResult.response || textDelta,
          usage: {
            inputTokens: finalResult.usage?.input_tokens ?? 0,
            outputTokens: finalResult.usage?.output_tokens ?? 0
          }
        });
      });
    });
  } finally {
    try { await rm(cwd, { recursive: true, force: true }); } catch {}
    if (conversationId) {
      const baseDir = join(opts.home, '.gemini', 'antigravity-cli');
      try { await rm(join(baseDir, 'brain', conversationId), { recursive: true, force: true }); } catch {}
      try { await rm(join(baseDir, 'annotations', `${conversationId}.pbtxt`), { force: true }); } catch {}
      try {
        const convDir = join(baseDir, 'conversations');
        const files = await readdir(convDir);
        for (const f of files) {
          if (f.startsWith(conversationId)) {
            await rm(join(convDir, f), { force: true, recursive: true });
          }
        }
      } catch {}
    }
  }
}
