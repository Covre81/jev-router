import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import { runCursorAgent } from '../src/providers/cursor/agent-runner.js';

function fakeChild(script: (child: any) => void) {
  const child = new EventEmitter() as any;
  child.pid = 99999;
  child.stdin = new Writable({ write(_c, _e, cb) { cb(); } });
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  setTimeout(() => script(child), 5);
  return child;
}

const line = (o: unknown) => Buffer.from(JSON.stringify(o) + '\n');
const readPrompt = (subtype: string) => ({
  type: 'tool_call', subtype, timestamp_ms: 2,
  tool_call: { readToolCall: { args: { path: '/tmp/jev-cursor-x/prompt.md' } } },
});

// Regression: the real Cursor stream (captured 2026-11-18 on cursor-agent 2026.11.11) narrates before
// reading prompt.md, and its `result` glues that narration to the answer:
// "I'll read `prompt.md` and follow it exactly.ok". Only the text after the read is the answer.
describe('cursor answer excludes narration before the prompt.md read', () => {
  it('returns only the post-read answer, not the concatenated result', async () => {
    const root = join(tmpdir(), `cursor-test-narration-${Date.now()}`);
    const ver = join(root, 'versions', '2026.11.11-test');
    await mkdir(ver, { recursive: true });
    await writeFile(join(ver, 'node'), '');
    await writeFile(join(ver, 'index.js'), '');
    try {
      const spawnFn = () => fakeChild((child) => {
        child.stdout.emit('data', line({ type: 'system', subtype: 'init' }));
        child.stdout.emit('data', line({ type: 'assistant', timestamp_ms: 1, message: { content: [{ type: 'text', text: "I'll read `prompt.md` and follow it exactly." }] } }));
        child.stdout.emit('data', line(readPrompt('started')));
        child.stdout.emit('data', line(readPrompt('completed')));
        child.stdout.emit('data', line({ type: 'assistant', message: { content: [{ type: 'text', text: 'ok' }] } }));
        child.stdout.emit('data', line({
          type: 'result', subtype: 'success', is_error: false,
          result: "I'll read `prompt.md` and follow it exactly.ok", usage: { inputTokens: 10, outputTokens: 2 },
        }));
        child.emit('close', 0);
      });

      const res = await runCursorAgent('Responda apenas: ok', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
      assert.equal(res.ok, true);
      if (res.ok) assert.equal(res.text, 'ok');
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
