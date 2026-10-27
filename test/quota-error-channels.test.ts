import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import { runAgy } from '../src/providers/gemini/agy-runner.js';
import { runCursorAgent } from '../src/providers/cursor/agent-runner.js';

const line = (o: unknown) => Buffer.from(JSON.stringify(o) + '\n');

function fakeChild(script: (child: any) => void) {
  const child = new EventEmitter() as any;
  child.stdin = new Writable({ write(_c, _e, cb) { cb(); } });
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.pid = 9999;
  setTimeout(() => script(child), 5);
  return child;
}

const runAgyWith = (script: (child: any) => void) =>
  runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 2000, home: tmpdir() }, (() => fakeChild(script)) as any);

describe('agy runner: quota only from genuine error channels', () => {
  it('assistant text mentioning rate limit / quota exceeded / 429 on a successful run is relayed, not quota', async () => {
    const answer = 'A rate limit returns HTTP 429 when the quota exceeded the window; the quota exceeded error is retryable.';
    const res = await runAgyWith((child) => {
      child.stdout.emit('data', line({ event: 'step_update', step_update: { step_type: 'agent_response', text_delta: answer } }));
      child.stdout.emit('data', line({ event: 'result', result: { status: 'SUCCESS', response: answer, usage: { input_tokens: 3, output_tokens: 9 } } }));
      child.emit('close', 0);
    });
    assert.deepEqual(res, { ok: true, text: answer, usage: { inputTokens: 3, outputTokens: 9 } });
  });

  it('streamed text only (no response field) with quota words still succeeds', async () => {
    const answer = 'Use a token bucket. When you hit 429 or "quota exceeded", back off.';
    const res = await runAgyWith((child) => {
      child.stdout.emit('data', line({ event: 'step_update', step_update: { step_type: 'agent_response', text_delta: answer } }));
      child.stdout.emit('data', line({ event: 'result', result: { status: 'SUCCESS' } }));
      child.emit('close', 0);
    });
    assert.equal(res.ok, true);
    if (res.ok) assert.equal(res.text, answer);
  });

  it('result status ERROR "Individual quota reached ... Resets in 134h55m17s" -> quota with ~134h reset', async () => {
    const before = Date.now();
    const res = await runAgyWith((child) => {
      child.stdout.emit('data', line({
        event: 'result',
        result: { status: 'ERROR', error: 'Individual quota reached. Please upgrade your subscription to increase your limits. Resets in 134h55m17s.' },
      }));
      child.emit('close', 1);
    });
    assert.equal(res.ok, false);
    if (!res.ok) {
      assert.equal(res.reason, 'quota');
      const expected = (134 * 3600 + 55 * 60 + 17) * 1000;
      const delta = res.quota!.resetAt - before;
      assert.ok(Math.abs(delta - expected) < 5_000, `resetAt delta ${delta} vs ${expected}`);
    }
  });

  it('error event channel with quota text -> quota', async () => {
    const res = await runAgyWith((child) => {
      child.stdout.emit('data', line({ event: 'error', error: 'status ERROR: Individual quota reached. Resets in 2h' }));
      child.emit('close', 1);
    });
    assert.equal(res.ok, false);
    if (!res.ok) assert.equal(res.reason, 'quota');
  });

  it('stderr-only quota text on non-zero exit -> quota', async () => {
    const res = await runAgyWith((child) => {
      child.stderr.emit('data', Buffer.from('fatal: RESOURCE_EXHAUSTED: Individual quota reached. Resets in 3h'));
      child.emit('close', 1);
    });
    assert.equal(res.ok, false);
    if (!res.ok) {
      assert.equal(res.reason, 'quota');
      assert.ok(res.quota);
    }
  });

  it('stderr noise on a successful run does not mark quota', async () => {
    const res = await runAgyWith((child) => {
      child.stderr.emit('data', Buffer.from('warn: rate limit exceeded on telemetry endpoint'));
      child.stdout.emit('data', line({ event: 'result', result: { status: 'SUCCESS', response: 'fine' } }));
      child.emit('close', 0);
    });
    assert.equal(res.ok, true);
  });
});

describe('cursor runner: quota only from genuine error channels', () => {
  async function fakeCursorRoot() {
    const root = await mkdtemp(join(tmpdir(), 'jev-cursor-q-'));
    const ver = join(root, 'versions', '2026.10.01-test');
    await mkdir(ver, { recursive: true });
    await writeFile(join(ver, 'node.exe'), '');
    await writeFile(join(ver, 'index.js'), '');
    return root;
  }

  const runCursorWith = async (script: (child: any) => void) => {
    const root = await fakeCursorRoot();
    try {
      return await runCursorAgent(
        'prompt',
        { bin: root, model: 'auto', timeoutMs: 2000, firstByteTimeoutMs: 1000 },
        (() => fakeChild(script)) as any,
      );
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  };

  it('successful result whose text discusses rate limits / quota exceeded / 429 is relayed', async () => {
    const answer = 'Rate limit exceeded errors are HTTP 429; quota exceeded means billing. Use a token bucket.';
    const res = await runCursorWith((child) => {
      child.stdout.emit('data', line({ type: 'result', subtype: 'success', is_error: false, result: answer }));
      child.emit('close', 0);
    });
    assert.equal(res.ok, true);
    if (res.ok) assert.equal(res.text, answer);
  });

  it('result is_error with "You\'ve hit your usage limit" -> quota', async () => {
    const res = await runCursorWith((child) => {
      child.stdout.emit('data', line({ type: 'result', subtype: 'error', is_error: true, result: "You've hit your usage limit. Resets in 2h" }));
      child.emit('close', 0);
    });
    assert.equal(res.ok, false);
    if (!res.ok) {
      assert.equal(res.reason, 'quota');
      assert.ok(res.quota);
    }
  });

  it('stderr-only quota text on non-zero exit -> quota', async () => {
    const res = await runCursorWith((child) => {
      child.stderr.emit('data', Buffer.from("You've hit your usage limit"));
      child.emit('close', 1);
    });
    assert.equal(res.ok, false);
    if (!res.ok) assert.equal(res.reason, 'quota');
  });
});
