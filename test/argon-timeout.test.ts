import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { describe, it } from 'node:test';
import { loadConfig } from '../src/config.js';
import { GeminiCliProvider } from '../src/providers/gemini/provider.js';

function silentChild() {
  const child = new EventEmitter() as any;
  child.pid = undefined; // nothing real to kill
  child.stdin = new EventEmitter();
  child.stdin.write = () => true;
  child.stdin.end = () => {};
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  return child;
}

const body = { model: 'claude-opus-5-5', max_tokens: 64, messages: [{ role: 'user', content: 'hi' }] } as any;

function provider() {
  return new GeminiCliProvider({
    bin: 'agy', model: 'flash-x', timeoutMs: 10_000, home: '.', maxPromptChars: 100_000,
    firstByteTimeoutMs: 60, idleTimeoutMs: 60,
    argonModels: ['argon-low-x', 'argon-high-x'], argonFirstByteTimeoutMs: 400, argonIdleTimeoutMs: 400,
    spawnFn: () => silentChild(),
  });
}

describe('Argon-specific no-output timeout', () => {
  it('Flash keeps the short first-byte limit', async () => {
    const t0 = Date.now();
    const res = await provider().send({ body } as any);
    const ms = Date.now() - t0;
    assert.equal(res.kind, 'unavailable');
    assert.ok(ms < 350, `flash took ${ms}ms`);
  });
  it('Argon models wait for the longer Argon limit', async () => {
    const t0 = Date.now();
    const res = await provider().send({ body, modelOverride: 'argon-high-x' } as any);
    const ms = Date.now() - t0;
    assert.equal(res.kind, 'unavailable');
    assert.ok(ms >= 380, `argon took ${ms}ms`);
  });
  it('defaults: Argon 180s, Flash/Cursor 60s', () => {
    const c = loadConfig({ CLASSIFIER: 'heuristic' } as any);
    assert.equal(c.argon.firstByteMs, 180_000);
    assert.equal(c.argon.idleMs, 180_000);
    assert.equal(c.timeouts.firstByteMs, 60_000);
    assert.equal(c.timeouts.idleMs, 60_000);
  });
});
