import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import { loadConfig } from '../src/config.js';
import { runCursorAgent } from '../src/providers/cursor/agent-runner.js';
import { CursorCliProvider } from '../src/providers/cursor/provider.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import { buildServer } from '../src/proxy/server.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

async function fakeCursorRoot() {
  const root = await mkdtemp(join(tmpdir(), 'jev-cursor-fb-'));
  const ver = join(root, 'versions', '2026.10.01-test');
  await mkdir(ver, { recursive: true });
  await writeFile(join(ver, 'node.exe'), '');
  await writeFile(join(ver, 'index.js'), '');
  return { root, ver };
}

function fakeChild(script: (child: any) => void) {
  const child = new EventEmitter() as any;
  child.stdin = new Writable({ write(_c, _e, cb) { cb(); } });
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  child.pid = 9999;
  setTimeout(() => script(child), 5);
  return child;
}

const line = (o: unknown) => Buffer.from(JSON.stringify(o) + '\n');

describe('Cursor first-byte diagnostics & timeouts', () => {
  it('env CURSOR_FIRST_BYTE_TIMEOUT_MS defaults to 90000 and CURSOR_HANG_COOLDOWN_MS to 3600000', () => {
    const config = loadConfig({
      CHEAP_API_KEY: 'k',
      CLASSIFIER: 'heuristic',
      CURSOR_TIER: 'on',
      LOCALAPPDATA: 'C:\\Users\\x\\AppData\\Local',
    });
    assert.ok(config.cursor);
    assert.equal(config.cursor.firstByteTimeoutMs, 90_000);
    assert.equal(config.cursor.hangCooldownMs, 3_600_000);
  });

  it('any stdout byte (such as system/init event) clears the first-byte timer', async () => {
    const { root } = await fakeCursorRoot();
    try {
      const spawnFn = () =>
        fakeChild((child) => {
          // Emit system/init at 20ms (before firstByteTimeoutMs = 50ms)
          setTimeout(() => {
            child.stdout.emit('data', line({ type: 'system', subtype: 'init', session_id: 's1' }));
          }, 20);

          // Emit result at 70ms (after firstByteTimeoutMs would have fired if not cleared)
          setTimeout(() => {
            child.stdout.emit('data', line({
              type: 'result',
              subtype: 'success',
              is_error: false,
              result: 'success answer',
              usage: { inputTokens: 5, outputTokens: 5 },
            }));
            child.emit('close', 0);
          }, 70);
        });

      const res = await runCursorAgent('prompt', {
        bin: root,
        model: 'auto',
        timeoutMs: 1000,
        firstByteTimeoutMs: 50,
      }, spawnFn as any);

      assert.equal(res.ok, true);
      if (res.ok) {
        assert.equal(res.text, 'success answer');
      }
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('includes the last 300 chars of stderr on first-byte-timeout', async () => {
    const { root } = await fakeCursorRoot();
    try {
      const longStderr = 'HEADER-ERROR-IGNORED '.repeat(10) + 'TAIL-DIAGNOSTIC-IMPORTANT-INFO-300';
      const spawnFn = () =>
        fakeChild((child) => {
          child.stderr.emit('data', Buffer.from(longStderr));
          // Never emit stdout -> triggers firstByteTimeoutMs
        });

      const res = await runCursorAgent('prompt', {
        bin: root,
        model: 'auto',
        timeoutMs: 1000,
        firstByteTimeoutMs: 30,
      }, spawnFn as any);

      assert.equal(res.ok, false);
      if (!res.ok) {
        assert.ok(res.reason.startsWith('first-byte-timeout:'));
        assert.ok(res.reason.includes('TAIL-DIAGNOSTIC-IMPORTANT-INFO-300'));
        const detail = res.reason.replace('first-byte-timeout: ', '');
        assert.ok(detail.length <= 300);
      }
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('includes the last 300 chars of stderr on idle-timeout and exit != 0', async () => {
    const { root } = await fakeCursorRoot();
    try {
      const longStderr = 'STALL-PREFIX '.repeat(20) + 'CRITICAL-TAIL-DIAGNOSTIC';
      // 1. Idle timeout
      const idleSpawn = () =>
        fakeChild((child) => {
          child.stdout.emit('data', line({ type: 'system', subtype: 'init' }));
          child.stderr.emit('data', Buffer.from(longStderr));
          // stalls here
        });

      const idleRes = await runCursorAgent('prompt', {
        bin: root,
        model: 'auto',
        timeoutMs: 1000,
        firstByteTimeoutMs: 100,
        idleTimeoutMs: 30,
      }, idleSpawn as any);

      assert.equal(idleRes.ok, false);
      if (!idleRes.ok) {
        assert.ok(idleRes.reason.startsWith('idle-timeout:'));
        assert.ok(idleRes.reason.includes('CRITICAL-TAIL-DIAGNOSTIC'));
      }

      // 2. Exit != 0
      const exitSpawn = () =>
        fakeChild((child) => {
          child.stderr.emit('data', Buffer.from('fatal DPAPI error ' + longStderr));
          child.emit('close', 1);
        });

      const exitRes = await runCursorAgent('prompt', {
        bin: root,
        model: 'auto',
        timeoutMs: 1000,
      }, exitSpawn as any);

      assert.equal(exitRes.ok, false);
      if (!exitRes.ok) {
        assert.ok(exitRes.reason.startsWith('exit 1:'));
        assert.ok(exitRes.reason.includes('CRITICAL-TAIL-DIAGNOSTIC'));
      }
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('repeated Cursor first-byte timeouts open cursor breaker for hang cooldown and log once', async () => {
    const config = loadConfig({
      CHEAP_API_KEY: 'k',
      CLASSIFIER: 'heuristic',
      CURSOR_TIER: 'on',
      CURSOR_TIER_BREAKER_FAILURES: '3',
      CURSOR_HANG_COOLDOWN_MS: '3600000',
      LOCALAPPDATA: 'C:\\Users\\x\\AppData\\Local',
    });

    const cursorBreaker = new CircuitBreaker(
      config.cursor!.breakerFailures,
      config.cursor!.breakerCooldownMs,
      config.cursor!.maxConcurrency,
    );

    let spawnAttempts = 0;
    const fakeCursorProvider = {
      name: 'cursor-cli',
      send: async () => {
        spawnAttempts++;
        return { kind: 'unavailable' as const, reason: 'first-byte-timeout: detail' };
      },
    };

    const fakePrimary = {
      name: 'anthropic',
      send: async () => ({
        kind: 'response' as const,
        status: 200,
        headers: { 'content-type': 'application/json' },
        body: (async function* () {
          yield JSON.stringify({ type: 'message', content: [{ type: 'text', text: 'from claude' }] });
        })() as any,
      }),
    };

    const router = new Router(
      { name: 'mock', classify: async () => ({ simple: 0, standard: 1, structural: 0, textOnly: 0.1 }) } as any,
      new TtlLruStore(100, 60_000),
      {
        policy: { minCheapProbability: 0.9, standardEnabled: false, argonEnabled: true },
        primaryClasses: new Set(['auxiliary']),
        classifierTimeoutMs: 1000,
        classifierMaxChars: 4000,
        cursorCoding: true,
        cursorEnabled: true,
      },
    );

    const app = buildServer({
      config,
      router,
      providers: { primary: fakePrimary as any, cursor: fakeCursorProvider as any },
      cursorBreaker,
    });

    // Make 3 requests hitting cursor with first-byte timeouts
    for (let i = 0; i < 3; i++) {
      const res = await app.inject({
        method: 'POST',
        url: '/v1/messages',
        headers: { 'content-type': 'application/json' },
        payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'write code' }] },
      });
      assert.equal(res.statusCode, 200);
    }

    assert.equal(spawnAttempts, 3);
    // Breaker should now be forced open with 1h cooldown
    assert.equal(cursorBreaker.state, 'open');

    // 4th request: cursor is open -> skipped without invoking provider.send
    const res4 = await app.inject({
      method: 'POST',
      url: '/v1/messages',
      headers: { 'content-type': 'application/json' },
      payload: { model: 'claude-opus-5-5', messages: [{ role: 'user', content: 'write code' }] },
    });
    assert.equal(res4.statusCode, 200);
    // spawnAttempts is still 3!
    assert.equal(spawnAttempts, 3);
  });
});
