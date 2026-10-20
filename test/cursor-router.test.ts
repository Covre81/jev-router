import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { loadConfig } from '../src/config.js';
import { runCursorAgent } from '../src/providers/cursor/agent-runner.js';
import { CursorCliProvider } from '../src/providers/cursor/provider.js';

describe('cursor config', () => {
  it('off by default; on gives verified defaults', () => {
    assert.equal(loadConfig({ CHEAP_API_KEY: 'k', CLASSIFIER: 'heuristic' }).cursor, undefined);
    const c = loadConfig({ CHEAP_API_KEY: 'k', CLASSIFIER: 'heuristic', CURSOR_TIER: 'on', LOCALAPPDATA: 'C:\\Users\\x\\AppData\\Local' });
    assert.ok(c.cursor);
    assert.equal(c.cursor.model, 'auto');
    assert.equal(c.cursor.timeoutMs, 180_000);
    assert.equal(c.cursor.maxPromptChars, 24_000);
    assert.equal(c.cursor.maxConcurrency, 1);
    assert.equal(c.cursor.breakerFailures, 3);
    assert.equal(c.cursor.breakerCooldownMs, 300_000);
    assert.ok(c.cursor.bin.endsWith('cursor-agent'));
  });
});

describe('router fallback-loop stickiness', () => {
  function setup(quotaCurrent: () => any, classification: any = { simple: 0, standard: 0, structural: 1, risk: 0.1, textOnly: 0.1 }, sessionTtlMs?: number) {
    const classifier = { name: 'test', calls: 0, classify: async () => { classifier.calls++; return classification; } };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.8, standardRoute: 'primary', standardEnabled: false },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      ...(sessionTtlMs !== undefined ? { sessionTtlMs } : {}),
      quota: { level: () => 'critical', current: quotaCurrent },
    } as any);
    return { router, store, classifier };
  }
  const req = (body: any, sid = 's1') =>
    ({ body, rawByteLength: 100, sessionId: sid, agentId: undefined, requestClass: undefined, contextCompacted: false }) as any;
  const humanTurn = { messages: [{ role: 'user', content: 'hi' }, { role: 'assistant', content: 'ok' }, { role: 'user', content: 'again' }] };
  const toolTurn = {
    messages: [
      { role: 'user', content: 'hi' },
      { role: 'assistant', content: [{ type: 'tool_use', id: '1', name: 'Read', input: {} }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: '1', content: 'ok' }] },
    ],
  };

  it('6. human turn before 5h resetAt stays on fallback tier; after resetAt clears and retries primary', async () => {
    let snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router } = setup(() => snap);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');

    const before = await router.decide(req(humanTurn));
    assert.equal(before.route, 'cursor');
    assert.equal(before.tier, 'cursor');
    assert.equal(before.reason, 'sticky');

    snap = { utilization: 1, window: '5h', resetAt: new Date(Date.now() - 1000), observedAt: new Date() };
    const after = await router.decide(req(humanTurn));
    assert.equal(after.route, 'primary');

    // mark was cleared: a tool continuation no longer sticks to cursor
    const cont = await router.decide(req(toolTurn));
    assert.notEqual(cont.route, 'cursor');
  });

  it('6b. 7d window with utilization 1 and future reset holds the fallback tier on a human turn', async () => {
    const snap: any = { utilization: 1, window: '7d', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router } = setup(() => snap);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'cursor');
    assert.equal(dec.tier, 'cursor');
  });

  it('6c. 5h window at low utilization with future resetAt (transient 529) clears the loop and retries primary', async () => {
    const snap: any = { utilization: 0.2, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router } = setup(() => snap);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'primary');
    // loop cleared: a tool continuation no longer sticks to cursor
    const cont = await router.decide(req(toolTurn));
    assert.notEqual(cont.route, 'cursor');
  });

  it('risk veto: loop human turn with high security risk => primary and loop cleared', async () => {
    const snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const risky = { simple: 0.99, standard: 0, structural: 0.01, risk: 0.95, riskScores: { security_sensitive: 0.95, destructive_or_production: 0, requires_inspection: 0 }, textOnly: 0.1 };
    const { router, classifier } = setup(() => snap, risky);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'primary');
    assert.equal(dec.tier, 'primary');
    assert.ok(classifier.calls >= 1);
    const cont = await router.decide(req(toolTurn));
    assert.notEqual(cont.route, 'cursor');
  });

  it('benign loop human turn (classified, not risky) stays on loop tier while 5h gate holds', async () => {
    const snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const benign = { simple: 0.99, standard: 0, structural: 0.01, risk: 0.05, riskScores: { security_sensitive: 0.05, destructive_or_production: 0, requires_inspection: 0 }, textOnly: 0.1 };
    const { router, classifier } = setup(() => snap, benign);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'cursor');
    assert.equal(dec.tier, 'cursor');
    assert.ok(classifier.calls >= 1);
  });

  it('no quota snapshot does not drop the loop on a human turn', async () => {
    const { router } = setup(() => undefined);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'cursor');
  });

  it('quota probe inside a marked loop is never routed to the loop tier', async () => {
    const snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router } = setup(() => snap);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    const dec = await router.decide(req({ max_tokens: 1, messages: [{ role: 'user', content: 'hi' }] }));
    assert.notEqual(dec.route, 'cursor');
    assert.equal(dec.route, 'primary');
  });

  it('fallback-loop mark older than the session TTL does not stick (tool continuation)', async () => {
    const { router } = setup(() => undefined, undefined, 60_000);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'cursor');
    // Age the mark past the TTL (lazy prune on read, no timers).
    const entry = (router as any).fallbackLoop.get('s1:main');
    if (entry && typeof entry === 'object') entry.markedAt = Date.now() - 61_000;
    else (router as any).fallbackLoop.set('s1:main', { tier: 'cursor', markedAt: Date.now() - 61_000 });
    const dec = await router.decide(req(toolTurn));
    assert.notEqual(dec.route, 'cursor');
    assert.equal(dec.route, 'primary');
    assert.equal((router as any).fallbackLoop.get('s1:main'), undefined);
  });

  it('tool-result continuation sticks to the marked tier without classification', async () => {
    const { router } = setup(() => undefined);
    (router as any).markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'gemini');
    const dec = await router.decide(req(toolTurn));
    assert.equal(dec.route, 'gemini');
    assert.equal(dec.reason, 'sticky');
  });

  it('cascade cheap failover: tool continuation routes to cheap (trivial) without demoting primary session', async () => {
    const snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router, store } = setup(() => snap);
    store.set('s1:main', 'primary');
    router.markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'trivial');
    assert.equal(store.get('s1:main'), 'primary');
    const dec = await router.decide(req(toolTurn));
    assert.equal(dec.route, 'cheap');
    assert.equal(dec.tier, 'trivial');
    assert.equal(dec.fallbackLoop, 'trivial');
    assert.equal(store.get('s1:main'), 'primary');
  });

  it('cascade cheap failover: non-exhausted quota releases next human turn to primary and clears loop', async () => {
    const snap: any = { utilization: 0.2, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router, store } = setup(() => snap);
    store.set('s1:main', 'primary');
    router.markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'trivial');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'primary');
    assert.equal(dec.tier, 'primary');
    assert.equal(store.get('s1:main'), 'primary');
    // loop cleared: a tool continuation no longer sticks to cheap
    const cont = await router.decide(req(toolTurn));
    assert.notEqual(cont.route, 'cheap');
    assert.notEqual(cont.tier, 'trivial');
  });

  it('cascade cheap failover: exhausted quota keeps next human turn on cheap', async () => {
    const snap: any = { utilization: 1, window: '5h', resetAt: new Date(Date.now() + 60_000), observedAt: new Date() };
    const { router, store } = setup(() => snap);
    store.set('s1:main', 'primary');
    router.markFallbackLoop({ route: 'primary', reason: 'classified', conversationKey: 's1:main' }, 'trivial');
    const dec = await router.decide(req(humanTurn));
    assert.equal(dec.route, 'cheap');
    assert.equal(dec.tier, 'trivial');
    assert.equal(dec.fallbackLoop, 'trivial');
    assert.equal(store.get('s1:main'), 'primary');
  });
});

async function fakeCursorRoot() {
  const root = await mkdtemp(join(tmpdir(), 'jev-cursor-root-'));
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
  child.pid = 4242;
  setTimeout(() => script(child), 5);
  return child;
}

const line = (o: unknown) => Buffer.from(JSON.stringify(o) + '\n');

describe('cursor runner/provider units', () => {
  it('7a2. strips MSYSTEM/EXEPATH so Cursor runs hooks with PowerShell, not Git Bash', async () => {
    const { root } = await fakeCursorRoot();
    let seen: any;
    process.env.MSYSTEM = 'MINGW64';
    process.env.EXEPATH = 'C:\\Program Files\\Git\\bin';
    try {
      const spawnFn = (_bin: string, _args: string[], opts: any) => {
        seen = opts;
        return fakeChild((child) => {
          child.stdout.emit('data', line({ type: 'result', subtype: 'success', is_error: false, result: 'ok' }));
          child.emit('close', 0);
        });
      };
      await runCursorAgent('p', { bin: root, model: 'auto', timeoutMs: 1000 }, spawnFn as any);
      assert.ok(!('MSYSTEM' in seen.env));
      assert.ok(!('EXEPATH' in seen.env));
    } finally {
      delete process.env.MSYSTEM;
      delete process.env.EXEPATH;
      await rm(root, { recursive: true, force: true });
    }
  });

  it('7a. success from stream-json events; strips CURSOR_API_KEY, prompt is last argv', async () => {
    const { root, ver } = await fakeCursorRoot();
    let seen: any;
    process.env.CURSOR_API_KEY = 'must-not-leak';
    process.env.ANTHROPIC_API_KEY = 'must-not-leak';
    process.env.TYPESAFE_API_KEY = 'must-not-leak';
    process.env.PROXY_AUTH_TOKEN = 'must-not-leak';
    process.env.CHEAP_API_KEY = 'must-not-leak';
    try {
      const spawnFn = (bin: string, args: string[], opts: any) => {
        seen = { bin, args, opts };
        return fakeChild((child) => {
          child.stdout.emit('data', line({ type: 'system', subtype: 'init', session_id: 's' }));
          child.stdout.emit('data', line({ type: 'assistant', timestamp_ms: 1, message: { content: [{ type: 'text', text: 'hello ' }] } }));
          child.stdout.emit('data', line({ type: 'assistant', timestamp_ms: 2, message: { content: [{ type: 'text', text: 'world' }] } }));
          child.stdout.emit('data', line({ type: 'assistant', message: { content: [{ type: 'text', text: 'hello world' }] } }));
          child.stdout.emit('data', line({ type: 'result', subtype: 'success', is_error: false, result: 'hello world', session_id: 's', usage: { inputTokens: 10, outputTokens: 2, cacheReadTokens: 3, cacheWriteTokens: 0 } }));
          child.emit('close', 0);
        });
      };
      const res = await runCursorAgent('the prompt', { bin: root, model: 'auto', timeoutMs: 1000 }, spawnFn as any);
      assert.deepEqual(res, { ok: true, text: 'hello world', usage: { inputTokens: 10, outputTokens: 2 } });
      assert.equal(seen.bin, join(ver, 'node.exe'));
      assert.equal(seen.args[0], join(ver, 'index.js'));
      assert.ok(seen.args.includes('-p'));
      assert.ok(seen.args.includes('--mode') && seen.args[seen.args.indexOf('--mode') + 1] === 'ask');
      assert.ok(seen.args.includes('--trust'));
      assert.ok(seen.args.includes('--output-format') && seen.args[seen.args.indexOf('--output-format') + 1] === 'stream-json');
      assert.ok(seen.args.includes('--model') && seen.args[seen.args.indexOf('--model') + 1] === 'auto');
      assert.equal(seen.args.at(-1), 'the prompt');
      assert.ok(!('CURSOR_API_KEY' in seen.opts.env));
      assert.ok(!('ANTHROPIC_API_KEY' in seen.opts.env));
      assert.ok(!('TYPESAFE_API_KEY' in seen.opts.env));
      assert.ok(!('PROXY_AUTH_TOKEN' in seen.opts.env));
      assert.ok(!('CHEAP_API_KEY' in seen.opts.env));
      assert.ok(!seen.args.includes('--yolo'));
      assert.ok(!seen.args.includes('-f'));
      assert.ok(!seen.opts.shell);
    } finally {
      delete process.env.CURSOR_API_KEY;
      delete process.env.ANTHROPIC_API_KEY;
      delete process.env.TYPESAFE_API_KEY;
      delete process.env.PROXY_AUTH_TOKEN;
      delete process.env.CHEAP_API_KEY;
      await rm(root, { recursive: true, force: true });
    }
  });

  it('7e. already-aborted signal => no spawn, reason aborted', async () => {
    const { root } = await fakeCursorRoot();
    let spawned = 0;
    try {
      const controller = new AbortController();
      controller.abort();
      const res = await runCursorAgent('p', { bin: root, model: 'auto', timeoutMs: 1000, signal: controller.signal }, ((...a: any[]) => { spawned++; return fakeChild(() => {}); }) as any);
      assert.equal(res.ok, false);
      assert.ok(!res.ok && res.reason === 'aborted');
      assert.equal(spawned, 0);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('7b. tool_call started => tool attempt:', async () => {
    const { root } = await fakeCursorRoot();
    try {
      const spawnFn = () =>
        fakeChild((child) => {
          child.stdout.emit('data', line({ type: 'tool_call', subtype: 'started', tool_call: { readToolCall: { args: {} } } }));
          child.emit('close', 1);
        });
      const res = await runCursorAgent('p', { bin: root, model: 'auto', timeoutMs: 1000 }, spawnFn as any);
      assert.equal(res.ok, false);
      assert.ok(!res.ok && res.reason.startsWith('tool attempt:'));
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('7c. exit 1 / no result => unavailable; missing layout => unavailable without spawning', async () => {
    const { root } = await fakeCursorRoot();
    try {
      const spawnFn = () =>
        fakeChild((child) => {
          child.stderr.emit('data', Buffer.from('unknown model'));
          child.emit('close', 1);
        });
      const res = await runCursorAgent('p', { bin: root, model: 'auto', timeoutMs: 1000 }, spawnFn as any);
      assert.equal(res.ok, false);
      assert.ok(!res.ok && res.reason.includes('exit 1'));

      let spawned = 0;
      const missing = await runCursorAgent('p', { bin: join(root, 'nope'), model: 'auto', timeoutMs: 1000 }, ((...a: any[]) => { spawned++; return fakeChild(() => {}); }) as any);
      assert.equal(missing.ok, false);
      assert.equal(spawned, 0);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('7d. provider: prompt over budget => unavailable without spawning; success synthesizes Anthropic JSON', async () => {
    const { root } = await fakeCursorRoot();
    let spawned = 0;
    try {
      const spawnFn = () => {
        spawned++;
        return fakeChild((child) => {
          child.stdout.emit('data', line({ type: 'result', subtype: 'success', is_error: false, result: 'cursor answer', usage: { inputTokens: 4, outputTokens: 5 } }));
          child.emit('close', 0);
        });
      };
      const p = new CursorCliProvider({ bin: root, model: 'auto', timeoutMs: 1000, maxPromptChars: 50, spawnFn } as any);
      const over = await p.send({ body: { model: 'claude', messages: [{ role: 'user', content: 'x'.repeat(500) }] } } as any);
      assert.deepEqual(over, { kind: 'unavailable', reason: 'transcript budget exceeded' });
      assert.equal(spawned, 0);

      const p2 = new CursorCliProvider({ bin: root, model: 'auto', timeoutMs: 1000, maxPromptChars: 100_000, spawnFn } as any);
      const ok = await p2.send({ body: { model: 'claude', messages: [{ role: 'user', content: 'hi' }] } } as any);
      assert.equal(ok.kind, 'response');
      let s = '';
      for await (const c of (ok as any).body) s += c.toString();
      const parsed = JSON.parse(s);
      assert.equal(parsed.content[0].text, 'cursor answer');
      assert.deepEqual(parsed.usage, { input_tokens: 4, output_tokens: 5 });
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });

  it('7e. provider: a transcript that would drop history declines without spawning (Gemini keeps it whole)', async () => {
    const { root } = await fakeCursorRoot();
    let spawned = 0;
    try {
      const spawnFn = () => {
        spawned++;
        return fakeChild((child) => {
          child.stdout.emit('data', line({ type: 'result', subtype: 'success', is_error: false, result: 'cursor answer' }));
          child.emit('close', 0);
        });
      };
      const p = new CursorCliProvider({ bin: root, model: 'auto', timeoutMs: 1000, maxPromptChars: 2_000, spawnFn } as any);
      const messages = [
        { role: 'user', content: 'x'.repeat(3_000) },
        { role: 'assistant', content: 'ok' },
        { role: 'user', content: 'and now?' },
      ];
      const cut = await p.send({ body: { model: 'claude', messages } } as any);
      assert.deepEqual(cut, { kind: 'unavailable', reason: 'transcript budget exceeded' });
      assert.equal(spawned, 0);

      const whole = await p.send({ body: { model: 'claude', messages: messages.slice(1) } } as any);
      assert.equal(whole.kind, 'response');
      assert.equal(spawned, 1);
    } finally {
      await rm(root, { recursive: true, force: true });
    }
  });
});
