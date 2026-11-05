import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { describe, it } from 'node:test';
import { inspectStopContext, onStopSignal, redactCommandLine } from '../src/shutdown-report.js';

// Forensics for unexplained router stops (9 console Ctrl+C since 2026-10-28, sender unknown).
// Windows does not tell a process who raised a console control event, so the context lists the
// parent and every process started in the two minutes before.

describe('inspectStopContext', () => {
  it('returns the parent and recent processes with command lines redacted', async () => {
    const fake = async () =>
      JSON.stringify({
        parent: { ProcessId: 17932, Name: 'cmd.exe', CommandLine: 'cmd /c start-router.cmd' },
        recent: [{ ProcessId: 4242, ParentProcessId: 1, Name: 'powershell.exe', CommandLine: 'powershell -c x --api-key=sk-ant-abcdef1234567890 TOKEN=supersecretvalue' }],
      });
    const r = await inspectStopContext({ inspect: fake, platform: 'win32' });
    assert.ok(r && 'parent' in r);
    assert.equal(r.parent?.Name, 'cmd.exe');
    const cmd = r.recent[0]!.CommandLine!;
    assert.ok(!cmd.includes('sk-ant-abcdef1234567890'), cmd);
    assert.ok(!cmd.includes('supersecretvalue'), cmd);
    assert.match(cmd, /powershell -c x/);
  });

  it('a single recent process (PowerShell serializes it as an object) still becomes a list', async () => {
    const r = await inspectStopContext({ inspect: async () => JSON.stringify({ parent: null, recent: { ProcessId: 1, Name: 'x.exe' } }), platform: 'win32' });
    assert.ok(r && 'recent' in r);
    assert.equal(r.recent.length, 1);
  });

  it('never throws: a failing inspection is reported', async () => {
    const r = await inspectStopContext({ inspect: async () => { throw new Error('boom'); }, platform: 'win32' });
    assert.deepEqual(r, { inspectError: 'boom' });
  });

  it('gives up on a slow inspection', async () => {
    const t0 = Date.now();
    const r = await inspectStopContext({ inspect: () => new Promise(() => {}), platform: 'win32', timeoutMs: 50 });
    assert.ok(Date.now() - t0 < 1_000);
    assert.deepEqual(r, { inspectError: 'inspect timeout' });
  });

  it('is skipped outside Windows', async () => {
    let called = false;
    const r = await inspectStopContext({ inspect: async () => ((called = true), '{}'), platform: 'linux' });
    assert.equal(called, false);
    assert.equal(r, undefined);
  });
});

describe('onStopSignal', () => {
  const harness = (inspect: () => Promise<string>) => {
    const events = new EventEmitter();
    const logs: { msg: string; [k: string]: unknown }[] = [];
    const order: string[] = [];
    let exitCode: number | undefined;
    let releaseStop!: () => void;
    const stopped = new Promise<void>((r) => (releaseStop = r));
    let resolveExit!: () => void;
    const exited = new Promise<void>((r) => (resolveExit = r));
    onStopSignal({
      log: (msg, entry) => { logs.push({ msg, ...entry }); order.push(msg); },
      stop: async () => { order.push('stop'); await stopped; },
      exit: (code) => { exitCode = code; order.push(`exit ${code}`); resolveExit(); },
      on: (signal, handler) => events.on(signal, handler),
      context: { inspect, platform: 'win32', timeoutMs: 200 },
    });
    return { events, logs, order, releaseStop, exited, exitCode: () => exitCode };
  };

  // Regression (review of 4d55721): the shutdown waited up to 5 s for the inspection, so the
  // worker died first on the same console Ctrl+C and the supervisor took it for a crash.
  it('logs the signal and starts stopping at once, without waiting for the inspection', async () => {
    let finishInspect!: (v: string) => void;
    const h = harness(() => new Promise<string>((r) => (finishInspect = r)));
    h.events.emit('SIGINT');
    assert.deepEqual(h.order, ['stop signal received', 'stop'], 'log and stop are synchronous with the signal');
    assert.equal(h.logs[0]!['signal'], 'SIGINT');
    assert.match(String(h.logs[0]!['at']), /^\d{4}-/);
    assert.equal(h.logs[0]!['ppid'], process.ppid);

    h.releaseStop();
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(h.exitCode(), undefined, 'waits for the inspection (bounded) before exiting');
    finishInspect(JSON.stringify({ parent: { Name: 'cmd.exe' }, recent: [] }));
    await h.exited;
    assert.deepEqual(h.order, ['stop signal received', 'stop', 'stop signal context', 'exit 0']);
    assert.equal((h.logs[1]!['parent'] as any).Name, 'cmd.exe');
  });

  it('a second signal while stopping forces the exit; the first line is already in the log', async () => {
    const h = harness(() => new Promise<string>(() => {}));
    h.events.emit('SIGINT');
    h.events.emit('SIGBREAK');
    await h.exited;
    assert.equal(h.exitCode(), 130);
    assert.deepEqual(h.order.slice(0, 2), ['stop signal received', 'stop']);
    assert.ok(h.logs.some((l) => l.msg === 'stop signal repeated: forcing exit' && l['signal'] === 'SIGBREAK'));
  });

  it('exits even when the inspection hangs (bounded by its timeout)', async () => {
    const h = harness(() => new Promise<string>(() => {}));
    h.events.emit('SIGTERM');
    h.releaseStop();
    await h.exited;
    assert.equal(h.exitCode(), 0);
    assert.equal(h.logs.find((l) => l.msg === 'stop signal context')!['inspectError'], 'inspect timeout');
  });
});

describe('redactCommandLine', () => {
  const cases: [string, string][] = [
    ['agent --api-key sk-1234567890abcdef', 'agent --api-key <REDACTED>'],
    ['x PASSWORD=hunter2 y', 'x PASSWORD=<REDACTED> y'],
    ['run AIzaSyA1234567890abcdefghijklmnopqrst', 'run <REDACTED>'],
    // the cases the review found leaking
    ['curl "https://api.x.com/v1?key=abc123&q=1"', 'curl "https://api.x.com/v1?key=<REDACTED>&q=1"'],
    ['curl https://h/x?access_token=t.k+n/=&a=b', 'curl https://h/x?access_token=<REDACTED>&a=b'],
    [`node -e '{"token":"abc","n":1}'`, `node -e '{"token":"<REDACTED>","n":1}'`],
    [`x {'api_key': 'p q r'}`, `x {'api_key': '<REDACTED>'}`],
    ['curl -H "Authorization: Basic dXNlcjpwYXNz"', 'curl -H "Authorization: Basic <REDACTED>"'],
    ['curl -H "Authorization: Bearer abc.def"', 'curl -H "Authorization: Bearer <REDACTED>"'],
    ['tool --token "a b c" --v', 'tool --token "<REDACTED>" --v'],
    ['git clone https://user:s3cr3t@github.com/o/r', 'git clone https://user:<REDACTED>@github.com/o/r'],
    ['x eyJhbGciOi.eyJzdWIiOiIx.c2lnbmF0dXJl y', 'x <REDACTED> y'],
    ['gh ghp_abcdefghijklmnopqrstuvwxyz0123 --x', 'gh <REDACTED> --x'],
    ['x ANTHROPIC_API_KEY=sk-ant-a/b+c=', 'x ANTHROPIC_API_KEY=<REDACTED>'],
  ];
  for (const [input, expected] of cases) {
    it(`redacts: ${input}`, () => assert.equal(redactCommandLine(input), expected));
  }

  it('leaves ordinary command lines alone', () => {
    for (const plain of [
      'node dist/cli.js serve',
      '"C:\\Program Files\\nodejs\\node.exe" C:\\Dev\\jev-router\\node_modules\\tsx\\dist\\cli.mjs',
      'git log --oneline -5',
      'monkey business --keyboard us',
      'npm run build -- --token-count',
    ]) {
      assert.equal(redactCommandLine(plain), plain);
    }
  });

  it('truncates long command lines', () => {
    assert.ok(redactCommandLine('a '.repeat(400)).length <= 240);
  });
});
