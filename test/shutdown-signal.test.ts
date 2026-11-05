import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { redactCommandLine, shutdownSignalReport } from '../src/serve.js';

// Forensics for unexplained router stops (9 console Ctrl+C since 2026-10-28, sender unknown):
// Windows does not tell a process who raised a console control event, so the log records the
// parent and every process started in the two minutes before, where the sender shows up.

describe('shutdownSignalReport', () => {
  it('records signal, time, pid, the parent and recent processes, with command lines redacted', async () => {
    const fake = async () =>
      JSON.stringify({
        parent: { ProcessId: 17932, Name: 'cmd.exe', CommandLine: 'cmd /c start-router.cmd' },
        recent: [{ ProcessId: 4242, ParentProcessId: 1, Name: 'powershell.exe', CommandLine: 'powershell -c x --api-key=sk-ant-abcdef1234567890 TOKEN=supersecretvalue' }],
      });
    const r = await shutdownSignalReport('SIGINT', { inspect: fake, platform: 'win32' });
    assert.equal(r['signal'], 'SIGINT');
    assert.equal('pid' in r, false); // the log line carries it already
    assert.equal(r['ppid'], process.ppid);
    assert.match(String(r['at']), /^\d{4}-\d\d-\d\dT/);
    assert.equal((r['parent'] as any).Name, 'cmd.exe');
    const cmd = (r['recent'] as any[])[0].CommandLine as string;
    assert.ok(!cmd.includes('sk-ant-abcdef1234567890'), cmd);
    assert.ok(!cmd.includes('supersecretvalue'), cmd);
    assert.match(cmd, /powershell -c x/);
  });

  it('never throws: a failing inspection is reported and shutdown can go on', async () => {
    const r = await shutdownSignalReport('SIGINT', { inspect: async () => { throw new Error('boom'); }, platform: 'win32' });
    assert.equal(r['signal'], 'SIGINT');
    assert.match(String(r['inspectError']), /boom/);
  });

  it('gives up on a slow inspection instead of holding the shutdown', async () => {
    const t0 = Date.now();
    const r = await shutdownSignalReport('SIGINT', { inspect: () => new Promise(() => {}), platform: 'win32', timeoutMs: 50 });
    assert.ok(Date.now() - t0 < 1_000);
    assert.match(String(r['inspectError']), /timeout/);
  });

  it('skips the Windows inspection elsewhere', async () => {
    let called = false;
    const r = await shutdownSignalReport('SIGTERM', { inspect: async () => ((called = true), '{}'), platform: 'linux' });
    assert.equal(called, false);
    assert.equal(r['signal'], 'SIGTERM');
  });
});

describe('redactCommandLine', () => {
  it('masks key=value secrets, bearer tokens and long token-like strings, and truncates', () => {
    assert.equal(redactCommandLine('agent --api-key sk-1234567890abcdef'), 'agent --api-key <REDACTED>');
    assert.equal(redactCommandLine('curl -H "Authorization: Bearer abc.def"'), 'curl -H "Authorization: <REDACTED>');
    assert.equal(redactCommandLine('x PASSWORD=hunter2 y'), 'x PASSWORD=<REDACTED> y');
    assert.equal(redactCommandLine('run AIzaSyA1234567890abcdefghijklmnopqrst'), 'run <REDACTED>');
    assert.equal(redactCommandLine('node dist/cli.js serve'), 'node dist/cli.js serve');
    assert.equal(redactCommandLine('a'.repeat(5) + ' ' + 'b '.repeat(200)).length <= 240, true);
  });
});
