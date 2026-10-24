#!/usr/bin/env node
import { fork } from 'node:child_process';
import { existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { controlPort, loadConfig } from './config.js';
import { loadEnv } from './env.js';
import { defaultTelemetryDbPath } from './paths.js';

const USAGE = `Usage: jev-router [command] [options]

Commands:
  serve            Start the gateway (default)
  reload           Swap in a freshly built worker without dropping open sessions
  stats           Print routing and token-savings statistics from the telemetry database
  statusline       One line for Claude Code's statusLine: router health and this session's last route
  context          Evaluate and inject ai-memory / graphify context for prompts
  agent-hook       SessionStart/SubagentStart/SubagentStop/SessionEnd hook: tracks running subagents for the status line
  help             Show this message

Options:
  --env <file>     Env file to load (default: ./.env, else ~/.jev-router/.env)
  --db <file>      stats: telemetry database (default: TELEMETRY_DB_PATH or ~/.jev-router/telemetry.db)
  --since <span>   stats: only the last <n>h / <n>d / <n>m (e.g. 24h, 7d)
  --json           stats: machine-readable output
  --by-profile     stats: main-loop calls per session, cache hit and USD by served model + effort
                   statusline: {"text","level"}, level ok | warn | critical

Variables already set in the shell win over the env file.`;

async function main(argv: string[]): Promise<number> {
  const { positionals, values } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      env: { type: 'string' },
      db: { type: 'string' },
      since: { type: 'string' },
      json: { type: 'boolean', default: false },
      'by-class': { type: 'boolean', default: false },
      'cache-misses': { type: 'boolean', default: false },
      'min-write': { type: 'string' },
      daily: { type: 'boolean', default: false },
      'by-profile': { type: 'boolean', default: false },
      help: { type: 'boolean', short: 'h', default: false },
    },
  });
  const command = positionals[0] ?? 'serve';
  if (values.help || command === 'help') {
    console.log(USAGE);
    return 0;
  }

  // Snapshot before the env file lands in process.env: a reloaded worker must
  // read the file afresh, and loadEnvFile never overwrites inherited values.
  const shellEnv = { ...process.env };
  const envFile = loadEnv(values.env);

  switch (command) {
    case 'serve': {
      const config = loadConfig();
      const { serve, supervise } = await import('./serve.js');
      if (process.env['JEV_ROUTER_WORKER'] === '1') {
        await serve(config, { worker: true });
      } else if (config.supervisor.enabled) {
        await supervise(config, () =>
          fork(fileURLToPath(import.meta.url), ['serve', ...(envFile ? ['--env', envFile] : [])], {
            env: { ...shellEnv, JEV_ROUTER_WORKER: '1' },
            // `tsx watch` / `node --test` flags would make the worker a watcher or a test run.
            execArgv: process.execArgv.filter((a) => !/^--(watch|test)/.test(a)),
          }),
        );
      } else {
        await serve(config);
      }
      return -1; // keep running
    }
    case 'reload': {
      // Only the control port is needed: works without provider keys.
      const url = `http://127.0.0.1:${controlPort()}/reload`;
      let res: Response;
      try {
        res = await fetch(url, { method: 'POST', signal: AbortSignal.timeout(60_000) });
      } catch {
        console.error(`No supervisor at ${url}: is the router running with SUPERVISOR_ENABLED=true?`);
        return 1;
      }
      console.log(await res.text());
      return res.ok ? 0 : 1;
    }
    case 'stats': {
      // stats must work without provider keys: read only what it needs.
      const dbPath = values.db ?? process.env['TELEMETRY_DB_PATH'] ?? defaultTelemetryDbPath();
      if (!existsSync(dbPath)) {
        console.error(`No telemetry database at ${dbPath}. Start the gateway with \`jev-router serve\` first.`);
        return 1;
      }
      const [{ openTelemetryDb }, statsModule, { pricingFromEnv }] = await Promise.all([
        import('./telemetry/db.js'),
        import('./telemetry/stats.js'),
        import('./telemetry/pricing.js'),
      ]);
      // Writable on purpose: applies pending migrations if the gateway has not run since an upgrade.
      const db = openTelemetryDb(dbPath);
      try {
        const pricing = pricingFromEnv();
        const opts = values.since ? { since: statsModule.parseSince(values.since) } : {};
        if (values['by-class']) {
          const stats = statsModule.computeStatsByClass(db, pricing, opts);
          console.log(values.json ? JSON.stringify(stats, null, 2) : statsModule.renderStatsByClass(stats));
        } else if (values['cache-misses']) {
          const missOpts: any = { ...opts };
          if (values['min-write'] !== undefined) {
            if (!/^[1-9]\d*$/.test(values['min-write']!)) {
              console.error('--min-write must be a positive integer.');
              return 1;
            }
            missOpts.minWrite = parseInt(values['min-write']!, 10);
          }
          const stats = statsModule.computeCacheMisses(db, pricing, missOpts);
          console.log(values.json ? JSON.stringify(stats, null, 2) : statsModule.renderCacheMisses(stats));
        } else if (values['by-profile']) {
          const stats = statsModule.computeProfileStats(db, pricing, opts);
          console.log(values.json ? JSON.stringify(stats, null, 2) : statsModule.renderProfileStats(stats));
        } else if (values.daily) {
          const stats = statsModule.computeDailyStats(db, pricing, opts);
          console.log(values.json ? JSON.stringify(stats, null, 2) : statsModule.renderDailyStats(stats));
        } else {
          const stats = statsModule.computeStats(db, { ...opts, pricing });
          console.log(values.json ? JSON.stringify(stats, null, 2) : statsModule.renderStats(stats));
        }
      } finally {
        db.close();
      }
      return 0;
    }
    case 'context': {
      const { runContextHook } = await import('./context/command.js');
      const code = await runContextHook(values.env);
      // Let Node exit on its own: a process.exit right after stdin closes trips a libuv
      // assertion on Windows (UV_HANDLE_CLOSING). If anything still holds the loop, such as
      // a JEV request the API never answers, force the exit shortly after; unref'd, so a
      // clean run never waits for it.
      process.exitCode = code;
      setTimeout(() => process.exit(code), 250).unref();
      return -1;
    }
    case 'agent-hook': {
      const [{ readHookInput }, { recordAgentEvent }] = await Promise.all([import('./hook-input.js'), import('./agents.js')]);
      try {
        // Claude Code gives the hook 5 s (settings.json): a missed SubagentStop would count a dead agent until SessionEnd.
        const input = await readHookInput(process.stdin, 3_000);
        if (input) recordAgentEvent(input);
      } catch {
        // A hook that fails only costs the ⏳ count, never the session.
      }
      // Same reason as context: no process.exit right after stdin closes; force it only if stdin never closed.
      setTimeout(() => process.exit(0), 250).unref();
      return 0;
    }
    case 'statusline': {
      const [{ readSessionId, statusLineData }, { renderStatusLine, statusLineReport }, { readBuildInfo }] = await Promise.all([
        import('./statusline.js'),
        import('./telemetry/statusline.js'),
        import('./build-info.js'),
      ]);
      const sessionId = await readSessionId();
      const localBuild = readBuildInfo();
      const data = await statusLineData({
        ...(localBuild ? { localBuild } : {}),
        dbPath: values.db ?? process.env['TELEMETRY_DB_PATH'] ?? defaultTelemetryDbPath(),
        healthUrl: `http://${process.env['HOST'] || '127.0.0.1'}:${process.env['PORT'] || '8787'}/healthz`,
        ...(sessionId ? { sessionId } : {}),
        // Opt-in: Claude Code's own status line already shows the plan's 5h/7d usage.
        ...(process.env['STATUSLINE_QUOTA'] === '1' ? { showQuota: true } : {}),
      });
      const line = values.json ? JSON.stringify(statusLineReport(data)) : renderStatusLine(data);
      // Exit once written: a stdin that never closed must not keep the status line hanging.
      process.stdout.write(`${line}\n`, () => process.exit(0));
      return -1;
    }
    default:
      console.error(`Unknown command "${command}".\n\n${USAGE}`);
      return 2;
  }
}

main(process.argv.slice(2)).then(
  (code) => {
    if (code >= 0) process.exitCode = code;
  },
  (err: unknown) => {
    console.error(err instanceof Error ? err.message : err);
    process.exitCode = 1;
  },
);
