import { execFile } from 'node:child_process';
import type { AddressInfo } from 'node:net';
import { runningBuild } from './build-info.js';
import type { ComplexityClassifier } from './classifier/classifier.js';
import { HeuristicClassifier } from './classifier/heuristic-classifier.js';
import { JevClassifier } from './classifier/jev-classifier.js';
import { removedEnvSet, type Config } from './config.js';
import type { Tier } from './domain/policy.js';
import { AnthropicProvider } from './providers/anthropic.js';
import { QuotaStore } from './quota.js';
import { buildServer } from './proxy/server.js';
import { Router } from './routing/router.js';
import { TtlLruStore } from './routing/session-store.js';
import { startSupervisor, type WorkerProcess } from './supervisor.js';
import { openTelemetryDb } from './telemetry/db.js';
import { recordQuota } from './telemetry/schema.js';
import { noopTelemetry, SqliteTelemetry, type TelemetrySink } from './telemetry/recorder.js';

import { ensureAgyHome } from './providers/gemini/isolation.js';
import { GeminiCliProvider } from './providers/gemini/provider.js';
import { CursorCliProvider } from './providers/cursor/provider.js';
import { OpenAICompatibleProvider } from './providers/openai/provider.js';
import { CircuitBreaker } from './providers/breaker.js';
import { ModelQuotaState } from './providers/quota-state.js';

const SECRET_FLAG = /((?:api[_-]?key|token|secret|password|passwd|authorization)(?:\s*[=:]\s*|\s+))(?:bearer\s+)?\S+/gi;
const TOKEN_LIKE = /\b(?:sk-[A-Za-z0-9_-]{8,}|[A-Za-z0-9_-]{32,})\b/g;

/** Command lines go to router.log: mask anything that looks like a credential, then truncate. */
export function redactCommandLine(cmd: string): string {
  return cmd.replace(SECRET_FLAG, '$1<REDACTED>').replace(TOKEN_LIKE, '<REDACTED>').slice(0, 240);
}

/** The parent and every process started in the last 2 minutes, as JSON (Windows only). */
function inspectWindowsProcesses(): Promise<string> {
  const script =
    `$p = Get-CimInstance Win32_Process -Filter 'ProcessId=${process.ppid}' | Select-Object ProcessId,ParentProcessId,Name,CommandLine; ` +
    `$since = (Get-Date).AddSeconds(-120); ` +
    `$r = @(Get-CimInstance Win32_Process | Where-Object { $_.CreationDate -gt $since } | ` +
    `Select-Object ProcessId,ParentProcessId,Name,CommandLine,@{n='Created';e={$_.CreationDate.ToString('o')}}); ` +
    `@{ parent = $p; recent = $r } | ConvertTo-Json -Compress -Depth 3`;
  return new Promise((resolve, reject) => {
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { windowsHide: true, timeout: 4_000 }, (err, stdout) =>
      err ? reject(err) : resolve(stdout),
    );
  });
}

interface ProcessInfo { CommandLine?: string | null; [key: string]: unknown }

/**
 * What to log when the router is asked to stop. Windows does not tell a process who raised a
 * console Ctrl+C, so the parent (the scheduled task's cmd.exe, normally) is only context: the
 * sender, if it is a fresh process, is in `recent`. Never throws and never waits past `timeoutMs`.
 */
export async function shutdownSignalReport(
  signal: string,
  { inspect = inspectWindowsProcesses, platform = process.platform, timeoutMs = 5_000 }: {
    inspect?: () => Promise<string>;
    platform?: NodeJS.Platform;
    timeoutMs?: number;
  } = {},
): Promise<Record<string, unknown>> {
  // No `pid`: every log line already carries it.
  const base = { signal, at: new Date().toISOString(), ppid: process.ppid };
  if (platform !== 'win32') return base;
  try {
    let timer: NodeJS.Timeout | undefined;
    const raw = await Promise.race([
      inspect(),
      new Promise<never>((_, reject) => { timer = setTimeout(() => reject(new Error('inspect timeout')), timeoutMs); }),
    ]).finally(() => clearTimeout(timer));
    const info = JSON.parse(raw) as { parent?: ProcessInfo | null; recent?: ProcessInfo[] | ProcessInfo | null };
    const clean = (p: ProcessInfo): ProcessInfo => ({ ...p, CommandLine: p.CommandLine ? redactCommandLine(p.CommandLine) : p.CommandLine ?? null });
    const recent = info.recent == null ? [] : Array.isArray(info.recent) ? info.recent : [info.recent];
    return { ...base, parent: info.parent ? clean(info.parent) : null, recent: recent.map(clean) };
  } catch (err) {
    return { ...base, inspectError: err instanceof Error ? err.message : String(err) };
  }
}

/** Logs who/what was around, then runs `stop`. One shot per signal, like the handlers it replaces. */
function onStopSignal(log: (entry: Record<string, unknown>) => void, stop: () => Promise<void>): void {
  for (const signal of ['SIGINT', 'SIGTERM', 'SIGBREAK'] as const) {
    process.once(signal, () => {
      void shutdownSignalReport(signal)
        .then((report) => log(report))
        .finally(() => void stop().then(() => process.exit(0)));
    });
  }
}

const SESSION_MAX_ENTRIES = 10_000;

export async function serve(config: Config, { worker = false } = {}): Promise<void> {
  let geminiConfig = config.gemini;
  if (geminiConfig) {
    try {
      await ensureAgyHome(geminiConfig.home);
    } catch (err) {
      console.warn(`Failed to initialize Gemini tier isolation (ensureAgyHome): ${err instanceof Error ? err.message : String(err)}. Gemini tier disabled.`);
      geminiConfig = undefined;
    }
  }

  const classifier: ComplexityClassifier =
    config.classifier.kind === 'jev'
      ? new JevClassifier({
          apiUrl: config.classifier.apiUrl,
          apiKey: config.classifier.apiKey,
          model: config.classifier.model,
          geminiEnabled: geminiConfig !== undefined,
        })
      : new HeuristicClassifier();

  let logTelemetryError: (err: unknown) => void = () => {};
  const db = config.telemetry.dbPath ? openTelemetryDb(config.telemetry.dbPath) : undefined;
  const telemetry: TelemetrySink = db ? new SqliteTelemetry(db, { onError: (err) => logTelemetryError(err) }) : noopTelemetry;

  const quota = config.quota.enabled
    ? new QuotaStore(config.quota, (snapshot) => {
        if (!db) return;
        try {
          recordQuota(db, snapshot);
        } catch (err) {
          logTelemetryError(err);
        }
      })
    : undefined;

  const router = new Router(
    classifier,
    new TtlLruStore<Tier>(SESSION_MAX_ENTRIES, config.router.sessionTtlMs),
    {
      policy: {
        minCheapProbability: config.router.minCheapProbability,
        minStandardProbability: config.router.minStandardProbability,
        argonEnabled: config.argon.enabled,
        // Argon owns the middle band; otherwise simple+standard mass sends medium/coding prompts to Flash.
        standardEnabled: !config.argon.enabled,
      },
      primaryClasses: config.router.primaryClasses,
      classifierTimeoutMs: config.classifier.timeoutMs,
      classifierMaxChars: config.classifier.maxChars,
      sessionTtlMs: config.router.sessionTtlMs,
      ...(quota
        ? {
            quota,
            pressurePolicy: {
              minCheapProbability: config.router.minCheapProbability,
              minStandardProbability: config.quota.minStandardProbability,
            },
          }
        : {}),
      ...(geminiConfig
        ? {
            geminiPolicy: {
              enabled: true,
              minTextOnly: geminiConfig.minTextOnly,
              pressureMinTextOnly: geminiConfig.pressureMinTextOnly,
              replacesCheap: true,
            },
            geminiFromPrimary: geminiConfig.fromPrimary,
          }
        : {}),
      primaryProfiles: config.primaryProfiles,
      argonPolicy: config.argon,
      cursorCoding: config.cursorCoding,
      cursorCodingMaxTextOnly: config.cursorCodingMaxTextOnly,
      cursorCodingMinMass: config.cursorCodingMinMass,
      cursorEnabled: config.cursor !== undefined,
    },
  );

  const geminiBreaker = geminiConfig
    ? new CircuitBreaker(geminiConfig.breakerFailures, geminiConfig.breakerCooldownMs, geminiConfig.maxConcurrency)
    : undefined;
  const argonBreaker = geminiConfig
    ? new CircuitBreaker(geminiConfig.breakerFailures, geminiConfig.breakerCooldownMs, geminiConfig.maxConcurrency)
    : undefined;
  const cursorBreaker = config.cursor
    ? new CircuitBreaker(config.cursor.breakerFailures, config.cursor.breakerCooldownMs, config.cursor.maxConcurrency)
    : undefined;

  const providers = {
    primary: new AnthropicProvider(config.primary),
    ...(geminiConfig
      ? {
          gemini: new GeminiCliProvider({
            ...geminiConfig,
            firstByteTimeoutMs: config.timeouts.firstByteMs,
            idleTimeoutMs: config.timeouts.idleMs,
            argonModels: [config.argon.modelLow, config.argon.modelMedium, config.argon.modelHigh],
            argonFirstByteTimeoutMs: config.argon.firstByteMs,
            argonIdleTimeoutMs: config.argon.idleMs,
          }),
        }
      : {}),
    ...(config.cursor
      ? {
          cursor: new CursorCliProvider({
            ...config.cursor,
            firstByteTimeoutMs: config.cursor.firstByteTimeoutMs,
            idleTimeoutMs: config.timeouts.idleMs,
          }),
        }
      : {}),
    ...(config.cheap ? { cheap: new OpenAICompatibleProvider(config.cheap) } : {}),
  };

  const quotaState = new ModelQuotaState();
  const build = runningBuild();
  const app = buildServer({
    config,
    router,
    providers,
    telemetry,
    quotaState,
    ...(build ? { build } : {}),
    ...(quota ? { quota } : {}),
    ...(geminiBreaker ? { geminiBreaker } : {}),
    ...(argonBreaker ? { argonBreaker } : {}),
    ...(cursorBreaker ? { cursorBreaker } : {}),
    timeouts: config.timeouts,
  });
  logTelemetryError = (err) => app.log.error({ err }, 'telemetry write failed');
  app.addHook('onClose', () => telemetry.close());

  await app.listen(worker ? { host: '127.0.0.1', port: 0 } : { host: config.host, port: config.port });
  if (worker) {
    process.send?.({ type: 'listening', port: (app.server.address() as AddressInfo).port });
    process.on('message', (msg: { type?: string } | null) => {
      if (msg?.type === 'drain') void app.close().then(() => process.exit(0));
    });
    process.once('disconnect', () => void app.close().then(() => process.exit(0)));
  }
  app.log.info(
    {
      classifier: classifier.name,
      build: build?.sha ?? 'src',
      ...(worker ? { worker: true } : {}),
      primary: config.primary.baseUrl,
      argon: config.argon.enabled ? config.argon.modelMedium : 'off',
      cursor: config.cursor ? config.cursor.model : 'off',
      cheap: config.cheap ? `${config.cheap.model} (${config.cheap.position})` : 'off',
      telemetry: config.telemetry.dbPath ?? 'disabled',
    },
    'jev-router ready',
  );
  const stale = removedEnvSet();
  if (stale.length > 0) app.log.warn({ stale }, 'ignored: these env vars were removed, delete them from .env');

  if (worker) {
    // The supervisor logs the signal (it receives the same console Ctrl+C); the worker just stops.
    for (const signal of ['SIGINT', 'SIGTERM'] as const) {
      process.once(signal, () => {
        void app.close().then(() => process.exit(0));
      });
    }
  } else {
    onStopSignal((report) => app.log.warn(report, 'stop signal received'), () => app.close());
  }
}

export async function supervise(config: Config, forkWorker: () => WorkerProcess): Promise<void> {
  const supervisor = await startSupervisor({
    host: config.host,
    port: config.port,
    controlPort: config.supervisor.controlPort,
    drainMs: config.supervisor.drainMs,
    forkWorker,
  });
  onStopSignal(
    (report) => console.log(JSON.stringify({ level: 40, time: Date.now(), pid: process.pid, name: 'supervisor', msg: 'stop signal received', ...report })),
    () => supervisor.close(),
  );
}
