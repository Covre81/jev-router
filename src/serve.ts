import type { AddressInfo } from 'node:net';
import { runningBuild } from './build-info.js';
import type { ComplexityClassifier } from './classifier/classifier.js';
import { HeuristicClassifier } from './classifier/heuristic-classifier.js';
import { JevClassifier } from './classifier/jev-classifier.js';
import { removedEnvSet, type Config } from './config.js';
import type { Tier } from './domain/policy.js';
import { AnthropicProvider } from './providers/anthropic.js';
import { CheapHealth } from './providers/cheap-health.js';
import { OpenAICompatibleProvider } from './providers/openai/provider.js';
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
import { CircuitBreaker } from './providers/gemini/breaker.js';

// ponytail: fixed cap, one entry per live conversation. An evicted cheap
// conversation is treated as unknown and goes primary: savings lost, never
// correctness. Raise it only if `stats` shows that under heavy parallel use.
const SESSION_MAX_ENTRIES = 10_000;

/** Input tokens a cheap model can take: 90% of its window minus the reserved output. */
const inputBudget = (contextTokens: number, maxOutputTokens: number) => Math.floor(contextTokens * 0.9) - maxOutputTokens;

/**
 * Composition root: the only place that knows concrete implementations.
 * As a supervisor's worker it listens on an ephemeral loopback port and
 * reports it over IPC; otherwise it owns HOST:PORT itself.
 */
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

  // Errors surface through the server's logger, which exists only after buildServer.
  let logTelemetryError: (err: unknown) => void = () => {};
  let logCheapState: (state: string) => void = () => {};
  const db = config.telemetry.dbPath ? openTelemetryDb(config.telemetry.dbPath) : undefined;
  const telemetry: TelemetrySink = db ? new SqliteTelemetry(db, { onError: (err) => logTelemetryError(err) }) : noopTelemetry;

  // Persisted on change only: the status line, another process, reads the latest row.
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
        standardRoute: config.router.standardRoute,
        standardEnabled: config.cheapStandard !== undefined,
        minStandardProbability: config.router.minStandardProbability,
      },
      primaryClasses: config.router.primaryClasses,
      // Input budget: 90% of the window, minus the output tokens we reserve.
      cheapContextTokens: inputBudget(config.cheap.contextTokens, config.cheap.maxOutputTokens),
      ...(config.cheapStandard
        ? { standardContextTokens: inputBudget(config.cheapStandard.contextTokens, config.cheap.maxOutputTokens) }
        : {}),
      classifierTimeoutMs: config.classifier.timeoutMs,
      classifierMaxChars: config.classifier.maxChars,
      sessionTtlMs: config.router.sessionTtlMs,
      ...(quota
        ? {
            quota,
            pressurePolicy: {
              minCheapProbability: config.quota.minCheapProbability,
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
            },
            geminiFromPrimary: geminiConfig.fromPrimary,
          }
        : {}),
      readFirstStandard: config.cheapReadFirstHint && config.readFirstStandard && config.cheapStandard !== undefined,
    },
  );

  const geminiBreaker = geminiConfig
    ? new CircuitBreaker(geminiConfig.breakerFailures, geminiConfig.breakerCooldownMs, geminiConfig.maxConcurrency)
    : undefined;
  const cursorBreaker = config.cursor
    ? new CircuitBreaker(config.cursor.breakerFailures, config.cursor.breakerCooldownMs, config.cursor.maxConcurrency)
    : undefined;

  const providers = {
    primary: new AnthropicProvider(config.primary),
    cheap: new OpenAICompatibleProvider(config.cheap),
    // Same endpoint and key, bigger model: the health check covers both tiers.
    ...(config.cheapStandard
      ? { standard: new OpenAICompatibleProvider({ ...config.cheap, model: config.cheapStandard.model }) }
      : {}),
    ...(geminiConfig
      ? { gemini: new GeminiCliProvider(geminiConfig) }
      : {}),
    ...(config.cursor
      ? { cursor: new CursorCliProvider(config.cursor) }
      : {}),
  };

  const cheapHealth = config.cheapHealth.enabled
    ? new CheapHealth({
        baseUrl: config.cheap.baseUrl,
        apiKey: config.cheap.apiKey,
        intervalMs: config.cheapHealth.intervalMs,
        onChange: (state) => logCheapState(state),
      })
    : undefined;

  const build = runningBuild();
  const app = buildServer({
    config,
    router,
    providers,
    telemetry,
    ...(cheapHealth ? { cheapHealth } : {}),
    ...(build ? { build } : {}),
    ...(quota ? { quota } : {}),
    ...(geminiBreaker ? { geminiBreaker } : {}),
    ...(cursorBreaker ? { cursorBreaker } : {}),
  });
  logTelemetryError = (err) => app.log.error({ err }, 'telemetry write failed');
  logCheapState = (state) => app.log[state === 'down' ? 'warn' : 'info']({ cheap: state }, 'cheap provider health changed');
  app.addHook('onClose', () => telemetry.close());
  app.addHook('onClose', () => cheapHealth?.stop());

  await cheapHealth?.start();
  await app.listen(worker ? { host: '127.0.0.1', port: 0 } : { host: config.host, port: config.port });
  if (worker) {
    process.send?.({ type: 'listening', port: (app.server.address() as AddressInfo).port });
    process.on('message', (msg: { type?: string } | null) => {
      // Retired by a reload: stop taking connections; the supervisor cuts leftovers after RELOAD_DRAIN_MS.
      if (msg?.type === 'drain') void app.close().then(() => process.exit(0));
    });
    // Supervisor gone (crash, killed task): never linger as an orphan.
    process.once('disconnect', () => void app.close().then(() => process.exit(0)));
  }
  app.log.info(
    {
      classifier: classifier.name,
      build: build?.sha ?? 'src',
      ...(worker ? { worker: true } : {}),
      primary: config.primary.baseUrl,
      cheap: `${config.cheap.baseUrl} (${config.cheap.model})`,
      standard: config.cheapStandard?.model ?? 'off',
      gemini: geminiConfig ? geminiConfig.model : 'off',
      cursor: config.cursor ? config.cursor.model : 'off',
      telemetry: config.telemetry.dbPath ?? 'disabled',
    },
    'jev-router ready',
  );
  const stale = removedEnvSet();
  if (stale.length > 0) app.log.warn({ stale }, 'ignored: these env vars were removed, delete them from .env');

  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.once(signal, () => {
      void app.close().then(() => process.exit(0));
    });
  }
}

/** Supervisor mode: owns HOST:PORT and the control port, forwards to a forked worker. */
export async function supervise(config: Config, forkWorker: () => WorkerProcess): Promise<void> {
  const supervisor = await startSupervisor({
    host: config.host,
    port: config.port,
    controlPort: config.supervisor.controlPort,
    drainMs: config.supervisor.drainMs,
    forkWorker,
  });
  for (const signal of ['SIGINT', 'SIGTERM'] as const) {
    process.once(signal, () => {
      void supervisor.close().then(() => process.exit(0));
    });
  }
}
