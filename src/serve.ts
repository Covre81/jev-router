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
import { CircuitBreaker } from './providers/breaker.js';

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
            firstByteTimeoutMs: config.timeouts.firstByteMs,
            idleTimeoutMs: config.timeouts.idleMs,
          }),
        }
      : {}),
  };

  const build = runningBuild();
  const app = buildServer({
    config,
    router,
    providers,
    telemetry,
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
