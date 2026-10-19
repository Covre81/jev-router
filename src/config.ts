import { z } from 'zod';
import { defaultTelemetryDbPath, jevHome } from './paths.js';
import { join } from 'node:path';
import { homedir } from 'node:os';

const bool = z.enum(['true', 'false']).transform((v) => v === 'true');
const csv = z.string().transform((v) => new Set(v.split(',').map((s) => s.trim()).filter(Boolean)));

const LOOPBACK = new Set(['127.0.0.1', '::1', 'localhost']);

export function expandEnv(str: string, env: NodeJS.ProcessEnv): string {
  return str.replace(/%([^%]+)%/g, (_, name) => env[name] ?? '');
}

const Env = z
  .object({
    HOST: z.string().default('127.0.0.1'),
    PORT: z.coerce.number().int().min(1).max(65_535).default(8787),
    LOG_LEVEL: z.enum(['fatal', 'error', 'warn', 'info', 'debug', 'trace']).default('info'),

    SUPERVISOR_ENABLED: bool.default(true),
    // 0 = PORT + 1. Loopback-only: it accepts `jev-router reload`.
    CONTROL_PORT: z.coerce.number().int().min(0).max(65_535).default(0),
    RELOAD_DRAIN_MS: z.coerce.number().int().min(0).default(120_000),

    ANTHROPIC_UPSTREAM_URL: z.url().default('https://api.anthropic.com'),
    UPSTREAM_AUTH_MODE: z.enum(['passthrough', 'inject']).default('passthrough'),
    ANTHROPIC_API_KEY: z.string().min(1).optional(),
    PROXY_AUTH_TOKEN: z.string().min(16).optional(),
    UPSTREAM_TIMEOUT_MS: z.coerce.number().int().positive().default(600_000),
    BODY_LIMIT_BYTES: z.coerce.number().int().positive().default(32 * 1024 * 1024),

    CLASSIFIER: z.enum(['jev', 'heuristic']).default('jev'),
    TYPESAFE_API_KEY: z.string().min(1).optional(),
    JEV_API_URL: z.url().default('https://api.typesafe.ai/v1/systemone'),
    JEV_MODEL: z.string().default('jev-latest'),
    JEV_TIMEOUT_MS: z.coerce.number().int().positive().default(1_500),
    CLASSIFIER_MAX_CHARS: z.coerce.number().int().positive().default(4_000),

    CHEAP_BASE_URL: z.url().default('http://127.0.0.1:11434/v1'),
    CHEAP_API_KEY: z.string().min(1),
    CHEAP_MODEL: z.string().min(1).default('gpt-oss:20b-cloud'),
    CHEAP_MAX_OUTPUT_TOKENS: z.coerce.number().int().positive().default(8_192),
    CHEAP_CONTEXT_TOKENS: z.coerce.number().int().positive().default(131_072),
    CHEAP_TIMEOUT_MS: z.coerce.number().int().positive().default(60_000),
    // Standard tier: level-2 work on a bigger cheap model, same CHEAP_BASE_URL. `off` disables it.
    CHEAP_MODEL_STANDARD: z.string().min(1).default('gemma4:31b-cloud'),
    CHEAP_STANDARD_CONTEXT_TOKENS: z.coerce.number().int().positive().optional(),
    CHEAP_HEALTH_ENABLED: bool.default(true),
    CHEAP_HEALTH_INTERVAL_MS: z.coerce.number().int().positive().default(30_000),

    ROUTER_MIN_CHEAP_PROBABILITY: z.coerce.number().min(0).max(1).default(0.9),
    ROUTER_MIN_STANDARD_PROBABILITY: z.coerce.number().min(0).max(1).default(0.75),
    ROUTER_STANDARD_ROUTE: z.enum(['primary', 'cheap']).default('primary'),
    ROUTER_PRIMARY_CLASSES: csv.default(new Set(['auxiliary', 'compaction'])),
    FAILOVER_ON_PRIMARY_RATE_LIMIT: bool.default(true),

    // Quota routing: from QUOTA_PRESSURE of the binding Claude window the cheap bars drop;
    // from QUOTA_CRITICAL a new human turn may leave a Claude-pinned session.
    QUOTA_ROUTING: bool.default(true),
    QUOTA_PRESSURE: z.coerce.number().min(0).max(1).default(0.8),
    QUOTA_CRITICAL: z.coerce.number().min(0).max(1).default(0.95),
    QUOTA_PRESSURE_MIN_CHEAP: z.coerce.number().min(0).max(1).default(0.7),
    QUOTA_PRESSURE_MIN_STANDARD: z.coerce.number().min(0).max(1).default(0.6),
    SESSION_TTL_MS: z.coerce.number().int().positive().default(6 * 60 * 60 * 1000),

    CHEAP_RETRY: bool.default(true),
    CHEAP_RETRY_DELAY_MS: z.coerce.number().int().min(0).default(250),
    CHEAP_RETRY_HEADERS_TIMEOUT_MS: z.coerce.number().int().positive().default(15000),
    CHEAP_READ_FIRST_HINT: bool.default(true),
    CHEAP_READ_FIRST_STANDARD: bool.default(true),

    TELEMETRY_ENABLED: bool.default(true),
    TELEMETRY_DIAGNOSTICS: bool.default(false),
    TELEMETRY_DB_PATH: z.string().min(1).optional(),
    GEMINI_TIER: z.enum(['off', 'on']).default('off'),
    GEMINI_TIER_BIN: z.string().optional(),
    GEMINI_TIER_MODEL: z.string().default('gemini-3.8-flash-high'),
    GEMINI_TIER_TIMEOUT_MS: z.coerce.number().int().positive().default(60_000),
    GEMINI_TIER_MIN_TEXT_ONLY: z.coerce.number().min(0).max(1).default(0.8),
    GEMINI_TIER_PRESSURE_MIN_TEXT_ONLY: z.coerce.number().min(0).max(1).default(0.6),
    GEMINI_TIER_FROM_PRIMARY: bool.default(true),
    GEMINI_TIER_MAX_CONCURRENCY: z.coerce.number().int().min(1).max(4).default(1),
    GEMINI_TIER_MAX_PROMPT_CHARS: z.coerce.number().int().positive().default(200_000),
    GEMINI_TIER_BREAKER_FAILURES: z.coerce.number().int().min(1).default(3),
    GEMINI_TIER_BREAKER_COOLDOWN_MS: z.coerce.number().int().positive().default(300_000),
    GEMINI_TIER_HOME: z.string().optional(), // defaults to <jevHome()>/agy-home in loadConfig
  })
  .superRefine((env, ctx) => {
    if (env.QUOTA_CRITICAL < env.QUOTA_PRESSURE) {
      ctx.addIssue({ code: 'custom', path: ['QUOTA_CRITICAL'], message: 'must be >= QUOTA_PRESSURE' });
    }
    if (env.CONTROL_PORT === 0 && env.PORT === 65_535) {
      ctx.addIssue({ code: 'custom', path: ['CONTROL_PORT'], message: 'PORT is 65535: set CONTROL_PORT explicitly' });
    }
    if (env.CLASSIFIER === 'jev' && !env.TYPESAFE_API_KEY) {
      ctx.addIssue({ code: 'custom', path: ['TYPESAFE_API_KEY'], message: 'required when CLASSIFIER=jev' });
    }
    if (env.UPSTREAM_AUTH_MODE === 'inject' && !env.ANTHROPIC_API_KEY) {
      ctx.addIssue({ code: 'custom', path: ['ANTHROPIC_API_KEY'], message: 'required when UPSTREAM_AUTH_MODE=inject' });
    }
    // A key-injecting proxy reachable from the network is an open relay to your bill.
    if (env.UPSTREAM_AUTH_MODE === 'inject' && !LOOPBACK.has(env.HOST) && !env.PROXY_AUTH_TOKEN) {
      ctx.addIssue({
        code: 'custom',
        path: ['PROXY_AUTH_TOKEN'],
        message: 'required when UPSTREAM_AUTH_MODE=inject and HOST is not loopback',
      });
    }
  });

/** Env vars that were removed: an old .env that still sets them is otherwise ignored without a word. */
const REMOVED_ENV = ['ROUTER_ALLOW_ESCALATION', 'SESSION_MAX_ENTRIES'] as const;

export function removedEnvSet(env: NodeJS.ProcessEnv = process.env): string[] {
  return REMOVED_ENV.filter((name) => env[name] !== undefined);
}

/**
 * Control port without the rest of the config: `jev-router reload` must work
 * without provider keys. Same rule as loadConfig (0 = PORT + 1).
 */
export function controlPort(env: NodeJS.ProcessEnv = process.env): number {
  return Number(env['CONTROL_PORT'] || 0) || Number(env['PORT'] || 8787) + 1;
}

export type Config =Readonly<ReturnType<typeof loadConfig>>;

/** Parse once at boot; any misconfiguration kills the process before it listens. */
export function loadConfig(env: NodeJS.ProcessEnv = process.env) {
  const parsed = Env.safeParse(env);
  if (!parsed.success) {
    throw new Error(`Invalid configuration:\n${z.prettifyError(parsed.error)}`);
  }
  const e = parsed.data;
  const trimSlash = (url: string) => url.replace(/\/+$/, '');

  let geminiBin = e.GEMINI_TIER_BIN;
  if (!geminiBin) {
    geminiBin = process.platform === 'win32'
      ? join(env.LOCALAPPDATA ?? join(homedir(), 'AppData', 'Local'), 'agy', 'bin', 'agy.exe')
      : 'agy';
  } else {
    geminiBin = expandEnv(geminiBin, env);
  }

  return {
    host: e.HOST,
    port: e.PORT,
    logLevel: e.LOG_LEVEL,
    supervisor: { enabled: e.SUPERVISOR_ENABLED, controlPort: e.CONTROL_PORT || e.PORT + 1, drainMs: e.RELOAD_DRAIN_MS },
    cheapHealth: { enabled: e.CHEAP_HEALTH_ENABLED, intervalMs: e.CHEAP_HEALTH_INTERVAL_MS },
    quota: {
      enabled: e.QUOTA_ROUTING,
      pressure: e.QUOTA_PRESSURE,
      critical: e.QUOTA_CRITICAL,
      minCheapProbability: e.QUOTA_PRESSURE_MIN_CHEAP,
      minStandardProbability: e.QUOTA_PRESSURE_MIN_STANDARD,
    },
    primary: {
      baseUrl: trimSlash(e.ANTHROPIC_UPSTREAM_URL),
      authMode: e.UPSTREAM_AUTH_MODE,
      apiKey: e.ANTHROPIC_API_KEY,
      timeoutMs: e.UPSTREAM_TIMEOUT_MS,
    },
    cheap: {
      baseUrl: trimSlash(e.CHEAP_BASE_URL),
      apiKey: e.CHEAP_API_KEY,
      model: e.CHEAP_MODEL,
      maxOutputTokens: e.CHEAP_MAX_OUTPUT_TOKENS,
      contextTokens: e.CHEAP_CONTEXT_TOKENS,
      timeoutMs: e.CHEAP_TIMEOUT_MS,
    },
    cheapRetry: {
      enabled: e.CHEAP_RETRY,
      delayMs: e.CHEAP_RETRY_DELAY_MS,
      headersTimeoutMs: e.CHEAP_RETRY_HEADERS_TIMEOUT_MS,
    },
    cheapReadFirstHint: e.CHEAP_READ_FIRST_HINT,
    readFirstStandard: e.CHEAP_READ_FIRST_STANDARD,
    // undefined = standard tier off: level-2 work stays on the primary (the old behavior).
    cheapStandard:
      e.CHEAP_MODEL_STANDARD === 'off'
        ? undefined
        : { model: e.CHEAP_MODEL_STANDARD, contextTokens: e.CHEAP_STANDARD_CONTEXT_TOKENS ?? e.CHEAP_CONTEXT_TOKENS },
    proxyAuthToken: e.PROXY_AUTH_TOKEN,
    bodyLimitBytes: e.BODY_LIMIT_BYTES,
    classifier:
      e.CLASSIFIER === 'jev'
        ? {
            kind: 'jev' as const,
            apiUrl: e.JEV_API_URL,
            apiKey: e.TYPESAFE_API_KEY!,
            model: e.JEV_MODEL,
            timeoutMs: e.JEV_TIMEOUT_MS,
            maxChars: e.CLASSIFIER_MAX_CHARS,
          }
        : { kind: 'heuristic' as const, timeoutMs: e.JEV_TIMEOUT_MS, maxChars: e.CLASSIFIER_MAX_CHARS },
    router: {
      minCheapProbability: e.ROUTER_MIN_CHEAP_PROBABILITY,
      minStandardProbability: e.ROUTER_MIN_STANDARD_PROBABILITY,
      standardRoute: e.ROUTER_STANDARD_ROUTE,
      primaryClasses: e.ROUTER_PRIMARY_CLASSES,
      failoverOnPrimaryRateLimit: e.FAILOVER_ON_PRIMARY_RATE_LIMIT,
      sessionTtlMs: e.SESSION_TTL_MS,
    },
    gemini:
      e.GEMINI_TIER === 'off'
        ? undefined
        : {
            bin: geminiBin,
            model: e.GEMINI_TIER_MODEL,
            timeoutMs: e.GEMINI_TIER_TIMEOUT_MS,
            minTextOnly: e.GEMINI_TIER_MIN_TEXT_ONLY,
            pressureMinTextOnly: e.GEMINI_TIER_PRESSURE_MIN_TEXT_ONLY,
            fromPrimary: e.GEMINI_TIER_FROM_PRIMARY,
            maxConcurrency: e.GEMINI_TIER_MAX_CONCURRENCY,
            maxPromptChars: e.GEMINI_TIER_MAX_PROMPT_CHARS,
            breakerFailures: e.GEMINI_TIER_BREAKER_FAILURES,
            breakerCooldownMs: e.GEMINI_TIER_BREAKER_COOLDOWN_MS,
            home: e.GEMINI_TIER_HOME ?? join(jevHome(env), 'agy-home'),
          },
    // dbPath is undefined when telemetry is disabled.
    telemetry: { dbPath: e.TELEMETRY_ENABLED ? (e.TELEMETRY_DB_PATH ?? defaultTelemetryDbPath(env)) : undefined, diagnostics: e.TELEMETRY_DIAGNOSTICS },
  };
}
