export const QUOTA_DEFAULT_COOLDOWN_MS = 6 * 60 * 60 * 1000; // 6 hours

const QUOTA_ERROR_PATTERNS: readonly (string | RegExp)[] = [
  /individual quota reached/i,
  /\bquota (?:exceeded|reached)\b/i,
  /exceeded your current quota/i,
  /\bresource[_ ]exhausted\b/i,
  /\brate limit (?:exceeded|reached)\b/i,
  /\b(?:http|status|error)\s*:?\s*429\b/i,
  /\b429 too many requests\b/i,
  /\busage limit\b/i,
  /you(?:'ve| have) hit your (?:\w+ )?limit/i,
  /out of credits/i,
];

/**
 * Parses duration strings like "136h47m52s", "2h", "45m", "1d 2h", "30s" into milliseconds.
 */
export function parseDurationMs(durationStr: string): number | null {
  const matches = [...durationStr.matchAll(/(\d+)\s*([dhms])/gi)];
  if (matches.length === 0) return null;
  let totalMs = 0;
  for (const match of matches) {
    const val = parseInt(match[1]!, 10);
    const unit = match[2]!.toLowerCase();
    if (unit === 'd') totalMs += val * 24 * 60 * 60 * 1000;
    else if (unit === 'h') totalMs += val * 60 * 60 * 1000;
    else if (unit === 'm') totalMs += val * 60 * 1000;
    else if (unit === 's') totalMs += val * 1000;
  }
  return totalMs > 0 ? totalMs : null;
}

/**
 * Parses a quota error string to determine whether it is a quota/rate-limit error,
 * and if so, extracts the reset timestamp (in epoch milliseconds).
 * Falls back to now + QUOTA_DEFAULT_COOLDOWN_MS if no reset time is mentioned.
 */
export function parseQuotaError(text: string, now = Date.now()): { resetAt: number } | null {
  if (!text || typeof text !== 'string') return null;

  const isQuota = QUOTA_ERROR_PATTERNS.some((pat) =>
    typeof pat === 'string' ? text.toLowerCase().includes(pat) : pat.test(text),
  );
  if (!isQuota) return null;

  // Check for "Resets in <duration>"
  const resetInMatch = text.match(/resets?\s+in\s+([0-9dhms\s]+)/i);
  if (resetInMatch && resetInMatch[1]) {
    const durationMs = parseDurationMs(resetInMatch[1]);
    if (durationMs !== null) {
      return { resetAt: now + durationMs };
    }
  }

  // Check for ISO timestamp
  const isoMatch = text.match(/\b(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:?\d{2})?)\b/);
  if (isoMatch && isoMatch[1]) {
    const dateMs = Date.parse(isoMatch[1]);
    if (!isNaN(dateMs)) {
      return { resetAt: dateMs };
    }
  }

  return { resetAt: now + QUOTA_DEFAULT_COOLDOWN_MS };
}

export interface QuotaEntry {
  readonly resetAt: number;
  readonly reason: string;
}

export interface ModelQuotaSnapshot {
  readonly state: 'ok' | 'exhausted';
  readonly until?: string;
  readonly reason?: string;
}

export interface QuotaLogger {
  info?(obj: unknown, msg?: string): void;
  warn?(obj: unknown, msg?: string): void;
}

export class ModelQuotaState {
  private readonly models = new Map<string, QuotaEntry>();
  private readonly loggedExhausted = new Set<string>();

  constructor(
    private readonly clock: () => number = () => Date.now(),
    private readonly logger?: QuotaLogger,
  ) {}

  markExhausted(model: string, resetAt: number, reason: string): void {
    const now = this.clock();
    if (resetAt <= now) return;

    this.models.set(model, { resetAt, reason });

    if (!this.loggedExhausted.has(model)) {
      this.loggedExhausted.add(model);
      const until = new Date(resetAt).toISOString();
      this.logger?.warn?.({ model, until, reason }, 'model quota exhausted');
    }
  }

  isAvailable(model: string): boolean {
    const entry = this.models.get(model);
    if (!entry) return true;

    const now = this.clock();
    if (now >= entry.resetAt) {
      this.models.delete(model);
      this.loggedExhausted.delete(model);
      this.logger?.info?.({ model }, 'model quota restored');
      return true;
    }
    return false;
  }

  snapshot(configuredModels: readonly string[] = []): Record<string, ModelQuotaSnapshot> {
    const result: Record<string, ModelQuotaSnapshot> = {};
    const now = this.clock();

    const allModels = new Set<string>(configuredModels);
    for (const m of this.models.keys()) {
      allModels.add(m);
    }

    for (const m of allModels) {
      const entry = this.models.get(m);
      if (entry && now < entry.resetAt) {
        result[m] = {
          state: 'exhausted',
          until: new Date(entry.resetAt).toISOString(),
          reason: entry.reason,
        };
      } else {
        if (entry && now >= entry.resetAt) {
          this.models.delete(m);
          this.loggedExhausted.delete(m);
          this.logger?.info?.({ model: m }, 'model quota restored');
        }
        result[m] = { state: 'ok' };
      }
    }

    return result;
  }
}
