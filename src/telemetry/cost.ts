import type { Pricing } from './pricing.js';
import type { FinalProvider, Outcome } from './schema.js';

/** The slice of a router_logs row the cost model needs. */
export interface CostRow {
  readonly id: number;
  readonly sessionId: string | null;
  readonly finalProvider: FinalProvider;
  readonly model: string | null;
  readonly requestedModel: string | null;
  readonly outcome: Outcome;
  readonly tokensIn: number | null;
  readonly tokensOut: number | null;
  readonly cacheReadTokens: number | null;
  readonly cacheWriteTokens: number | null;
  /** Tokens the JEV classification of this turn billed (null when JEV was not called). */
  readonly jevTokensIn: number | null;
  readonly jevTokensOut: number | null;
  readonly cacheWrite5mTokens?: number | null;
  readonly cacheWrite1hTokens?: number | null;
}

export interface NetCost {
  /** What Anthropic actually billed (or would bill, at list price) for primary rows. */
  readonly anthropicUsd: number;
  /** What the cheap provider billed, failed attempts included. */
  readonly cheapUsd: number;
  /** Counterfactual: the same traffic, all on Anthropic, with a warm prompt cache. */
  readonly baselineUsd: number;
  /** Baseline minus actual for the cheap requests that were served OK. */
  readonly grossSavingsUsd: number;
  /** Extra Anthropic cost of re-sending history uncached after leaving the cheap route. */
  readonly cachePenaltyUsd: number;
  /** Cheap requests that failed (stream error, client abort…): paid for, then redone on Anthropic. */
  readonly failedCheapUsd: number;
  /** What the JEV classifier billed: the router's own overhead. */
  readonly jevUsd: number;
  /** baseline − (anthropic + cheap + jev). Positive = profit, negative = loss. */
  readonly netUsd: number;
  /** cheap → Anthropic transitions that paid the cache-miss penalty. */
  readonly transitions: number;
  /** Served rows (outcome ok) priced at $0 because the response carried no usage; failures never have any. */
  readonly rowsWithoutUsage: number;
  /** Anthropic models missing from the price table (priced at $0, so the result is optimistic). */
  readonly unpricedModels: readonly string[];
}

const PER_MTOK = 1_000_000;

/**
 * Net cost of routing versus sending everything to Anthropic.
 *
 * The baseline assumes the conversation stayed on Anthropic with Claude Code's
 * prompt caching working: the part of the prompt already sent on the previous
 * turn is a cache read, the new part is a cache write. That is deliberately
 * harsh on the router:
 *
 *   - a cheap request "saves" only what that warm-cache Anthropic request would
 *     have cost, not the full uncached price;
 *   - the first Anthropic request after one or more cheap turns (escalation or
 *     fallback) re-sends the history the cache never saw; the difference to a
 *     warm-cache request is booked as the cache-miss penalty;
 *   - a cheap request that failed counts its full cost and saves nothing,
 *     because the retry on Anthropic is logged (and paid) as its own row.
 *
 * A failed cheap attempt is not a turn: its Anthropic retry is compared with
 * the turn before it. `rows` must be ordered by id. Rows without a session cannot be linked to a
 * previous turn and never pay the transition penalty.
 */
export function computeNetCost(rows: readonly CostRow[], pricing: Pricing): NetCost {
  const writeMult = pricing.cacheWriteMultiplier;
  const unpriced = new Set<string>();
  const last = new Map<string, CostRow>();
  let anthropicUsd = 0;
  let cheapUsd = 0;
  let baselineUsd = 0;
  let grossSavingsUsd = 0;
  let cachePenaltyUsd = 0;
  let failedCheapUsd = 0;
  let jevUsd = 0;
  let transitions = 0;
  let rowsWithoutUsage = 0;

  const priceOf = (row: CostRow) => {
    const model = row.requestedModel ?? row.model;
    const price = pricing.primary(model);
    if (!price && model) unpriced.add(model);
    return price ?? { input: 0, output: 0, cacheRead: 0 };
  };

  /** Anthropic cost of `row` had the conversation's cache been warm up to the previous turn. */
  const warmAnthropicUsd = (row: CostRow, prev: CostRow | undefined) => {
    const p = priceOf(row);
    const tin = row.tokensIn ?? 0;
    const history = Math.min(prev?.tokensIn ?? 0, tin);
    return (history * p.cacheRead + (tin - history) * p.input * writeMult + (row.tokensOut ?? 0) * p.output) / PER_MTOK;
  };

  for (const row of rows) {
    const prev = row.sessionId ? last.get(row.sessionId) : undefined;
    if (row.tokensIn === null && row.outcome === 'ok') rowsWithoutUsage++;
    jevUsd += ((row.jevTokensIn ?? 0) * pricing.jev.input + (row.jevTokensOut ?? 0) * pricing.jev.output) / PER_MTOK;

    if (row.finalProvider === 'openai' || row.finalProvider === 'gemini' || row.finalProvider === 'cursor') {
      const actual = row.finalProvider === 'openai' ? ((row.tokensIn ?? 0) * pricing.cheap.input + (row.tokensOut ?? 0) * pricing.cheap.output) / PER_MTOK : 0;
      cheapUsd += actual;
      if (row.outcome === 'ok') {
        const warm = warmAnthropicUsd(row, prev);
        baselineUsd += warm;
        grossSavingsUsd += warm - actual;
      } else {
        failedCheapUsd += actual;
      }
    } else {
      const p = priceOf(row);
      const tin = row.tokensIn ?? 0;
      const read = row.cacheReadTokens ?? 0;
      const write = row.cacheWriteTokens ?? 0;
      let writeCost = write * p.input * writeMult;
      if (row.cacheWrite5mTokens != null || row.cacheWrite1hTokens != null) {
        writeCost = ((row.cacheWrite5mTokens ?? 0) * 1.25 + (row.cacheWrite1hTokens ?? 0) * 2) * p.input;
      }
      const actual =
        (Math.max(0, tin - read - write) * p.input + writeCost + read * p.cacheRead +
          (row.tokensOut ?? 0) * p.output) /
        PER_MTOK;
      anthropicUsd += actual;
      if ((prev?.finalProvider === 'openai' || prev?.finalProvider === 'gemini' || prev?.finalProvider === 'cursor') && row.tokensIn !== null) {
        const counterfactual = Math.min(actual, warmAnthropicUsd(row, prev));
        baselineUsd += counterfactual;
        cachePenaltyUsd += actual - counterfactual;
        transitions++;
      } else {
        baselineUsd += actual;
      }
    }

    // A failed cheap attempt is retried with the same prompt, so it is not the
    // turn the next request's history builds on.
    const servedTurn = row.tokensIn !== null && (row.finalProvider === 'anthropic' || row.outcome === 'ok');
    if (row.sessionId && servedTurn) last.set(row.sessionId, row);
  }

  return {
    anthropicUsd,
    cheapUsd,
    baselineUsd,
    grossSavingsUsd,
    cachePenaltyUsd,
    failedCheapUsd,
    jevUsd,
    netUsd: baselineUsd - anthropicUsd - cheapUsd - jevUsd,
    transitions,
    rowsWithoutUsage,
    unpricedModels: [...unpriced].sort(),
  };
}
