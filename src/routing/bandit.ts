/**
 * Contextual bandit for the cheap route, replay only (nothing here is wired into routing).
 *
 * One arm is learned: "serve this turn on the cheap tier", with y = 1 when it went wrong
 * (inspection miss, failure, tool error or interruption on the next turn). The other arm,
 * Claude, is the safe baseline and needs no model. LinUCB keeps a ridge regression of y on the
 * JEV features plus an uncertainty width; a turn is vetoed (sent to Claude) when the pessimistic
 * estimate `mean + width` of its failure rate exceeds `maxBad`. Unknown regions of the feature
 * space therefore start vetoed, and the width shrinks as evidence accumulates.
 *
 * ponytail: d is ~8, so A⁻¹ is kept directly and updated by Sherman–Morrison, O(d²) per row.
 * Wire it into decideTier only after `replay` shows it beats the fixed 0.9 threshold.
 */

export interface JevFeatures {
  readonly simple: number;
  readonly standard: number;
  readonly structural: number;
  readonly risk?: number | null | undefined;
  readonly textOnly?: number | null | undefined;
}

// toolsOffered is deliberately not a feature: inspection_miss (the dominant label) only exists when
// tools were offered, so the model would learn the label's own definition.
export function featuresOf(jev: JevFeatures, tokensIn: number | null | undefined): number[] {
  return [
    1,
    jev.simple,
    jev.standard,
    jev.structural,
    jev.risk ?? 0,
    jev.textOnly ?? 0,
    // scaled so a 1M-token prompt is ~1: keeps every feature in [0, 1] for a well-conditioned ridge
    Math.log10(1 + Math.max(0, tokensIn ?? 0)) / 6,
  ];
}

export interface LinUcbOptions {
  /** Width multiplier: higher vetoes unfamiliar turns for longer. */
  readonly alpha?: number;
  /** Ridge prior: higher trusts the zero prior (and so the width) for longer. */
  readonly lambda?: number;
}

export class LinUcb {
  private readonly ainv: number[][];
  private readonly b: number[];
  private readonly alpha: number;

  constructor(
    private readonly dim: number,
    { alpha = 0.5, lambda = 1 }: LinUcbOptions = {},
  ) {
    this.alpha = alpha;
    this.ainv = Array.from({ length: dim }, (_, i) => Array.from({ length: dim }, (_, j) => (i === j ? 1 / lambda : 0)));
    this.b = new Array<number>(dim).fill(0);
  }

  private mul(x: readonly number[]): number[] {
    return this.ainv.map((row) => row.reduce((s, v, j) => s + v * x[j]!, 0));
  }

  /** Predicted failure rate and the uncertainty around it. */
  score(x: readonly number[]): { mean: number; bound: number } {
    const ax = this.mul(x);
    const mean = this.mul(this.b).reduce((s, v, j) => s + v * x[j]!, 0);
    const variance = x.reduce((s, v, j) => s + v * ax[j]!, 0);
    return { mean, bound: this.alpha * Math.sqrt(Math.max(0, variance)) };
  }

  update(x: readonly number[], y: number): void {
    const ax = this.mul(x);
    const denom = 1 + x.reduce((s, v, j) => s + v * ax[j]!, 0);
    for (let i = 0; i < this.dim; i++) {
      for (let j = 0; j < this.dim; j++) this.ainv[i]![j]! -= (ax[i]! * ax[j]!) / denom;
      this.b[i]! += y * x[i]!;
    }
  }
}

export interface ReplayRow {
  readonly x: readonly number[];
  readonly bad: boolean;
}

export interface ReplayOptions {
  /** Rows to observe before any veto: the width is meaningless on an empty model. */
  readonly minRows: number;
  /** Failure rate above which the pessimistic estimate vetoes the cheap route. */
  readonly maxBad: number;
  readonly alpha?: number;
}

export interface ReplayResult {
  readonly n: number;
  /** Rows after the warm-up: the only ones that could be vetoed, so the honest base for any rate. */
  readonly eligible: number;
  readonly vetoed: number;
  readonly badAmongVetoed: number;
  readonly badAmongKept: number;
  /** Bad turns among the eligible rows. */
  readonly badEligible: number;
  /** Failure rate of the whole stream: what serving everything cheap would have cost. */
  readonly baseRate: number;
}

/**
 * Online simulation over turns the cheap route actually served, in order: predict from what was
 * known before the row, then learn its outcome. A vetoed row still teaches the model (its outcome
 * is known in the log); live, only served rows would, so this is optimistic about learning speed.
 */
export function replay(rows: readonly ReplayRow[], { minRows, maxBad, alpha }: ReplayOptions): ReplayResult {
  const dim = rows[0]?.x.length ?? 1;
  const model = new LinUcb(dim, alpha === undefined ? {} : { alpha });
  let vetoed = 0;
  let badAmongVetoed = 0;
  let badAmongKept = 0;
  let badEligible = 0;
  rows.forEach((row, i) => {
    const { mean, bound } = model.score(row.x);
    const veto = i >= minRows && mean + bound > maxBad;
    if (i >= minRows && row.bad) badEligible++;
    if (veto) {
      vetoed++;
      if (row.bad) badAmongVetoed++;
    } else if (row.bad) badAmongKept++;
    model.update(row.x, row.bad ? 1 : 0);
  });
  const bad = badAmongVetoed + badAmongKept;
  return { n: rows.length, eligible: Math.max(0, rows.length - minRows), badEligible, vetoed, badAmongVetoed, badAmongKept, baseRate: rows.length ? bad / rows.length : 0 };
}
