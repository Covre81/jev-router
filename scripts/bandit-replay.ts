/**
 * Replays the cheap-route history through the LinUCB bandit (src/routing/bandit.ts) and prints
 * what it would have vetoed. Read-only: it never touches routing or the database.
 *
 *   npx tsx scripts/bandit-replay.ts [--db <file>] [--min-rows 50]
 *
 * Rows: turns the cheap tier (Gemini/Ollama) served that carry a fresh JEV score. "Sticky" turns
 * carry none and are skipped. A turn is bad when it was an inspection miss, did not finish ok, or
 * the user's very next message in the session hit a tool error or interrupted (needs the
 * prev_* columns; databases from before that migration are scored without them).
 */
import { homedir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parseArgs } from 'node:util';
import { featuresOf, replay, type ReplayRow } from '../src/routing/bandit.js';
import { openTelemetryDb, type TelemetryDb } from '../src/telemetry/db.js';

interface Raw {
  session_id: string | null;
  final_provider: string;
  outcome: string;
  inspection_miss: number | null;
  tokens_in: number | null;
  tools_offered: number | null;
  jev_decision: string | null;
  prev_tool_errors: number | null;
  prev_interrupted: number | null;
}

export function loadReplayRows(db: TelemetryDb): ReplayRow[] {
  const cols = new Set((db.prepare('PRAGMA table_info(router_logs)').all() as { name: string }[]).map((c) => c.name));
  const react = cols.has('prev_tool_errors')
    ? 'prev_tool_errors, prev_interrupted'
    : 'NULL AS prev_tool_errors, NULL AS prev_interrupted';
  const raws = db
    .prepare(
      `SELECT session_id, final_provider, outcome, inspection_miss, tokens_in, tools_offered, jev_decision, ${react}
       FROM router_logs ORDER BY id`,
    )
    .all() as unknown as Raw[];

  const rows: { x: number[]; bad: boolean }[] = [];
  const last = new Map<string, { bad: boolean } | undefined>(); // session -> its previous row, if it is a replay row
  for (const r of raws) {
    const key = r.session_id ?? '';
    const prev = last.get(key);
    if (prev && ((r.prev_tool_errors ?? 0) > 0 || r.prev_interrupted === 1)) prev.bad = true;

    let mine: { x: number[]; bad: boolean } | undefined;
    if ((r.final_provider === 'gemini' || r.final_provider === 'openai') && r.jev_decision) {
      const jev = JSON.parse(r.jev_decision) as Parameters<typeof featuresOf>[0];
      mine = { x: featuresOf(jev, r.tokens_in), bad: r.inspection_miss === 1 || r.outcome !== 'ok' };
      rows.push(mine);
    }
    last.set(key, mine);
  }
  return rows;
}

const pct = (n: number, d: number) => (d === 0 ? '—' : `${((100 * n) / d).toFixed(1)}%`);

export function report(rows: readonly ReplayRow[], minRows: number): string {
  const lines = [`cheap turns with a JEV score: ${rows.length} (first ${minRows} only warm the model up)`];
  if (rows.length < minRows * 2) return [...lines, `too few rows to replay (need at least ${minRows * 2}).`].join('\n');
  lines.push('maxBad  vetoed        bad caught   lift vs random  bad rate kept (base)');
  for (const maxBad of [0.05, 0.1, 0.15, 0.25]) {
    const r = replay(rows, { minRows, maxBad });
    const keptRows = r.eligible - r.vetoed;
    // 1.0x = vetoing the same share of the eligible turns at random would catch as many bad ones
    const lift = r.vetoed === 0 || r.badEligible === 0 ? '—' : `${(r.badAmongVetoed / r.badEligible / (r.vetoed / r.eligible)).toFixed(2)}x`;
    lines.push(
      `${maxBad.toFixed(2)}    ${pct(r.vetoed, r.eligible).padEnd(12)}  ${pct(r.badAmongVetoed, r.badEligible).padEnd(11)}  ${lift.padEnd(14)}  ${pct(r.badEligible - r.badAmongVetoed, keptRows)} (${pct(r.badEligible, r.eligible)})`,
    );
  }
  lines.push(
    '',
    'EXPLORATORY: one dataset, no confidence interval, the best of several limits is picked on the',
    'same rows, vetoed rows still teach the model, and only turns the fixed 0.9 rule already sent',
    'cheap are seen (no counterfactual). It cannot show the bandit beats the fixed threshold.',
  );
  return lines.join('\n');
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const { values } = parseArgs({ options: { db: { type: 'string' }, 'min-rows': { type: 'string' } } });
  const db = openTelemetryDb(values.db ?? join(homedir(), '.jev-router', 'telemetry.db'), { readonly: true });
  console.log(report(loadReplayRows(db), Number(values['min-rows'] ?? 50)));
}
