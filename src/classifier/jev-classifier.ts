import { z } from 'zod';
import { toDistribution, type ComplexityDistribution } from '../domain/complexity.js';
import type { Classification, ClassificationInput, ComplexityClassifier, RiskQuestion, RiskScores } from './classifier.js';
import { JevClient } from '../context/jev-client.js';

/**
 * Adapter for TypeSafe JEV (`POST /v1/systemone`).
 *
 * JEV does not generate text: it answers typed questions about a `state` with
 * calibrated probabilities. We ask ONE ordinal `score` question with three
 * levels and keep the full distribution — the policy, not the classifier,
 * decides what to do with uncertainty.
 */
export const QUESTION_ID = 'task_complexity';

const COMPLEXITY_CRITERIA = [
  'Simple and self-contained: answering a question, explaining code, a one-file edit, a rename, formatting, a docstring or a single small unit test, running a known command.',
  'Standard feature work: a bugfix or feature touching a few files, a React Native screen or hook, a Python module with its tests, debugging with a clear reproduction.',
  'Structural or complex: Clean Architecture/SOLID refactors across layers, test-suite design with heavy fixtures or mocking, concurrency, security or performance root causes, migrations, ambiguous requirements.',
] as const;

/**
 * Speculative Nouls asked in the same call (evaluated in parallel, no extra
 * latency): work JEV may score as simple that must still stay on the primary.
 */
export const RISK_QUESTIONS = {
  security_sensitive: {
    type: 'noul',
    instructions:
      'The request changes authentication, authorization, password hashing, token validation, cryptography, or how secrets and API keys are stored.',
  },
  destructive_or_production: {
    type: 'noul',
    instructions: 'The request deletes data or files, runs a database migration, or acts on a production system.',
  },
  // The cheap model answered "is everything ok here?" without checking anything and
  // invented a diff summary: questions about the repository's state need its tools.
  requires_inspection: {
    type: 'noul',
    instructions:
      "The request asks for a technical audit, verdict, or report about the codebase or repository's current state (what changed in commits, what is broken in the repo, or a codebase health check) that can only be answered after running tools to inspect multiple files, git diffs, or logs not provided in the prompt (e.g. 'summarize what changed in the statusline', 'is everything ok here?'). Casual greetings, small talk, social check-ins, or open conversational questions to the assistant (e.g. 'bom dia', 'tudo bem?', 'tudo indo bem?', 'o que tens hoje?', 'how are you?', 'what do you have today?'), as well as requests to make a specific edit, run a named command, or explain code/text included in the prompt, are NOT this.",
  },
} as const satisfies Record<RiskQuestion, unknown>;

/** Risk answers are a safety gate: required, so a missing one fails the call toward primary. */
const RiskNoul = z.looseObject({ noul: z.number().min(0).max(1) });
// Spelled out (not derived from RISK_QUESTIONS) so the parsed answers stay typed; `satisfies` keeps the two in sync.
const JevRisk = z.looseObject({
  answers: z.looseObject({
    security_sensitive: RiskNoul,
    destructive_or_production: RiskNoul,
    requires_inspection: RiskNoul,
  } satisfies Record<RiskQuestion, typeof RiskNoul>),
});

/** The part of the JEV response the adapter depends on. Unknown fields are tolerated. */
export const JevResponse = z.looseObject({
  answers: z.looseObject({
    [QUESTION_ID]: z.looseObject({
      probabilities: z.record(z.string(), z.number().min(0).max(1)),
    }),
  }),
});

/** Billed tokens, read leniently: a missing or malformed usage block only loses the cost line. */
const JevUsage = z.looseObject({
  usage: z.looseObject({ input_tokens: z.number().int().min(0), output_tokens: z.number().int().min(0) }),
});

/** Versioned model that answered, read leniently like the usage. */
const JevModel = z.looseObject({ model: z.string().min(1) });

export interface JevClassifierOptions {
  readonly apiUrl: string;
  readonly apiKey: string;
  readonly model: string;
  readonly geminiEnabled?: boolean;
}

export class JevClassifier implements ComplexityClassifier {
  readonly name = 'jev';

  constructor(private readonly options: JevClassifierOptions) {}

  async classify(input: ClassificationInput, signal: AbortSignal): Promise<Classification> {
    const client = new JevClient({ apiUrl: this.options.apiUrl, apiKey: this.options.apiKey });
    const payload = await client.postSystemOne(jevRequestBody(this.options.model, input, this.options.geminiEnabled), signal);

    const usage = JevUsage.safeParse(payload);
    const model = JevModel.safeParse(payload);
    const textOnly = parseJevTextOnly(payload);
    const riskScores = parseJevRiskScores(payload);
    return {
      ...parseJevAnswer(payload),
      risk: maxRisk(riskScores),
      riskScores,
      ...(usage.success
        ? { usage: { inputTokens: usage.data.usage.input_tokens, outputTokens: usage.data.usage.output_tokens } }
        : {}),
      ...(model.success ? { model: model.data.model } : {}),
      ...(textOnly !== undefined ? { textOnly } : {}),
    };
  }
}

/** Request body for one ordinal complexity question. */
export function jevRequestBody(model: string, input: ClassificationInput, geminiEnabled?: boolean): object {
  const extraQuestions = geminiEnabled ? {
    text_answer_suffices: {
      type: 'noul',
      instructions: "The request can be fully and correctly answered with a written reply alone — an explanation, a concept, a plan, a design or architecture discussion, a recommendation, or a review of code or text that is already included in the conversation — and does NOT require reading files that are not in the conversation, searching the repository, running commands or tests, browsing, or creating or editing any file."
    }
  } : {};

  return {
    model,
    state: buildState(input),
    questions: {
      [QUESTION_ID]: {
        type: 'score',
        instructions: 'How complex is this request for an AI coding agent working in the developer repository?',
        criteria: COMPLEXITY_CRITERIA,
      },
      ...RISK_QUESTIONS,
      ...extraQuestions,
    },
  };
}

/** Production parsing: schema check, then levels "0" | "1" | "2" (a missing level counts as 0). */
export function parseJevAnswer(payload: unknown): ComplexityDistribution {
  const { probabilities: p } = JevResponse.parse(payload).answers[QUESTION_ID]!;
  return toDistribution(p['0'] ?? 0, p['1'] ?? 0, p['2'] ?? 0);
}

/** Each risk Noul by question id; throws when any is missing. */
export function parseJevRiskScores(payload: unknown): RiskScores {
  const a = JevRisk.parse(payload).answers;
  return {
    security_sensitive: a.security_sensitive.noul,
    destructive_or_production: a.destructive_or_production.noul,
    requires_inspection: a.requires_inspection.noul,
  };
}

const maxRisk = (scores: RiskScores): number => Math.max(...Object.values(scores));

/** Highest of the risk Nouls; throws when any is missing. */
export function parseJevRisk(payload: unknown): number {
  return maxRisk(parseJevRiskScores(payload));
}

export function parseJevTextOnly(payload: unknown): number | undefined {
  try {
    const parsed = z.looseObject({
      answers: z.looseObject({
        text_answer_suffices: z.looseObject({ noul: z.number().min(0).max(1) }).optional()
      })
    }).safeParse(payload);
    if (parsed.success && parsed.data.answers.text_answer_suffices) {
      return parsed.data.answers.text_answer_suffices.noul;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

/**
 * Strict contract check used by scripts/test-jev-real.ts against production.
 * Stricter than parseJevAnswer on purpose: it flags what the adapter would
 * silently tolerate (missing or extra levels, mass that does not sum to 1).
 * Returns human-readable issues; empty means the contract holds.
 */
export function jevContractIssues(payload: unknown): string[] {
  const parsed = JevResponse.safeParse(payload);
  if (!parsed.success) {
    return parsed.error.issues.map((i) => `schema: ${i.path.join('.') || '(root)'}: ${i.message}`);
  }
  const p = parsed.data.answers[QUESTION_ID]!.probabilities;
  const issues: string[] = [];
  const keys = Object.keys(p).sort();
  for (const level of ['0', '1', '2']) {
    if (!(level in p)) issues.push(`probabilities: level "${level}" missing (got keys ${JSON.stringify(keys)})`);
  }
  const extra = keys.filter((k) => !['0', '1', '2'].includes(k));
  if (extra.length > 0) issues.push(`probabilities: unexpected levels ${JSON.stringify(extra)}`);
  const sum = Object.values(p).reduce((a, b) => a + b, 0);
  if (Math.abs(sum - 1) > 0.02) issues.push(`probabilities: sum is ${sum.toFixed(4)}, expected 1 ± 0.02`);
  const risk = JevRisk.safeParse(payload);
  if (!risk.success) issues.push(...risk.error.issues.map((i) => `risk: ${i.path.join('.')}: ${i.message}`));
  return issues;
}

/** JEV consumes unstructured state; give it the request plus the cheap structural signals. */
export function buildState(input: ClassificationInput): string {
  return [
    `Developer request to an AI coding agent:`,
    input.text,
    ``,
    `Context: conversation turn ${input.turnCount}, ${input.toolCount} tools available, ~${input.estimatedInputTokens} input tokens.`,
  ].join('\n');
}
