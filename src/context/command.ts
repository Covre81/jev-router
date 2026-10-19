import { z } from 'zod';
import { runPipeline, type PipelineConfig } from './pipeline.js';

const HookConfigSchema = z.looseObject({
  CONTEXT_MODE: z.enum(['inject', 'shadow', 'off']).default('inject'),
  CONTEXT_L_CURRENT: z.coerce.number().default(0.8),
  CONTEXT_L_OTHER: z.coerce.number().default(0.9),
  CONTEXT_MAX_ITEMS: z.coerce.number().int().default(5),
  CONTEXT_MAX_CHARS: z.coerce.number().int().default(1500),
  CONTEXT_TIMEOUT_MS: z.coerce.number().int().default(3000),
  CONTEXT_HOOK_BUDGET_MS: z.unknown().transform(val => {
    if (val === undefined) return 5000;
    const num = Number(val);
    if (isNaN(num)) return 5000;
    return Math.max(1000, Math.min(7000, Math.floor(num)));
  }).default(5000),
  CONTEXT_GRAPH_ROOT: z.string().optional(),
  CONTEXT_AI_MEMORY_BIN: z.string().default('ai-memory'),
  TYPESAFE_API_KEY: z.string().optional(),
  JEV_API_URL: z.string().default('https://api.typesafe.ai/v1/systemone'),
  JEV_MODEL: z.string().default('jev-latest'),
});

export async function runContextHook(explicitEnvFile?: string, exitFn: (code: number) => void = process.exit): Promise<number> {
  let watchdog: NodeJS.Timeout | undefined;
  let stdoutWritten = false;
  let isAntigravityInput = false;
  try {
    // 1. Parse config (using custom schema)
    const rawConfig = HookConfigSchema.parse(process.env);
    const budgetMs = rawConfig.CONTEXT_HOOK_BUDGET_MS;

    watchdog = setTimeout(() => {
      if (!stdoutWritten) {
        if (isAntigravityInput) {
          stdoutWritten = true;
          process.stdout.write('{}');
        }
        process.stderr.write(`[context] hook budget of ${budgetMs} ms exceeded, exiting without context\n`);
        exitFn(0);
      }
    }, budgetMs);
    watchdog.unref();

    const config: PipelineConfig = {
      mode: rawConfig.CONTEXT_MODE,
      lCurrent: rawConfig.CONTEXT_L_CURRENT,
      lOther: rawConfig.CONTEXT_L_OTHER,
      maxItems: rawConfig.CONTEXT_MAX_ITEMS,
      maxChars: rawConfig.CONTEXT_MAX_CHARS,
      timeoutMs: Math.min(rawConfig.CONTEXT_TIMEOUT_MS, Math.max(500, budgetMs - 1500)),
      graphRoot: rawConfig.CONTEXT_GRAPH_ROOT,
      aiMemoryBin: rawConfig.CONTEXT_AI_MEMORY_BIN,
      apiKey: rawConfig.TYPESAFE_API_KEY,
      apiUrl: rawConfig.JEV_API_URL,
      model: rawConfig.JEV_MODEL,
    };

    // 2. Read stdin
    let inputStr = '';
    process.stdin.setEncoding('utf8');
    for await (const chunk of process.stdin) {
      inputStr += chunk;
    }

    if (inputStr.startsWith('\ufeff')) {
      inputStr = inputStr.slice(1);
    }

    if (!inputStr.trim()) {
      process.stderr.write('[context] Empty stdin received\n');
      return 0; // exit 0 without saving
    }

    let rawInput: unknown;
    try {
      rawInput = JSON.parse(inputStr);
    } catch (err: any) {
      process.stderr.write(`[context] Invalid JSON on stdin: ${err.message}\n`);
      return 0; // exit 0 without saving
    }

    isAntigravityInput =
      typeof rawInput === 'object' &&
      rawInput !== null &&
      'conversationId' in rawInput &&
      !('session_id' in rawInput);

    // 3. Run Pipeline
    const result = await runPipeline(rawInput, config);

    // 4. Output to stdout and write status/diagnostics to stderr if needed
    if (result.stdout) {
      stdoutWritten = true;
      process.stdout.write(result.stdout);
    }

    if (result.outcome !== 'ok' && result.outcome !== 'skipped_origin' && result.outcome !== 'skipped_cwd') {
      process.stderr.write(`[context] Pipeline completed with outcome: ${result.outcome}${result.error ? `: ${result.error}` : ''}\n`);
    }

    return 0;
  } catch (err: any) {
    if (isAntigravityInput && !stdoutWritten) {
      stdoutWritten = true;
      process.stdout.write('{}');
    }
    process.stderr.write(`[context] Uncaught pipeline error: ${err.message}\n`);
    return 0; // Always exit 0 as per guidelines
  } finally {
    if (watchdog) clearTimeout(watchdog);
  }
}
