import type { Provider, ProviderRequest, ProviderResult } from '../provider.js';
import { anthropicResponseFromCliText } from '../gemini/provider.js';
import { renderGeminiPrompt } from '../gemini/prompt.js';
import { runCursorAgent, type CursorRunnerOptions, type SpawnCursor } from './agent-runner.js';

export interface CursorCliProviderOptions {
  readonly bin: string;
  readonly model: string;
  readonly timeoutMs: number;
  readonly firstByteTimeoutMs?: number;
  readonly idleTimeoutMs?: number;
  readonly maxPromptChars: number;
  readonly spawnFn?: SpawnCursor;
}

export class CursorCliProvider implements Provider {
  readonly name = 'cursor-cli';

  constructor(private readonly opts: CursorCliProviderOptions) {}

  async send(req: ProviderRequest): Promise<ProviderResult> {
    if (!req.body) return { kind: 'unavailable', reason: 'no body' };

    const prompt = renderGeminiPrompt(req.body, this.opts.maxPromptChars, { variant: 'cursor' });
    if (!prompt) return { kind: 'unavailable', reason: 'transcript budget exceeded' };
    // A cut transcript answers without the history; Gemini's budget keeps it whole, so decline and let it take the turn.
    // ponytail: marker detection; a transcript quoting the marker also declines (costs only a Gemini hop)
    if (prompt.includes('[earlier conversation omitted]')) return { kind: 'unavailable', reason: 'transcript budget exceeded' };
    // Windows command-line cap: over-budget prompts never spawn (renderGeminiPrompt already enforces maxPromptChars).
    if (prompt.length > this.opts.maxPromptChars) return { kind: 'unavailable', reason: 'transcript budget exceeded' };

    const runnerOpts: CursorRunnerOptions = {
      bin: this.opts.bin,
      model: this.opts.model,
      timeoutMs: this.opts.timeoutMs,
      firstByteTimeoutMs: this.opts.firstByteTimeoutMs,
      idleTimeoutMs: this.opts.idleTimeoutMs,
      signal: req.signal,
    };
    const res = await runCursorAgent(prompt, runnerOpts, this.opts.spawnFn);
    if (!res.ok) return { kind: 'unavailable', reason: res.reason };
    return anthropicResponseFromCliText({ rawText: res.text, usage: res.usage, model: this.opts.model, stream: (req.body as { stream?: unknown }).stream === true });
  }
}
