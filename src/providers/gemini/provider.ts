import { Readable } from 'node:stream';
import { randomUUID } from 'node:crypto';
import type { Provider, ProviderRequest, ProviderResult } from '../provider.js';
import { runAgy, type AgyRunnerOptions, type SpawnAgy } from './agy-runner.js';
import { renderGeminiPrompt } from './prompt.js';
import { sseEvent } from '../openai/sse.js';

export interface ParsedGeminiToolCall {
  readonly id: string;
  readonly name: string;
  readonly input: Record<string, unknown>;
}

const JEV_TOOL_USE_RE = /<jev_tool_use>\s*([\s\S]*?)\s*<\/jev_tool_use>/g;

export function parseGeminiToolCalls(rawText: string): {
  text: string;
  toolCalls: ParsedGeminiToolCall[];
} {
  const toolCalls: ParsedGeminiToolCall[] = [];
  const cleaned = rawText.replace(JEV_TOOL_USE_RE, (fullMatch, inner: string) => {
    const trimmed = inner.trim().replace(/^```(?:json)?\s*/i, '').replace(/\s*```$/, '').trim();
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === 'object' && typeof parsed.name === 'string' && parsed.name.length > 0) {
        const input =
          parsed.input && typeof parsed.input === 'object' && !Array.isArray(parsed.input)
            ? (parsed.input as Record<string, unknown>)
            : {};
        toolCalls.push({
          id: `toolu_jev_${randomUUID().replace(/-/g, '').slice(0, 20)}`,
          name: parsed.name,
          input,
        });
        return '';
      }
    } catch {
      // Leave malformed block as text rather than dropping it silently.
    }
    return fullMatch;
  });

  return {
    text: cleaned.trim(),
    toolCalls,
  };
}

/** Anthropic JSON/SSE synthesis shared by the CLI-backed tiers (Gemini, Cursor). */
export function anthropicResponseFromCliText(opts: {
  readonly rawText: string;
  readonly usage: { readonly inputTokens: number; readonly outputTokens: number };
  readonly model: string;
  readonly stream: boolean | undefined;
}): ProviderResult {
  const rawText = opts.rawText.trim();
  const { text, toolCalls } = parseGeminiToolCalls(rawText);
  if (toolCalls.length === 0 && (text.startsWith('JEV_NEEDS_TOOLS') || text === 'JEV_NEEDS_TOOLS')) {
    return { kind: 'unavailable', reason: 'model asked for tools' };
  }
  const stopReason = toolCalls.length > 0 ? 'tool_use' : 'end_turn';
  const usage = opts.usage;
  const msgId = `msg_jev_${randomUUID()}`;
  const headers: Record<string, string> = {
    'content-type': opts.stream ? 'text/event-stream' : 'application/json',
  };
  if (opts.stream) {
    headers['cache-control'] = 'no-cache';
    const model = opts.model;
    async function* generateStream() {
      yield sseEvent('message_start', {
        message: {
          id: msgId,
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: usage.inputTokens, output_tokens: 0 },
        },
      });
      let index = 0;
      if (text.length > 0 || toolCalls.length === 0) {
        yield sseEvent('content_block_start', { index, content_block: { type: 'text', text: '' } });
        const chunkLen = 2000;
        for (let i = 0; i < text.length; i += chunkLen) {
          yield sseEvent('content_block_delta', { index, delta: { type: 'text_delta', text: text.slice(i, i + chunkLen) } });
        }
        yield sseEvent('content_block_stop', { index });
        index++;
      }
      for (const call of toolCalls) {
        yield sseEvent('content_block_start', { index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } });
        yield sseEvent('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: JSON.stringify(call.input) } });
        yield sseEvent('content_block_stop', { index });
        index++;
      }
      yield sseEvent('message_delta', { delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: usage.outputTokens } });
      yield sseEvent('message_stop', {});
    }
    return { kind: 'response', status: 200, headers, body: Readable.from(generateStream()) };
  }
  const content: Array<Record<string, unknown>> = [];
  if (text.length > 0 || toolCalls.length === 0) content.push({ type: 'text', text });
  for (const call of toolCalls) content.push({ type: 'tool_use', id: call.id, name: call.name, input: call.input });
  const responseJson = {
    id: msgId,
    type: 'message',
    role: 'assistant',
    model: opts.model,
    content,
    stop_reason: stopReason,
    stop_sequence: null,
    usage: { input_tokens: usage.inputTokens, output_tokens: usage.outputTokens },
  };
  return { kind: 'response', status: 200, headers, body: Readable.from([JSON.stringify(responseJson)]) };
}

export class GeminiCliProvider implements Provider {
  readonly name = 'gemini-cli';

  constructor(
    private readonly opts: {
      readonly bin: string;
      readonly model: string;
      readonly timeoutMs: number;
      readonly home: string;
      readonly maxPromptChars: number;
      readonly spawnFn?: SpawnAgy;
    }
  ) {}

  async send(req: ProviderRequest): Promise<ProviderResult> {
    if (!req.body) {
      return { kind: 'unavailable', reason: 'no body' };
    }

    const prompt = renderGeminiPrompt(req.body, this.opts.maxPromptChars);
    if (!prompt) {
      return { kind: 'unavailable', reason: 'transcript budget exceeded' };
    }

    const runnerOpts: AgyRunnerOptions = {
      bin: this.opts.bin,
      model: this.opts.model,
      timeoutMs: this.opts.timeoutMs,
      home: this.opts.home,
      signal: req.signal,
    };

    const res = await runAgy(prompt, runnerOpts, this.opts.spawnFn);

    if (!res.ok) {
      return { kind: 'unavailable', reason: res.reason };
    }

    return anthropicResponseFromCliText({ rawText: res.text, usage: res.usage, model: this.opts.model, stream: (req.body as { stream?: unknown }).stream === true });
  }
}

