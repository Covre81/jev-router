import { Readable } from 'node:stream';
import { request } from 'undici';
import type { Provider, ProviderRequest, ProviderResult } from '../provider.js';
import { readSseData } from './sse.js';
import { NotTranslatableError, toChatCompletion } from './translate-request.js';
import { toAnthropicMessage, toAnthropicStream, TranslationError, withErrorEvent } from './translate-response.js';

export interface OpenAICompatibleOptions {
  /** e.g. https://api.groq.com/openai/v1 or https://openrouter.ai/api/v1 */
  readonly baseUrl: string;
  readonly apiKey: string;
  readonly model: string;
  readonly maxOutputTokens: number;
  readonly timeoutMs: number;
}

/**
 * Cheap provider: any OpenAI-compatible `/chat/completions` endpoint, wrapped
 * so the client still sees the Anthropic Messages wire format.
 *
 * Every failure before the first byte reaches the client is reported as
 * `unavailable` (never thrown), so the caller can fail over to the primary.
 */
export class OpenAICompatibleProvider implements Provider {
  readonly name: string;

  constructor(private readonly options: OpenAICompatibleOptions) {
    this.name = `openai-compatible:${new URL(options.baseUrl).host}`;
  }

  async send(req: ProviderRequest): Promise<ProviderResult> {
    if (!req.body) return { kind: 'unavailable', reason: 'unparseable body' };

    let chat;
    try {
      chat = toChatCompletion(req.body, { model: this.options.model, maxOutputTokens: this.options.maxOutputTokens });
    } catch (err) {
      if (err instanceof NotTranslatableError) return { kind: 'unavailable', reason: `not translatable: ${err.message}` };
      throw err;
    }

    let res;
    try {
      res = await request(`${this.options.baseUrl}/chat/completions`, {
        method: 'POST',
        headers: { authorization: `Bearer ${this.options.apiKey}`, 'content-type': 'application/json' },
        body: JSON.stringify(chat),
        signal: req.signal,
        headersTimeout: req.headersTimeoutMs ?? this.options.timeoutMs,
        bodyTimeout: this.options.timeoutMs,
      });
    } catch (err) {
      if (req.signal.aborted) throw err;
      return { kind: 'unavailable', reason: `network: ${(err as Error).message}` };
    }

    if (res.statusCode !== 200) {
      const detail = (await res.body.text()).slice(0, 300);
      return { kind: 'unavailable', reason: `HTTP ${res.statusCode}: ${detail}` };
    }

    if (!chat.stream) {
      try {
        const message = toAnthropicMessage(await res.body.json(), this.options.model);
        // A reasoning model can spend max_tokens thinking and answer nothing.
        if (message.content.length === 0) return { kind: 'unavailable', reason: `empty response (${message.stop_reason})` };
        return {
          kind: 'response',
          status: 200,
          headers: { 'content-type': 'application/json' },
          body: Readable.from([JSON.stringify(message)]),
        };
      } catch (err) {
        if (err instanceof TranslationError || err instanceof SyntaxError) {
          return { kind: 'unavailable', reason: `bad response: ${err.message}` };
        }
        throw err;
      }
    }

    const events = withErrorEvent(toAnthropicStream(readSseData(res.body), this.options.model), () => {
      if (!req.signal.aborted) req.onStreamFailure?.();
    });
    // ponytail: hold events until the first content block, so a reasoning-only answer
    // fails over before the client sees a byte; after that, text streams live as before.
    // No content block before message_delta, an error event or the end = nothing for the client.
    const it = events[Symbol.asyncIterator]();
    const head: string[] = [];
    let hasContent = false;
    for (let next = await it.next(); !next.done; next = await it.next()) {
      head.push(next.value);
      if (next.value.startsWith('event: content_block_start')) {
        hasContent = true;
        break;
      }
      if (next.value.startsWith('event: message_delta') || next.value.startsWith('event: error')) break;
    }
    if (!hasContent) {
      await it.return(undefined);
      const error = head.find((e) => e.startsWith('event: error'));
      const detail = error ? (JSON.parse(error.slice(error.indexOf('data: ') + 6)) as { error: { message: string } }).error.message : '';
      return { kind: 'unavailable', reason: error ? `stream failed before content: ${detail}` : 'empty response (stream)' };
    }
    async function* replay(): AsyncGenerator<string> {
      yield* head;
      for (let next = await it.next(); !next.done; next = await it.next()) yield next.value;
    }
    return {
      kind: 'response',
      status: 200,
      headers: { 'content-type': 'text/event-stream', 'cache-control': 'no-cache' },
      body: Readable.from(replay()),
    };
  }
}
