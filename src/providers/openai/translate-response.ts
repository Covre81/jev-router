import { randomUUID } from 'node:crypto';
import { z } from 'zod';
import { sseEvent } from './sse.js';

/**
 * OpenAI Chat Completions → Anthropic Messages, for both the JSON body and the
 * SSE stream. The client (Claude Code) only speaks Anthropic, so this half of
 * the translation is mandatory, not optional.
 */
export class TranslationError extends Error {
  override readonly name = 'TranslationError';
}

const Usage = z.looseObject({
  prompt_tokens: z.number().optional(),
  completion_tokens: z.number().optional(),
});

const ToolCall = z.looseObject({
  id: z.string().nullish(),
  function: z.looseObject({ name: z.string().nullish(), arguments: z.string().nullish() }),
});

const Completion = z.looseObject({
  choices: z
    .array(
      z.looseObject({
        message: z.looseObject({ content: z.string().nullish(), tool_calls: z.array(ToolCall).nullish() }),
        finish_reason: z.string().nullish(),
      }),
    )
    .min(1),
  usage: Usage.nullish(),
});

const Chunk = z.looseObject({
  choices: z
    .array(
      z.looseObject({
        delta: z
          .looseObject({
            content: z.string().nullish(),
            tool_calls: z.array(ToolCall.extend({ index: z.number() }).partial({ function: true })).nullish(),
          })
          .nullish(),
        finish_reason: z.string().nullish(),
      }),
    )
    .nullish(),
  usage: Usage.nullish(),
});

type StopReason = 'end_turn' | 'max_tokens' | 'tool_use' | 'refusal';

function stopReason(finish: string | null | undefined, hasToolCalls: boolean): StopReason {
  if (hasToolCalls) return 'tool_use';
  if (finish === 'length') return 'max_tokens';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** Anthropic validates tool_use ids against ^[a-zA-Z0-9_-]+$ when the history is replayed. */
function toolUseId(id: string | null | undefined): string {
  const clean = (id ?? '').replace(/[^a-zA-Z0-9_-]/g, '_');
  return clean || `toolu_${randomUUID().replaceAll('-', '')}`;
}

function parseArguments(raw: string | null | undefined, name: string): unknown {
  try {
    const input: unknown = JSON.parse(raw || '{}');
    if (input === null || typeof input !== 'object' || Array.isArray(input)) throw new Error('not an object');
    return input;
  } catch {
    throw new TranslationError(`tool call "${name}" produced invalid JSON arguments`);
  }
}

const messageId = () => `msg_jev_${randomUUID().replaceAll('-', '')}`;

const usageOf = (u: z.infer<typeof Usage> | null | undefined) => ({
  input_tokens: u?.prompt_tokens ?? 0,
  output_tokens: u?.completion_tokens ?? 0,
});

export function toAnthropicMessage(
  payload: unknown,
  model: string,
): { readonly content: readonly object[]; readonly stop_reason: StopReason; readonly [key: string]: unknown } {
  const parsed = Completion.safeParse(payload);
  if (!parsed.success) throw new TranslationError('unexpected chat completion shape');
  const choice = parsed.data.choices[0]!;
  const toolCalls = choice.message.tool_calls ?? [];

  const content: object[] = [];
  if (choice.message.content) content.push({ type: 'text', text: choice.message.content });
  for (const call of toolCalls) {
    const name = call.function.name ?? '';
    content.push({ type: 'tool_use', id: toolUseId(call.id), name, input: parseArguments(call.function.arguments, name) });
  }

  return {
    id: messageId(),
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReason(choice.finish_reason, toolCalls.length > 0),
    stop_sequence: null,
    usage: usageOf(parsed.data.usage),
  };
}

interface PendingToolCall {
  id: string | undefined;
  name: string;
  args: string;
}

/**
 * Streams text deltas live; buffers tool calls and emits each as one complete
 * tool_use block at the end. Tool arguments are short, and buffering
 * guarantees a well-formed block sequence (Claude Code stops reading a stream
 * that references a closed block) and lets us validate the JSON before the
 * client can execute it.
 */
export async function* toAnthropicStream(dataEvents: AsyncIterable<string>, model: string): AsyncGenerator<string> {
  yield sseEvent('message_start', {
    message: {
      id: messageId(),
      type: 'message',
      role: 'assistant',
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: 0, output_tokens: 0 },
    },
  });

  let index = 0;
  let textOpen = false;
  let finish: string | null | undefined;
  let usage: z.infer<typeof Usage> | null | undefined;
  const tools = new Map<number, PendingToolCall>();

  for await (const data of dataEvents) {
    if (data === '[DONE]') break;
    let json: unknown;
    try {
      json = JSON.parse(data);
    } catch {
      throw new TranslationError('malformed stream chunk');
    }
    const chunk = Chunk.safeParse(json);
    if (!chunk.success) throw new TranslationError('unexpected stream chunk shape');
    if (chunk.data.usage) usage = chunk.data.usage;

    for (const choice of chunk.data.choices ?? []) {
      const text = choice.delta?.content;
      if (text) {
        if (!textOpen) {
          yield sseEvent('content_block_start', { index, content_block: { type: 'text', text: '' } });
          textOpen = true;
        }
        yield sseEvent('content_block_delta', { index, delta: { type: 'text_delta', text } });
      }
      for (const call of choice.delta?.tool_calls ?? []) {
        const pending = tools.get(call.index) ?? { id: undefined, name: '', args: '' };
        pending.id ??= call.id ?? undefined;
        pending.name += call.function?.name ?? '';
        pending.args += call.function?.arguments ?? '';
        tools.set(call.index, pending);
      }
      if (choice.finish_reason) finish = choice.finish_reason;
    }
  }

  if (finish === undefined) throw new TranslationError('stream ended without a finish_reason');

  if (textOpen) {
    yield sseEvent('content_block_stop', { index });
    index++;
  }
  for (const [, call] of [...tools.entries()].sort(([a], [b]) => a - b)) {
    const args = JSON.stringify(parseArguments(call.args, call.name));
    yield sseEvent('content_block_start', {
      index,
      content_block: { type: 'tool_use', id: toolUseId(call.id), name: call.name, input: {} },
    });
    yield sseEvent('content_block_delta', { index, delta: { type: 'input_json_delta', partial_json: args } });
    yield sseEvent('content_block_stop', { index });
    index++;
  }

  yield sseEvent('message_delta', {
    delta: { stop_reason: stopReason(finish, tools.size > 0), stop_sequence: null },
    usage: usageOf(usage),
  });
  yield sseEvent('message_stop', {});
}

/** Ends a broken stream with an Anthropic `error` event instead of a dropped socket. */
export async function* withErrorEvent(
  events: AsyncIterable<string>,
  onFailure: (err: unknown) => void,
): AsyncGenerator<string> {
  try {
    yield* events;
  } catch (err) {
    onFailure(err);
    yield sseEvent('error', {
      error: { type: 'api_error', message: `jev-router: cheap provider stream failed (${String(err)})` },
    });
  }
}
