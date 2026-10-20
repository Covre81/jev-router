import { createHash } from 'node:crypto';
import { z } from 'zod';

/**
 * Only the fields the router reads are validated. Everything else is an open
 * list that must reach Anthropic untouched (Claude Code gateway contract), hence
 * `looseObject` — unknown keys survive parse → mutate → serialize.
 */
const ContentBlock = z.looseObject({ type: z.string() });

const Message = z.looseObject({
  role: z.string(),
  content: z.union([z.string(), z.array(ContentBlock)]),
});

export const MessagesBody = z.looseObject({
  model: z.string().min(1),
  messages: z.array(Message).min(1),
  max_tokens: z.number().int().positive().optional(),
  thinking: z.looseObject({ type: z.string() }).optional(),
  output_config: z.record(z.string(), z.unknown()).optional(),
  speed: z.string().optional(),
  tool_choice: z.looseObject({ type: z.string() }).optional(),
});

export type MessagesBody = z.infer<typeof MessagesBody>;
type Message = z.infer<typeof Message>;

export function parseMessagesBody(raw: Buffer): MessagesBody | undefined {
  try {
    const result = MessagesBody.safeParse(JSON.parse(raw.toString('utf8')));
    return result.success ? result.data : undefined;
  } catch {
    return undefined;
  }
}

const SYSTEM_REMINDER = /<system-reminder>[\s\S]*?<\/system-reminder>/g;

/** Claude Code inlines hook output as `system` messages; they are not conversation turns. */
export function turns(body: MessagesBody): Message[] {
  return body.messages.filter((m) => m.role !== 'system');
}

export function hasToolResult(message: Message): boolean {
  return Array.isArray(message.content) && message.content.some((b) => b.type === 'tool_result');
}

export function humanTextOf(message: Message): string | undefined {
  const text =
    typeof message.content === 'string'
      ? message.content
      : message.content
          .filter((b): b is typeof b & { text: string } => b.type === 'text' && typeof b.text === 'string')
          .map((b) => b.text)
          .join('\n');

  const cleaned = text.replace(SYSTEM_REMINDER, '').trim();
  return cleaned.length > 0 ? cleaned : undefined;
}

/**
 * Text typed by the human in the latest turn, or `undefined` when the latest
 * user message is a tool-result continuation of the agent loop (not a new task).
 */
export function latestHumanText(body: MessagesBody): string | undefined {
  const last = turns(body).at(-1);
  if (!last || last.role !== 'user' || hasToolResult(last)) return undefined;

  return humanTextOf(last);
}

/**
 * Human text driving the conversation: the most recent `user` turn without a
 * tool_result and with non-empty text. Unlike `latestHumanText`, this scans
 * past tool-result continuations to the human request driving the tool loop.
 */
export function lastHumanText(body: MessagesBody): string | undefined {
  const all = turns(body);
  for (let i = all.length - 1; i >= 0; i--) {
    const turn = all[i]!;
    if (turn.role !== 'user' || hasToolResult(turn)) continue;
    const text = humanTextOf(turn);
    if (text) return text;
  }
  return undefined;
}

export function isFreshConversation(body: MessagesBody): boolean {
  return turns(body).length === 1;
}

/** Stable per-conversation key for clients that send no session header. */
export function conversationFingerprint(body: MessagesBody): string {
  return createHash('sha256').update(JSON.stringify(body.messages[0])).digest('hex').slice(0, 32);
}

/**
 * Deliberately pessimistic token estimate (3 bytes/token; real JSON+code is
 * closer to 3.5–4). Over-estimating only excludes a small-window model earlier,
 * which is the safe direction for a context-window guard.
 */
export function estimateInputTokens(rawByteLength: number): number {
  return Math.ceil(rawByteLength / 3);
}

export function hasImageDocumentOrToolChoice(body: MessagesBody): boolean {
  if (body.tool_choice?.type === 'any' || body.tool_choice?.type === 'tool') {
    return true;
  }
  const last = turns(body).at(-1);
  if (!last || last.role !== 'user') return false;
  if (Array.isArray(last.content)) {
    return last.content.some((b) => b.type === 'image' || b.type === 'document');
  }
  return false;
}
