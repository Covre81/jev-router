import type { MessagesBody } from '../../routing/messages-body.js';

const SYSTEM_REMINDER = /<system-reminder>([\s\S]*?)<\/system-reminder>/g;
const ENV_OR_HANDOFF_BLOCK = /<(?:env|session_handoff)>[\s\S]*?<\/(?:env|session_handoff)>/g;
const MAX_TOOLS_SECTION_CHARS = 32_000;

function extractSystemEnv(system: unknown): string {
  if (!system) return '';
  const raw =
    typeof system === 'string'
      ? system
      : Array.isArray(system)
        ? system
            .filter((b): b is { type: string; text: string } => Boolean(b && typeof b === 'object' && (b as any).type === 'text' && typeof (b as any).text === 'string'))
            .map((b) => b.text)
            .join('\n')
        : '';
  if (!raw) return '';
  const matches = raw.match(ENV_OR_HANDOFF_BLOCK);
  if (!matches || matches.length === 0) return '';
  return matches.join('\n\n') + '\n\n';
}

/** Keys that cost chars without telling the model how to shape an input. */
const SCHEMA_NOISE = new Set(['description', '$schema', 'additionalProperties', 'title', 'examples']);

/** The whole JSON schema (nesting, enums, array items) minus prose: a flattened one hides the shape of nested inputs. */
function compactSchema(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(compactSchema);
  if (!node || typeof node !== 'object') return node;
  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(node)) {
    // Under `properties` the keys are field names: a field called "description" must survive.
    if (k === 'properties' && v && typeof v === 'object' && !Array.isArray(v)) {
      out[k] = Object.fromEntries(Object.entries(v).map(([name, sub]) => [name, compactSchema(sub)]));
    } else if (!SCHEMA_NOISE.has(k)) {
      out[k] = compactSchema(v);
    }
  }
  return out;
}

function renderClientTools(tools: unknown[]): string {
  const lines: string[] = [
    'Available Client Tools (emit <jev_tool_use>{"name":"ToolName","input":{...}}</jev_tool_use> to call):',
  ];
  let totalChars = lines[0]!.length;

  for (const raw of tools) {
    if (!raw || typeof raw !== 'object') continue;
    const t = raw as { name?: unknown; description?: unknown; input_schema?: unknown };
    if (typeof t.name !== 'string' || !t.name) continue;
    const desc =
      typeof t.description === 'string'
        ? t.description.replace(/\s+/g, ' ').trim().slice(0, 220)
        : '';
    const schemaStr = t.input_schema && typeof t.input_schema === 'object' ? JSON.stringify(compactSchema(t.input_schema)) : '{}';
    const line = `- ${t.name}: ${desc} | input: ${schemaStr}`;
    if (totalChars + line.length + 1 > MAX_TOOLS_SECTION_CHARS) break;
    lines.push(line);
    totalChars += line.length + 1;
  }

  return lines.length > 1 ? lines.join('\n') + '\n\n' : '';
}

export function renderGeminiPrompt(
  body: MessagesBody,
  maxChars: number,
  opts: { readonly variant?: 'gemini' | 'cursor' } = {},
): string | undefined {
  const rawTools = Array.isArray((body as any).tools) ? ((body as any).tools as unknown[]) : [];
  const toolsSection = rawTools.length > 0 ? renderClientTools(rawTools) : '';
  const cursor = opts.variant === 'cursor';

  const preambleText = toolsSection
    ? 'You are answering ONE turn of a conversation between a developer and a coding assistant.\n' +
      (cursor
        ? 'CRITICAL RULE FOR TOOLS: Do NOT use your own tools — do not read or write files, run commands or browse with them. Any use of your own tools counts as failure.\n'
        : 'CRITICAL RULE FOR TOOLS: Do NOT invoke your built-in platform tools (such as view_file, write_to_file, replace_file_content, run_command, grep_search, find_by_name) directly — direct built-in tool calls are blocked by the sandbox.\n') +
      'Instead, you have full permission to use the client tools listed under "Available Client Tools" below by emitting one or more XML blocks in your text response using this exact format:\n' +
      '<jev_tool_use>{"name": "ToolName", "input": {"param": "value"}}</jev_tool_use>\n' +
      'When you emit <jev_tool_use>...</jev_tool_use>, the client executes the tool in the real workspace and returns [tool result: ...] in the next turn so you can continue.\n' +
      'If you do not need any tool for this turn, reply normally in the same language the user writes in using Markdown.\n\n' +
      toolsSection
    : cursor
      ? 'You are answering ONE turn of a conversation between a developer and a coding assistant. You have NO client tools — do not use your own tools to read or write files, run commands or browse (any use of your own tools counts as failure). Answer the LAST user message using only the transcript. Reply in the same language the user writes in. Use Markdown. If a correct answer genuinely requires reading files not shown, running commands or editing files, reply with exactly JEV_NEEDS_TOOLS and nothing else.\n\n'
      : 'You are answering ONE turn of a conversation between a developer and a coding assistant. You have NO tools — do not try to read or write files, run commands or browse (any tool use is blocked and counts as failure). Answer the LAST user message using only the transcript. Reply in the same language the user writes in. Use Markdown. If a correct answer genuinely requires reading files not shown, running commands or editing files, reply with exactly JEV_NEEDS_TOOLS and nothing else.\n\n';

  const envSection = extractSystemEnv((body as any).system);

  const reminderSet = new Set<string>();
  const reminderList: string[] = []; // kept in order of appearance
  const messages: string[] = [];
  
  const extractReminders = (content: string): string => {
    const matches = [...content.matchAll(SYSTEM_REMINDER)];
    for (const match of matches) {
      let inner = match[1] || '';
      if (inner.length > 4000) inner = inner.substring(0, 4000) + '...';
      const rendered = `<system-reminder>${inner}</system-reminder>`;
      if (!reminderSet.has(rendered)) {
        reminderSet.add(rendered);
        reminderList.push(rendered);
      }
    }
    return content.replace(SYSTEM_REMINDER, '').trim();
  };

  for (const m of body.messages) {
    if (m.role === 'system') continue;
    
    let text = `### ${m.role === 'user' ? 'User' : 'Assistant'}\n`;
    if (typeof m.content === 'string') {
      let content = m.content;
      if (m.role === 'user') {
        content = extractReminders(content);
      }
      text += content + '\n\n';
    } else {
      for (const b of m.content) {
        if (b.type === 'text' && typeof b.text === 'string') {
          let content = b.text;
          if (m.role === 'user') {
            content = extractReminders(content);
          }
          text += content + '\n';
        } else if (b.type === 'tool_use') {
          const input = JSON.stringify(b.input || {}).substring(0, 2000);
          text += `[assistant called tool ${(b as any).name} with input: ${input}]\n`;
        } else if (b.type === 'tool_result') {
          const bAny = b as any;
          let content = '';
          if (typeof bAny.content === 'string') {
            content = bAny.content;
          } else if (Array.isArray(bAny.content)) {
            content = bAny.content
              .filter((blk: any) => blk.type === 'text')
              .map((blk: any) => blk.text)
              .join('\n');
          } else {
            content = JSON.stringify(bAny.content || {});
          }
          text += `[tool result: ${content.substring(0, 12000)}]\n`;
        } else if (b.type === 'image') {
          text += `[image omitted]\n`;
        } else if (b.type === 'document') {
          text += `[document omitted]\n`;
        }
      }
      text += '\n';
    }
    messages.push(text);
  }

  // Cap reminders to 12000 chars total, keeping the most recent ones
  let finalReminders = '';
  if (reminderList.length > 0) {
    let remLen = 0;
    const keptReminders: string[] = [];
    for (let i = reminderList.length - 1; i >= 0; i--) {
      const r = reminderList[i]! + '\n\n';
      if (remLen + r.length > 12000) break;
      remLen += r.length;
      keptReminders.unshift(r);
    }
    if (keptReminders.length > 0) {
      finalReminders = "System reminders:\n" + keptReminders.join('');
    }
  }

  const preamble = preambleText + envSection + finalReminders;

  // Budget
  let prompt = preamble;
  const kept: string[] = [];
  let chars = preamble.length;
  
  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (chars + msg.length > maxChars) {
      if (kept.length === 0) return undefined; // latest message alone doesn't fit
      kept.unshift("[earlier conversation omitted]\n\n");
      break;
    }
    chars += msg.length;
    kept.unshift(msg);
  }
  
  prompt += kept.join('');
  return prompt.trim();
}

