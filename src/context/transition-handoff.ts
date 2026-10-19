import { spawn } from 'node:child_process';
import { cleanFtsQuery, findAiMemoryToml } from './memory-source.js';
import type { Tier } from '../domain/policy.js';
import type { MessagesBody } from '../routing/messages-body.js';

const SYSTEM_REMINDER_RE = /<system-reminder>[\s\S]*?<\/system-reminder>/g;
const WORKING_DIR_RE = /Working directory:\s*([^\r\n<]+)/i;
const HANDOFF_PAGE_PATH = 'working/jev-session-handoff.md';
const DEFAULT_MAX_HANDOFF_CHARS = 5_000;

const FILE_INPUT_KEYS = [
  'file_path',
  'path',
  'TargetFile',
  'AbsolutePath',
  'notebook_path',
  'DirectoryPath',
  'SearchPath',
] as const;

export type ExecAiMemoryFn = (
  bin: string,
  args: string[],
  stdin?: string,
  timeoutMs?: number,
) => Promise<string>;

export function defaultExecAiMemory(
  bin: string,
  args: string[],
  stdin?: string,
  timeoutMs = 450,
): Promise<string> {
  return new Promise((resolve, reject) => {
    let settled = false;
    const child = spawn(bin, args, {
      stdio: ['pipe', 'pipe', 'pipe'],
      windowsHide: true,
    });

    let stdout = '';
    let stderr = '';

    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        try {
          child.kill();
        } catch {}
        reject(new Error(`ai-memory timeout after ${timeoutMs}ms`));
      }
    }, timeoutMs);
    timer.unref?.();

    child.stdout.on('data', (chunk: Buffer) => {
      stdout += chunk.toString('utf8');
    });
    child.stderr.on('data', (chunk: Buffer) => {
      stderr += chunk.toString('utf8');
    });
    child.on('error', (err) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        reject(err);
      }
    });
    child.on('close', (code) => {
      if (!settled) {
        settled = true;
        clearTimeout(timer);
        if (code === 0) resolve(stdout);
        else reject(new Error(`ai-memory exit ${code}: ${stderr.slice(0, 200)}`));
      }
    });

    if (stdin !== undefined) {
      child.stdin.write(stdin, 'utf8');
    }
    child.stdin.end();
  });
}

export function extractWorkingDirFromSystem(system: unknown): string | undefined {
  if (!system) return undefined;
  const raw =
    typeof system === 'string'
      ? system
      : Array.isArray(system)
        ? system
            .filter(
              (b): b is { type: string; text: string } =>
                Boolean(b && typeof b === 'object' && (b as any).type === 'text' && typeof (b as any).text === 'string'),
            )
            .map((b) => b.text)
            .join('\n')
        : '';
  if (!raw) return undefined;
  const match = WORKING_DIR_RE.exec(raw);
  return match?.[1]?.trim() || undefined;
}

export interface TranscriptSummary {
  readonly text: string;
  readonly filesTouched: readonly string[];
  readonly lastQuery: string;
}

function cleanText(raw: string): string {
  return raw.replace(SYSTEM_REMINDER_RE, '').replace(/\s+/g, ' ').trim();
}

function extractFilesFromInput(input: unknown, out: Set<string>): void {
  if (!input || typeof input !== 'object') return;
  const rec = input as Record<string, unknown>;
  for (const key of FILE_INPUT_KEYS) {
    const val = rec[key];
    if (typeof val === 'string' && val.trim().length > 0) {
      out.add(val.trim());
    }
  }
}

export function summarizeRecentTranscript(
  messages: MessagesBody['messages'],
  maxTurns = 4,
): TranscriptSummary {
  const nonSystem = messages.filter((m) => m.role !== 'system');
  const filesSet = new Set<string>();

  // Group into turns starting at each user message
  const groups: { user?: (typeof nonSystem)[number]; assistants: (typeof nonSystem)[number][] }[] = [];
  for (const msg of nonSystem) {
    if (msg.role === 'user') {
      groups.push({ user: msg, assistants: [] });
    } else if (msg.role === 'assistant') {
      if (groups.length === 0) {
        groups.push({ assistants: [msg] });
      } else {
        groups[groups.length - 1]!.assistants.push(msg);
      }
    }
  }

  // Extract touched files across all recent groups (up to last 8 groups) for continuity
  for (const g of groups.slice(-8)) {
    for (const a of g.assistants) {
      if (Array.isArray(a.content)) {
        for (const b of a.content) {
          if (b.type === 'tool_use') {
            extractFilesFromInput((b as any).input, filesSet);
          }
        }
      }
    }
  }

  const recentGroups = groups.slice(-maxTurns);
  const turnLines: string[] = [];
  let lastQuery = '';

  for (const g of recentGroups) {
    if (g.user) {
      if (typeof g.user.content === 'string') {
        const cleaned = cleanText(g.user.content);
        if (cleaned) {
          lastQuery = cleaned;
          turnLines.push(`- User: ${cleaned.slice(0, 350)}`);
        }
      } else if (Array.isArray(g.user.content)) {
        const parts: string[] = [];
        for (const b of g.user.content) {
          if (b.type === 'text' && typeof (b as any).text === 'string') {
            const cleaned = cleanText((b as any).text);
            if (cleaned) {
              lastQuery = cleaned;
              parts.push(cleaned.slice(0, 300));
            }
          } else if (b.type === 'tool_result') {
            const rawRes = (b as any).content;
            const resStr =
              typeof rawRes === 'string'
                ? rawRes
                : Array.isArray(rawRes)
                  ? rawRes
                      .filter((x: any) => x?.type === 'text' && typeof x?.text === 'string')
                      .map((x: any) => x.text)
                      .join(' ')
                  : JSON.stringify(rawRes ?? '');
            const compactRes = resStr.replace(/\s+/g, ' ').trim().slice(0, 200);
            parts.push(`[tool_result: ${compactRes}]`);
          }
        }
        if (parts.length > 0) {
          turnLines.push(`- User: ${parts.join(' | ')}`);
        }
      }
    }

    for (const a of g.assistants) {
      if (typeof a.content === 'string') {
        const cleaned = a.content.replace(/\s+/g, ' ').trim();
        if (cleaned) {
          turnLines.push(`- Assistant: ${cleaned.slice(0, 350)}`);
        }
      } else if (Array.isArray(a.content)) {
        const parts: string[] = [];
        for (const b of a.content) {
          if (b.type === 'text' && typeof (b as any).text === 'string') {
            const cleaned = (b as any).text.replace(/\s+/g, ' ').trim();
            if (cleaned) parts.push(cleaned.slice(0, 300));
          } else if (b.type === 'tool_use') {
            const name = String((b as any).name ?? 'tool');
            const input = (b as any).input;
            extractFilesFromInput(input, filesSet);
            const inputSummary = JSON.stringify(input ?? {}).slice(0, 160);
            parts.push(`[called ${name}(${inputSummary})]`);
          }
        }
        if (parts.length > 0) {
          turnLines.push(`- Assistant: ${parts.join(' ')}`);
        }
      }
    }
  }

  const filesTouched = [...filesSet];
  const filesLine =
    filesTouched.length > 0 ? `- Files touched: ${filesTouched.join(', ')}\n` : '';
  const text = `${filesLine}${turnLines.join('\n')}`.trim();

  return {
    text,
    filesTouched,
    lastQuery,
  };
}

export async function buildTransitionHandoff(options: {
  readonly body: MessagesBody;
  readonly previousTier: Tier;
  readonly currentTier: Tier;
  readonly aiMemoryBin?: string | undefined;
  readonly cwd?: string | undefined;
  readonly timeoutMs?: number | undefined;
  readonly maxChars?: number | undefined;
  readonly execFn?: ExecAiMemoryFn | undefined;
}): Promise<string> {
  const maxChars = options.maxChars ?? DEFAULT_MAX_HANDOFF_CHARS;
  const timeoutMs = options.timeoutMs ?? 450;
  const bin = options.aiMemoryBin ?? process.env['CONTEXT_AI_MEMORY_BIN'] ?? 'ai-memory';
  const execFn = options.execFn ?? defaultExecAiMemory;
  const cwd = options.cwd ?? extractWorkingDirFromSystem(options.body['system']) ?? process.cwd();

  const summary = summarizeRecentTranscript(options.body.messages, 4);
  const memConfig = findAiMemoryToml(cwd);

  let memorySection = '';
  if (memConfig) {
    const querySource = [summary.lastQuery, ...summary.filesTouched].join(' ').trim();
    const ftsQuery = cleanFtsQuery(querySource);

    const readPagePromise = execFn(
      bin,
      [
        'read-page',
        '--json',
        '--workspace',
        memConfig.workspace,
        '--project',
        memConfig.project,
        '--path',
        HANDOFF_PAGE_PATH,
      ],
      undefined,
      timeoutMs,
    ).catch(() => '');

    const searchPromise = ftsQuery
      ? execFn(
          bin,
          [
            'search',
            '--json',
            '-n',
            '4',
            '--workspace',
            memConfig.workspace,
            '--project',
            memConfig.project,
            ftsQuery,
          ],
          undefined,
          timeoutMs,
        ).catch(() => '')
      : Promise.resolve('');

    const [rawPage, rawSearch] = await Promise.all([readPagePromise, searchPromise]);
    const memLines: string[] = [];

    if (rawPage) {
      try {
        const parsed = JSON.parse(rawPage);
        const bodyText = typeof parsed?.body === 'string' ? parsed.body.trim() : '';
        if (bodyText) {
          memLines.push(`- Prior Handoff: ${bodyText.replace(/\s+/g, ' ').slice(0, 600)}`);
        }
      } catch {}
    }

    if (rawSearch) {
      try {
        const parsed = JSON.parse(rawSearch);
        const hits = Array.isArray(parsed) ? parsed : (parsed?.hits ?? []);
        for (const h of hits.slice(0, 3)) {
          if (!h || typeof h !== 'object') continue;
          const p = String((h as any).path || '');
          if (p === HANDOFF_PAGE_PATH) continue;
          const title = String((h as any).title || p);
          const snippet = String((h as any).snippet || (h as any).body || '')
            .replace(/\s+/g, ' ')
            .trim()
            .slice(0, 300);
          if (snippet) {
            memLines.push(`- [${title} (${p})]: ${snippet}`);
          }
        }
      } catch {}
    }

    if (memLines.length > 0) {
      memorySection = `\n### Project Memory & Handoff (ai-memory)\n${memLines.join('\n')}`;
    }

    // Non-blocking background write to update working/jev-session-handoff.md in ai-memory
    const bgBody = [
      `# JEV Session Handoff`,
      ``,
      `- Last transition: \`${options.previousTier} -> ${options.currentTier}\` at ${new Date().toISOString()}`,
      ...(summary.filesTouched.length > 0 ? [`- Files touched: ${summary.filesTouched.join(', ')}`] : []),
      ``,
      `## Recent Session Summary`,
      summary.text.slice(0, 2000),
    ].join('\n');

    void execFn(
      bin,
      [
        'write-page',
        '--workspace',
        memConfig.workspace,
        '--project',
        memConfig.project,
        '--tier',
        'working',
        '--kind',
        'fact',
        '--path',
        HANDOFF_PAGE_PATH,
        '--title',
        'JEV Session Handoff',
        '--body',
        '-',
      ],
      bgBody,
      2000,
    ).catch(() => {});
  }

  const header = `Model transition: ${options.previousTier} -> ${options.currentTier}. Continue seamlessly without losing context or repeating completed work.\n### Active Session State (last 4 turns)\n`;
  const inner = `${header}${summary.text}${memorySection}`.slice(0, maxChars - 40);

  return `<session_handoff>\n${inner}\n</session_handoff>`;
}

/**
 * Injects the `<session_handoff>` block into `body.system` without mutating
 * any existing prompt-cached system blocks.
 * - Anthropic (`primary`): receives it as the trailing system block after cached blocks.
 * - Gemini (`gemini`): `renderGeminiPrompt` extracts `<session_handoff>` into its preamble.
 * - OpenAI (`trivial` / `standard`): `toChatCompletion` includes it in the initial `system` message.
 */
export function withTransitionHandoff(body: MessagesBody, handoffBlock: string): MessagesBody {
  const existing = body['system'];
  let nextSystem: unknown[];
  if (typeof existing === 'string') {
    nextSystem = existing.length > 0
      ? [{ type: 'text', text: existing }, { type: 'text', text: handoffBlock }]
      : [{ type: 'text', text: handoffBlock }];
  } else if (Array.isArray(existing)) {
    nextSystem = [...existing, { type: 'text', text: handoffBlock }];
  } else {
    nextSystem = [{ type: 'text', text: handoffBlock }];
  }

  return {
    ...body,
    system: nextSystem,
  };
}
