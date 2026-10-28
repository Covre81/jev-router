import { execFile } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { promisify } from 'node:util';

const execFileAsync = promisify(execFile);

export interface AiMemoryConfig {
  readonly workspace: string;
  readonly project: string;
}

export interface MemoryHit {
  readonly path: string;
  readonly title: string;
  readonly snippet: string;
}

/**
 * Searches up from startDir for the nearest `.ai-memory.toml` and returns
 * the parsed workspace and project.
 */
export function findAiMemoryToml(startDir: string): AiMemoryConfig | undefined {
  let current = resolve(startDir);
  while (true) {
    const file = join(current, '.ai-memory.toml');
    if (existsSync(file)) {
      try {
        const content = readFileSync(file, 'utf8');
        // Simple TOML-like parsing using regex
        const workspaceMatch = content.match(/workspace\s*=\s*["']([^"']+)["']/);
        const projectMatch = content.match(/project\s*=\s*["']([^"']+)["']/);
        if (workspaceMatch && projectMatch) {
          return {
            workspace: workspaceMatch[1]!,
            project: projectMatch[1]!,
          };
        }
      } catch {
        // Ignore read/parse errors
      }
    }
    const parent = dirname(current);
    if (parent === current) break;
    current = parent;
  }
  return undefined;
}

/**
 * Strips FTS5 syntax, keeps terms with length between 3 and 12 characters,
 * and joins them using ' OR '.
 */
export function cleanFtsQuery(prompt: string): string {
  // Strip FTS5 syntax characters, keep unicode alphanumeric
  const cleaned = prompt.replace(/[^\p{L}\p{N}]+/gu, ' ');
  const words = cleaned.split(/\s+/).filter(Boolean);
  
  // Filter words: length >= 3 and length <= 12
  const filtered = words
    .map((w) => w.toLowerCase())
    .filter((w) => w.length >= 3 && w.length <= 12);
    
  // Deduplicate and join with ' OR '
  const unique = Array.from(new Set(filtered));
  return unique.join(' OR ');
}

/**
 * Executes `ai-memory search` to find matching documents.
 */
export async function searchMemory(options: {
  readonly bin: string;
  readonly cwd: string;
  readonly prompt: string;
  readonly timeoutMs: number;
  readonly signal?: AbortSignal;
}): Promise<readonly MemoryHit[]> {
  const config = findAiMemoryToml(options.cwd);
  if (!config) {
    return [];
  }

  const query = cleanFtsQuery(options.prompt);
  if (!query) {
    return [];
  }

  try {
    const { stdout } = await execFileAsync(
      options.bin,
      [
        'search',
        '--json',
        '-n',
        '15',
        '--workspace',
        config.workspace,
        '--project',
        config.project,
        query,
      ],
      {
        timeout: options.timeoutMs,
        signal: options.signal,
        // No console window flash on Windows (hook runs without a console).
        windowsHide: true,
      }
    );

    const parsed: unknown = JSON.parse(stdout);
    const hits = Array.isArray(parsed) ? parsed : ((parsed as any)?.hits ?? []);
    
    return (hits as any[]).map((h: any) => ({
      path: String(h.path || ''),
      title: String(h.title || h.path || ''),
      snippet: String(h.snippet || h.body || h.content || ''),
    }));
  } catch (err) {
    // If the process fails or command is not found, we let the caller know
    throw err;
  }
}
