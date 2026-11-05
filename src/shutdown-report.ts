import { execFile } from 'node:child_process';

/**
 * Forensics for router stops nobody explains (console Ctrl+C with no known sender since
 * 2026-10-28). Windows does not tell a process who raised a console control event, so the best
 * evidence is the parent and the processes started just before; a sender that already exited is
 * not in the list (Windows' process-creation audit, event 4688, is the tool for that).
 */

export type StopSignal = 'SIGINT' | 'SIGTERM' | 'SIGBREAK';
export const STOP_SIGNALS: readonly StopSignal[] = ['SIGINT', 'SIGTERM', 'SIGBREAK'];

// ---------- redaction ----------

// Names whose value is a credential. Matched only after a non-letter (so ANTHROPIC_API_KEY and
// ?key= match, monkey/keyboard do not) and only when a value follows (`=`, `:`, or a --flag).
const NAME = String.raw`(?<![A-Za-z])(?:api[_-]?key|apikey|access[_-]?token|refresh[_-]?token|client[_-]?secret|private[_-]?key|token|secret|password|passwd|pwd|credentials?|auth|key|sig|signature)(?![A-Za-z-])`;
const R = '<REDACTED>';

/** Applied in order; each keeps the label and masks only the value. */
const RULES: readonly [RegExp, string | ((...m: string[]) => string)][] = [
  // scheme://user:password@host
  [/(\b[a-z][a-z0-9+.-]*:\/\/[^\s:/@]+:)[^\s@/]+@/gi, `$1${R}@`],
  // Authorization: Basic|Bearer|Token|Digest <value>  (header or flag, quoted or not)
  [/(\bauthorization["']?\s*[:=]\s*["']?)(?:(basic|bearer|token|digest)\s+)?[^\s"',;]+/gi, (_m, head, scheme) => `${head}${scheme ? `${scheme} ` : ''}${R}`],
  // name = "value with spaces" | 'value' | JSON "name":"value"
  [new RegExp(String.raw`(["']?)(${NAME})\1(\s*[=:]\s*)(["'])(?:(?!\4).)*\4`, 'gi'), `$1$2$1$3$4${R}$4`],
  // --name "value with spaces"
  [new RegExp(String.raw`(--?${NAME})(\s+)(["'])(?:(?!\3).)*\3`, 'gi'), `$1$2$3${R}$3`],
  // name=value | name: value (URL query stops at & and quotes)
  [new RegExp(String.raw`(${NAME})(\s*[=:]\s*)[^\s&"',;]+`, 'gi'), `$1$2${R}`],
  // --name value
  [new RegExp(String.raw`(--?${NAME})(\s+)(?![-"'])[^\s"']+`, 'gi'), `$1$2${R}`],
  // well-known token shapes, wherever they appear: JWT, sk-/pk-, GitHub, Slack, Google, AWS
  [/\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+/g, R],
  [/\b(?:sk|pk|rk)-[A-Za-z0-9_-]{8,}|\bgh[pousr]_[A-Za-z0-9]{20,}|\bgithub_pat_[A-Za-z0-9_]{20,}|\bxox[abprs]-[A-Za-z0-9-]{10,}|\bAIza[0-9A-Za-z_-]{20,}|\bAKIA[0-9A-Z]{16}\b/g, R],
  // any other long opaque string (hashes, base64): no '/', '\' or '.', so paths survive
  [/(?<![A-Za-z0-9_+=-])[A-Za-z0-9_+-]{32,}={0,2}(?![A-Za-z0-9_+=-])/g, R],
];

/** Command lines go to router.log: mask anything that looks like a credential, then truncate. */
export function redactCommandLine(cmd: string): string {
  let out = cmd;
  for (const [re, to] of RULES) out = out.replace(re, to as string);
  return out.slice(0, 240);
}

// ---------- inspection ----------

export interface ProcessInfo {
  readonly ProcessId?: number;
  readonly ParentProcessId?: number;
  readonly Name?: string;
  readonly CommandLine?: string | null;
  readonly Created?: string;
}

export type StopContext =
  | { readonly parent: ProcessInfo | null; readonly recent: readonly ProcessInfo[] }
  | { readonly inspectError: string };

/** The parent and every process started in the last 2 minutes, as JSON (Windows only). */
function inspectWindowsProcesses(): Promise<string> {
  const script =
    `$p = Get-CimInstance Win32_Process -Filter 'ProcessId=${process.ppid}' | Select-Object ProcessId,ParentProcessId,Name,CommandLine; ` +
    `$since = (Get-Date).AddSeconds(-120); ` +
    `$r = @(Get-CimInstance Win32_Process | Where-Object { $_.CreationDate -gt $since } | ` +
    `Select-Object ProcessId,ParentProcessId,Name,CommandLine,@{n='Created';e={$_.CreationDate.ToString('o')}}); ` +
    `@{ parent = $p; recent = $r } | ConvertTo-Json -Compress -Depth 3`;
  return new Promise((resolve, reject) => {
    execFile('powershell.exe', ['-NoProfile', '-NonInteractive', '-Command', script], { windowsHide: true, timeout: 4_000 }, (err, stdout) =>
      err ? reject(err) : resolve(stdout),
    );
  });
}

export interface InspectOptions {
  readonly inspect?: () => Promise<string>;
  readonly platform?: NodeJS.Platform;
  readonly timeoutMs?: number;
}

/** Never throws and never waits past `timeoutMs`; undefined outside Windows. */
export async function inspectStopContext({
  inspect = inspectWindowsProcesses,
  platform = process.platform,
  timeoutMs = 5_000,
}: InspectOptions = {}): Promise<StopContext | undefined> {
  if (platform !== 'win32') return undefined;
  let timer: NodeJS.Timeout | undefined;
  try {
    const raw = await Promise.race([
      inspect(),
      new Promise<never>((_, reject) => {
        timer = setTimeout(() => reject(new Error('inspect timeout')), timeoutMs);
      }),
    ]);
    const info = JSON.parse(raw) as { parent?: ProcessInfo | null; recent?: ProcessInfo[] | ProcessInfo | null };
    const redactProcess = (p: ProcessInfo): ProcessInfo => ({ ...p, CommandLine: p.CommandLine ? redactCommandLine(p.CommandLine) : (p.CommandLine ?? null) });
    const recent = info.recent == null ? [] : Array.isArray(info.recent) ? info.recent : [info.recent];
    return { parent: info.parent ? redactProcess(info.parent) : null, recent: recent.map(redactProcess) };
  } catch (err) {
    return { inspectError: err instanceof Error ? err.message : String(err) };
  } finally {
    clearTimeout(timer);
  }
}

// ---------- handler ----------

export interface StopSignalOptions {
  readonly log: (msg: string, entry: Record<string, unknown>) => void;
  readonly stop: () => Promise<void>;
  readonly exit?: (code: number) => void;
  readonly on?: (signal: StopSignal, handler: () => void) => void;
  readonly context?: InspectOptions;
}

/**
 * On the first stop signal: log it at once, start stopping at once, and inspect in parallel.
 * Exit once stopping is done and the inspection settled (bounded by its timeout). A second
 * signal forces the exit right away; the first line is already in the log.
 */
export function onStopSignal({
  log,
  stop,
  exit = (code) => process.exit(code),
  on = (signal, handler) => void process.on(signal, handler),
  context,
}: StopSignalOptions): void {
  let stopping = false;
  for (const signal of STOP_SIGNALS) {
    on(signal, () => {
      if (stopping) {
        log('stop signal repeated: forcing exit', { signal });
        exit(130);
        return;
      }
      stopping = true;
      log('stop signal received', { signal, at: new Date().toISOString(), ppid: process.ppid });
      const inspected = inspectStopContext(context).then((ctx) => {
        if (ctx) log('stop signal context', { signal, ...ctx });
      });
      const stopped = stop().catch((err: unknown) => log('stop failed', { signal, err: err instanceof Error ? err.message : String(err) }));
      void Promise.allSettled([stopped, inspected]).then(() => exit(0));
    });
  }
}
