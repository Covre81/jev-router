import type { EventEmitter } from 'node:events';
import { Agent, createServer, request, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';

/** What the supervisor needs from a forked `cli.js serve` (a ChildProcess fits). */
export interface WorkerProcess extends EventEmitter {
  readonly pid?: number | undefined;
  send(message: unknown): boolean;
  kill(): boolean;
}

export interface SupervisorOptions {
  readonly host: string;
  /** Public port Claude Code talks to (0 = any, for tests). */
  readonly port: number;
  /** Loopback control port for `reload` (0 = any, for tests). */
  readonly controlPort: number;
  /** How long a retired worker may finish open streams before they are cut. */
  readonly drainMs: number;
  readonly forkWorker: () => WorkerProcess;
  /** One delay per restart attempt after a crash; when they run out, the supervisor gives up. */
  readonly restartDelaysMs?: readonly number[];
  /** A worker that neither listens nor dies within this is killed and counts as a failed start. */
  readonly startupTimeoutMs?: number;
  /** A worker that stayed up this long resets the restart budget. */
  readonly stableMs?: number;
  readonly log?: (msg: string, extra?: Record<string, unknown>) => void;
}

export interface Supervisor {
  readonly port: number;
  readonly controlPort: number;
  reload(): Promise<ReloadResult>;
  close(): Promise<void>;
}

export type ReloadResult = { readonly ok: true; readonly pid: number | undefined } | { readonly ok: false; readonly error: string };

interface Worker {
  readonly proc: WorkerProcess;
  readonly port: number;
  /** Own keep-alive pool: destroying it is how a drained worker's leftovers are cut. */
  readonly agent: Agent;
  readonly inflight: Set<ServerResponse>;
  readonly listeningAt: number;
  retired: boolean;
}

const LOOPBACK = new Set(['127.0.0.1', '::1', '::ffff:127.0.0.1']);

export function isLoopback(address: string | undefined): boolean {
  return address !== undefined && LOOPBACK.has(address);
}

function anthropicError(res: ServerResponse, status: number, message: string): void {
  if (res.headersSent) return void res.destroy();
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ type: 'error', error: { type: 'api_error', message: `jev-router: ${message}` } }));
}

function listen(server: Server, port: number, host: string): Promise<number> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(port, host, () => resolve((server.address() as AddressInfo).port));
  });
}

/**
 * Reverse proxy in front of a forked worker, so a rebuild reaches Claude Code
 * without dropping it: `reload` starts a new worker, points new requests at
 * it, and lets the old one finish its streams. Chosen over node:cluster,
 * whose shared listening socket is fragile on Windows. Bodies are piped, never
 * buffered: SSE must reach the client as the worker writes it.
 */
export async function startSupervisor(options: SupervisorOptions): Promise<Supervisor> {
  const {
    drainMs,
    forkWorker,
    restartDelaysMs = [1_000, 2_000, 4_000, 4_000, 4_000],
    startupTimeoutMs = 30_000,
    stableMs = 30_000,
    log = defaultLog,
  } = options;

  let active: Worker | undefined;
  let switching = false;
  let closed = false;
  let restarts = 0;
  let restartTimer: NodeJS.Timeout | undefined;
  const workers = new Set<Worker>();
  const timers = new Set<NodeJS.Timeout>();

  /** Fork and wait for `{type:'listening', port}`; rejects if the worker dies or hangs first. */
  function spawn(): Promise<Worker> {
    const proc = forkWorker();
    return new Promise<Worker>((resolve, reject) => {
      let worker: Worker | undefined;
      const timeout = setTimeout(() => {
        proc.kill();
        reject(new Error('worker did not start in time'));
      }, startupTimeoutMs);
      proc.on('message', (msg: unknown) => {
        const m = msg as { type?: unknown; port?: unknown } | null;
        if (worker || m?.type !== 'listening' || typeof m.port !== 'number') return;
        clearTimeout(timeout);
        worker = {
          proc,
          port: m.port,
          agent: new Agent({ keepAlive: true }),
          inflight: new Set(),
          listeningAt: Date.now(),
          retired: false,
        };
        workers.add(worker);
        resolve(worker);
      });
      // setImmediate: a worker can die before the caller has made it `active`;
      // deferring lets that assignment land, so the crash is never missed.
      proc.on('exit', (code: unknown) =>
        setImmediate(() => {
          clearTimeout(timeout);
          if (!worker) return reject(new Error(`worker exited before listening (code ${String(code)})`));
          workers.delete(worker);
          worker.agent.destroy();
          if (worker === active) onCrash(worker);
        }),
      );
    });
  }

  function onCrash(worker: Worker): void {
    active = undefined;
    if (Date.now() - worker.listeningAt >= stableMs) restarts = 0;
    log('worker exited', { pid: worker.proc.pid });
    scheduleRestart();
  }

  function scheduleRestart(): void {
    if (closed) return;
    const delay = restartDelaysMs[restarts];
    if (delay === undefined) {
      log('worker keeps failing: giving up; run `jev-router reload` once fixed', { attempts: restarts });
      return;
    }
    restarts++;
    restartTimer = setTimeout(() => {
      spawn().then(
        (w) => {
          if (closed) return retire(w, 0);
          active = w;
          log('worker restarted', { pid: w.proc.pid });
        },
        (err: unknown) => {
          log('worker failed to start', { err: String(err) });
          scheduleRestart();
        },
      );
    }, delay);
  }

  /** Stop routing to `worker`; kill it when idle, or cut what is left after `afterMs`. */
  function retire(worker: Worker, afterMs: number): void {
    worker.retired = true;
    worker.proc.send({ type: 'drain' });
    const finish = () => {
      worker.agent.destroy();
      for (const res of worker.inflight) res.destroy();
      worker.proc.kill();
    };
    if (worker.inflight.size === 0) return finish();
    const timer = setTimeout(finish, afterMs);
    timers.add(timer);
    worker.proc.once('exit', () => clearTimeout(timer));
  }

  async function reload(): Promise<ReloadResult> {
    if (switching) return { ok: false, error: 'reload already in progress' };
    switching = true;
    try {
      const next = await spawn();
      const old = active;
      active = next;
      restarts = 0;
      clearTimeout(restartTimer);
      if (old) retire(old, drainMs);
      log('reloaded', { pid: next.proc.pid, previous: old?.proc.pid });
      return { ok: true, pid: next.proc.pid };
    } catch (err) {
      return { ok: false, error: err instanceof Error ? err.message : String(err) };
    } finally {
      switching = false;
    }
  }

  function forward(req: IncomingMessage, res: ServerResponse): void {
    const worker = active;
    if (!worker) {
      if (req.url === '/healthz') {
        res.writeHead(503, { 'content-type': 'application/json' }).end(JSON.stringify({ ok: false, error: 'worker down' }));
      } else {
        anthropicError(res, 503, 'worker down');
      }
      req.resume();
      return;
    }
    worker.inflight.add(res);
    const done = () => {
      worker.inflight.delete(res);
      if (worker.retired && worker.inflight.size === 0) {
        worker.agent.destroy();
        worker.proc.kill();
      }
    };
    res.once('close', done);

    const upstream = request(
      { host: '127.0.0.1', port: worker.port, method: req.method, path: req.url, headers: req.headers, agent: worker.agent },
      (up) => {
        res.writeHead(up.statusCode ?? 502, up.headers);
        res.flushHeaders();
        up.pipe(res);
        up.once('error', () => res.destroy());
        up.once('aborted', () => res.destroy());
      },
    );
    upstream.once('error', () => anthropicError(res, 502, 'worker unreachable'));
    // Client gone (Esc in Claude Code): stop the worker's work too.
    res.once('close', () => {
      if (!res.writableFinished) upstream.destroy();
    });
    req.pipe(upstream);
  }

  const proxy = createServer(forward);
  const control = createServer((req, res) => {
    req.resume();
    const reply = (status: number, body: unknown) =>
      res.writeHead(status, { 'content-type': 'application/json' }).end(JSON.stringify(body));
    // Bound to loopback already; the peer check is the second lock.
    if (!isLoopback(req.socket.remoteAddress)) return reply(403, { ok: false, error: 'loopback only' });
    if (req.method !== 'POST' || req.url !== '/reload') return reply(404, { ok: false, error: 'not found' });
    void reload().then((r) => reply(r.ok ? 200 : r.error.includes('in progress') ? 409 : 500, r));
  });

  // First worker before the public port opens; a failed first start goes through the restart budget.
  await spawn().then(
    (w) => {
      active = w;
    },
    (err: unknown) => {
      log('worker failed to start', { err: String(err) });
      scheduleRestart();
    },
  );
  const port = await listen(proxy, options.port, options.host);
  const controlPort = await listen(control, options.controlPort, '127.0.0.1');
  log('supervisor ready', { port, controlPort, worker: active?.proc.pid });

  return {
    port,
    controlPort,
    reload,
    async close() {
      closed = true;
      clearTimeout(restartTimer);
      for (const t of timers) clearTimeout(t);
      for (const w of workers) {
        w.agent.destroy();
        w.proc.kill();
      }
      active = undefined;
      for (const server of [proxy, control]) {
        server.closeAllConnections();
        await new Promise<void>((r) => server.close(() => r()));
      }
    },
  };
}

/** One pino-shaped JSON line on stdout (router.log), like the worker's Fastify logger. */
export function supervisorLog(msg: string, extra: Record<string, unknown> = {}, level = 30): void {
  console.log(JSON.stringify({ level, time: Date.now(), pid: process.pid, name: 'supervisor', msg, ...extra }));
}

function defaultLog(msg: string, extra: Record<string, unknown> = {}): void {
  supervisorLog(msg, extra);
}
