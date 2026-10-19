import assert from 'node:assert';
import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { join, parse, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { DatabaseSync } from 'node:sqlite';
import { describe, before, after, afterEach, it } from 'node:test';
import { tmpdir } from 'node:os';
import { openTelemetryDb } from '../src/telemetry/db.js';

import { getMachineOrigin } from '../src/context/origin.js';
import { cleanFtsQuery, searchMemory } from '../src/context/memory-source.js';
import { getGraphSources } from '../src/context/graph-source.js';
import type { GraphData, GraphCommunity } from '../src/context/graph-source.js';
import { JevClient, JevHttpError } from '../src/context/jev-client.js';
import { rankCommunities } from '../src/context/relevance.js';
import { scoreCandidates } from '../src/context/select.js';
import type { Candidate } from '../src/context/select.js';
import { recordContextRun } from '../src/context/log.js';
import { runPipeline, type PipelineConfig } from '../src/context/pipeline.js';
import { isProjectDir } from '../src/context/project-dir.js';
import { MIGRATIONS } from '../src/telemetry/schema.js';

/** Index of the migration that created context_runs (the layout 62d6389 shipped). */
const CONTEXT_MIGRATION = MIGRATIONS.findIndex((m) => m.includes('CREATE TABLE context_runs'));

async function readBody(req: IncomingMessage): Promise<any> {
  let raw = '';
  for await (const chunk of req) raw += chunk;
  return JSON.parse(raw);
}

/**
 * Answers like the real POST /v1/systemone (checked against it on 2026-10-07):
 * a Choice needs `criteria` as a {key: description} dictionary, or the API
 * says 422 and echoes the question; probabilities come back keyed by those keys.
 */
async function jevLike(req: IncomingMessage, res: ServerResponse, noul = 0.9): Promise<void> {
  answerLikeJev(await readBody(req), res, noul);
}

function answerLikeJev(body: any, res: ServerResponse, noul: number): void {
  const answers: Record<string, unknown> = {};
  for (const [id, q] of Object.entries<any>(body.questions ?? {})) {
    if (q.type === 'choice') {
      if (!q.criteria || typeof q.criteria !== 'object' || Array.isArray(q.criteria)) {
        res.writeHead(422, { 'content-type': 'application/json' });
        res.end(
          JSON.stringify({
            detail: [{ type: 'dict_type', loc: ['body', 'questions', id, 'choice', 'criteria'], msg: 'Input should be a valid dictionary', input: q }],
          }),
        );
        return;
      }
      const keys = Object.keys(q.criteria);
      const probabilities = Object.fromEntries(keys.map((k, i) => [k, Math.max(0.01, 0.9 - i * 0.1)]));
      answers[id] = { type: 'choice', choice: keys[0], confidence: 0.9, probabilities };
    } else {
      answers[id] = { type: 'noul', noul };
    }
  }
  res.writeHead(200, { 'content-type': 'application/json' });
  res.end(JSON.stringify({ model: 'jev-1.13.0', answers, usage: { input_tokens: 100, output_tokens: 10 } }));
}

async function listen(handler: (req: IncomingMessage, res: ServerResponse) => void): Promise<{ server: Server; url: string }> {
  const server = createServer(handler);
  await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
  return { server, url: `http://127.0.0.1:${(server.address() as AddressInfo).port}/v1/systemone` };
}

/** A tiny graphify output: community 1 is the routing code, community 2 the telemetry. */
const GRAPH = {
  nodes: [
    { id: 'A', label: 'selectRoute', community: 1 },
    { id: 'B', label: 'riskVeto', community: 1 },
    { id: 'C', label: 'SqliteTelemetry', community: 2 },
    { id: 'D', label: 'computeStats', community: 2 },
  ],
  links: [
    { source: 'A', target: 'B' },
    { source: 'C', target: 'D' },
    { source: 'A', target: 'C' },
  ],
};

function writeGraph(root: string, project: string, content: string = JSON.stringify(GRAPH)): void {
  const dir = join(root, project, 'graphify-out');
  mkdirSync(dir, { recursive: true });
  writeFileSync(join(dir, 'graph.json'), content, 'utf8');
}

const baseConfig = (apiUrl: string, over: Partial<PipelineConfig> = {}): PipelineConfig => ({
  mode: 'inject',
  lCurrent: 0.8,
  lOther: 0.9,
  maxItems: 5,
  maxChars: 1500,
  timeoutMs: 2000,
  aiMemoryBin: 'ai-memory-not-installed',
  apiKey: 'mock_key',
  apiUrl,
  model: 'jev-latest',
  ...over,
});

describe('context-origin', () => {
  it('correctly classifies machine origin tags', () => {
    assert.strictEqual(getMachineOrigin('<task-notification> hello'), 'machine:task-notification');
    assert.strictEqual(getMachineOrigin('  <wake> system reboot '), 'machine:wake');
    assert.strictEqual(getMachineOrigin('\n<relay> test-run'), 'machine:relay');
    assert.strictEqual(getMachineOrigin('human prompt with <task-notification> in middle'), undefined);
    assert.strictEqual(getMachineOrigin('plain human text'), undefined);
  });
});

describe('context-memory', () => {
  it('cleans FtsQuery removing syntax characters and filtering word sizes', () => {
    const rawPrompt = 'Fix the parsing! Use standard AND, OR, NOT operations with longwordtoolongtobevalid.';
    const terms = cleanFtsQuery(rawPrompt).split(' OR ');
    for (const t of ['fix', 'the', 'parsing', 'use', 'standard', 'operations']) assert.ok(terms.includes(t), t);
    assert.ok(!terms.includes('longwordtoolongtobevalid'));
    assert.ok(terms.every((t) => t.length >= 3 && t.length <= 12));
  });

  it('returns empty array when .ai-memory.toml is missing', async () => {
    const tempDir = mkdtempSync(join(tmpdir(), 'jev-ctx-notoml-'));
    try {
      assert.deepStrictEqual(await searchMemory({ bin: 'node', cwd: tempDir, prompt: 'test prompt', timeoutMs: 1000 }), []);
    } finally {
      rmSync(tempDir, { recursive: true, force: true });
    }
  });
});

describe('context-graph', () => {
  let root: string;
  before(() => {
    root = mkdtempSync(join(tmpdir(), 'jev-ctx-graph-'));
  });
  after(() => rmSync(root, { recursive: true, force: true }));

  it('computes degrees and describes each community by its top nodes', () => {
    const dir = mkdtempSync(join(root, 'one-'));
    writeGraph(dir, 'proj1');
    const [graph] = getGraphSources(dir, dir);
    assert.ok(graph);
    assert.strictEqual(graph.project, 'proj1');
    const routing = graph.communities.find((c) => c.id === '1')!;
    assert.strictEqual(routing.topNodes[0]!.label, 'selectRoute');
    assert.strictEqual(routing.topNodes[0]!.degree, 2);
    assert.strictEqual(routing.description, 'selectRoute, riskVeto');
  });

  it('loads byte-identical graphs once (sha256), keeps different ones', () => {
    const dir = mkdtempSync(join(root, 'dup-'));
    writeGraph(dir, 'V&C Home Services');
    writeGraph(dir, 'V-C-Home-Services');
    assert.strictEqual(getGraphSources(dir, dir).length, 1, 'the two copies of V&C become one');
    writeGraph(dir, 'other', JSON.stringify({ ...GRAPH, nodes: GRAPH.nodes.slice(0, 2) }));
    assert.strictEqual(getGraphSources(dir, dir).length, 2);
  });

  it('skips a broken graph.json without losing the good ones', () => {
    const dir = mkdtempSync(join(root, 'broken-'));
    writeGraph(dir, 'good');
    writeGraph(dir, 'broken', 'broken { json');
    writeGraph(dir, 'not-a-graph', '{"nodes":"nope"}');
    const graphs = getGraphSources(dir, dir);
    assert.deepStrictEqual(
      graphs.map((g) => g.project),
      ['good'],
    );
  });
});

describe('context-relevance', () => {
  let fake: { server: Server; url: string };
  let client: JevClient;
  let requests: any[] = [];
  let malformed = false;

  before(async () => {
    fake = await listen(async (req, res) => {
      const body = await readBody(req);
      if (malformed) {
        res.writeHead(200, { 'content-type': 'application/json' });
        res.end(JSON.stringify({ answers: { relevant_community: { probabilities: 'high' }, c0: { noul: 'yes' } } }));
        return;
      }
      requests.push(body);
      answerLikeJev(body, res, 0.85);
    });
    client = new JevClient({ apiUrl: fake.url, apiKey: 'mock_key' });
  });
  after(() => fake.server.close());

  const graphOf = (n: number, project = 'proj'): GraphData => ({
    path: `${project}/graph.json`,
    sha256: project,
    project,
    isCurrent: true,
    communities: Array.from({ length: n }, (_, i): GraphCommunity => ({ id: String(i), topNodes: [], description: `node${i}` })),
  });

  it('asks a Choice with criteria as a {key: description} dictionary, as the real API requires', async () => {
    requests = [];
    const ranked = await rankCommunities({ client, model: 'jev-model', prompt: 'hello', graphs: [graphOf(3)], signal: AbortSignal.timeout(2000) });
    const q = requests[0].questions.relevant_community;
    assert.strictEqual(q.type, 'choice');
    assert.ok(q.criteria && !Array.isArray(q.criteria), 'criteria is a dictionary');
    assert.deepStrictEqual(Object.values(q.criteria), ['node0', 'node1', 'node2']);
    assert.deepStrictEqual(
      ranked.map((r) => [r.community.id, r.p]),
      [
        ['0', 0.9],
        ['1', 0.8],
      ],
      'probabilities map back to communities through the criteria keys',
    );
  });

  it('chunks 300 communities into 2 requests, and asks every graph in parallel', async () => {
    requests = [];
    const ranked = await rankCommunities({
      client,
      model: 'jev-model',
      prompt: 'hello',
      graphs: [graphOf(300, 'a'), graphOf(3, 'b')],
      signal: AbortSignal.timeout(2000),
    });
    assert.strictEqual(requests.length, 3);
    assert.ok(requests.every((r) => Object.keys(r.questions.relevant_community.criteria).length <= 240));
    assert.strictEqual(ranked.length, 4, 'two best communities per graph');
  });

  it('batches 175 candidates into 6 requests', async () => {
    requests = [];
    const candidates: Candidate[] = Array.from({ length: 175 }, (_, i) => ({
      id: `c${i}`,
      type: 'memory',
      path: `doc${i}.md`,
      content: `snippet${i}`,
      isCurrent: true,
      project: 'proj',
    }));
    const scored = await scoreCandidates({ client, model: 'jev-model', prompt: 'hello', candidates, signal: AbortSignal.timeout(2000) });
    assert.strictEqual(requests.length, 6);
    assert.strictEqual(scored.length, 175);
    assert.strictEqual(scored[0]!.p, 0.85);
  });

  it('rejects a malformed answer instead of guessing', async () => {
    malformed = true;
    try {
      await assert.rejects(
        rankCommunities({ client, model: 'm', prompt: 'hello', graphs: [graphOf(3)], signal: AbortSignal.timeout(2000) }),
        /schema/,
      );
      const one: Candidate[] = [{ id: 'c0', type: 'memory', path: 'a.md', content: 'x', isCurrent: true, project: 'p' }];
      await assert.rejects(scoreCandidates({ client, model: 'm', prompt: 'hello', candidates: one, signal: AbortSignal.timeout(2000) }), /schema/);
    } finally {
      malformed = false;
    }
  });

  it('reports an HTTP error with its status and the API reason, not the echoed input', async () => {
    const reject422 = await listen(async (req, res) => {
      const body = await readBody(req);
      res.writeHead(422, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ detail: [{ loc: ['body', 'questions', 'x', 'choice', 'criteria'], msg: 'Field required', input: body }] }));
    });
    try {
      const err = await new JevClient({ apiUrl: reject422.url, apiKey: 'secret-key-value' })
        .postSystemOne({ questions: {}, state: 'the user prompt' }, AbortSignal.timeout(2000))
        .then(
          () => undefined,
          (e: unknown) => e,
        );
      assert.ok(err instanceof JevHttpError);
      assert.strictEqual(err.statusCode, 422);
      assert.strictEqual(err.message, 'JEV HTTP 422: questions.x.choice.criteria: Field required');
      assert.ok(!err.message.includes('secret-key-value') && !err.message.includes('the user prompt'));
    } finally {
      reject422.server.close();
    }
  });
});

describe('context-pipeline', () => {
  let ok: { server: Server; url: string };
  let rejecting: { server: Server; url: string };
  let dir: string;
  let dbFile: string;

  before(async () => {
    ok = await listen((req, res) => void jevLike(req, res, 0.95));
    rejecting = await listen(async (req, res) => {
      await readBody(req);
      res.writeHead(422, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ detail: [{ loc: ['body', 'questions', 'relevant_community', 'choice', 'criteria'], msg: 'Field required' }] }));
    });
    dir = mkdtempSync(join(tmpdir(), 'jev-ctx-pipeline-'));
    writeGraph(dir, 'proj');
    dbFile = join(dir, 'telemetry.db');
    process.env['TELEMETRY_DB_PATH'] = dbFile;
    openTelemetryDb(dbFile).close();
  });

  after(() => {
    ok.server.close();
    rejecting.server.close();
    delete process.env['TELEMETRY_DB_PATH'];
    rmSync(dir, { recursive: true, force: true });
  });

  const lastRun = (session: string): any => {
    const db = new DatabaseSync(dbFile);
    try {
      return db.prepare('SELECT * FROM context_runs WHERE session_id = ? ORDER BY id DESC').get(session);
    } finally {
      db.close();
    }
  };
  const input = (session: string, prompt = 'how does selectRoute apply the risk veto?') => ({ prompt, session_id: session, cwd: join(dir, 'proj') });

  it('records machine prompts as skipped_origin without calling JEV', async () => {
    const res = await runPipeline(input('sess-machine', '<task-notification> done'), baseConfig('http://127.0.0.1:1/v1/systemone'));
    assert.strictEqual(res.outcome, 'skipped_origin');
    assert.strictEqual(res.stdout, '');
    assert.strictEqual(lastRun('sess-machine').outcome, 'skipped_origin');
  });

  it('injects the graph communities JEV finds relevant, and keeps only a hash of the prompt', async () => {
    const res = await runPipeline(input('sess-ok'), baseConfig(ok.url, { graphRoot: dir }));
    assert.strictEqual(res.outcome, 'ok');
    const out = JSON.parse(res.stdout);
    assert.strictEqual(out.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
    assert.match(out.hookSpecificOutput.additionalContext, /\[grafo\] proj › /);
    const run = lastRun('sess-ok');
    assert.match(run.prompt_hash, /^[0-9a-f]{64}$/);
    assert.strictEqual(run.prompt_chars, 'how does selectRoute apply the risk veto?'.length);
    assert.strictEqual('prompt' in run, false, 'the prompt text is never stored');
    assert.strictEqual(run.error, null);
    assert.ok(run.latency_ms >= 0);
  });

  it('says which source fell when one of them does (partial)', async () => {
    writeFileSync(join(dir, 'proj', '.ai-memory.toml'), 'workspace = "default"\nproject = "proj"\n');
    try {
      const res = await runPipeline(input('sess-partial'), baseConfig(ok.url, { graphRoot: dir, aiMemoryBin: join(dir, 'missing-ai-memory') }));
      assert.strictEqual(res.outcome, 'partial');
      assert.match(res.error ?? '', /^memory: /);
      assert.ok(res.stdout.includes('additionalContext'), 'the graph still answers');
      assert.match(lastRun('sess-partial').error, /^memory: /);
    } finally {
      rmSync(join(dir, 'proj', '.ai-memory.toml'));
    }
  });

  it('records why JEV failed (status and reason) instead of a bare jev_error', async () => {
    const res = await runPipeline(input('sess-jev'), baseConfig(rejecting.url, { graphRoot: dir }));
    assert.strictEqual(res.outcome, 'jev_error');
    assert.strictEqual(res.stdout, '');
    assert.strictEqual(res.error, 'jev: JEV HTTP 422: questions.relevant_community.choice.criteria: Field required');
    assert.strictEqual(lastRun('sess-jev').error, res.error);
  });

  it('handles pipeline timeouts', async () => {
    const hung = await listen(() => {});
    try {
      const res = await runPipeline(input('sess-timeout'), baseConfig(hung.url, { graphRoot: dir, timeoutMs: 50 }));
      assert.strictEqual(res.outcome, 'timeout');
      assert.match(res.error ?? '', /timeout after 50 ms/);
    } finally {
      hung.server.closeAllConnections();
      hung.server.close();
    }
  });

  it('truncates oversized prompts sent to JEV at 4000 chars while recording the full prompt length', async () => {
    const seenBodies: any[] = [];
    const inspectServer = await listen(async (req, res) => {
      const body = await readBody(req);
      seenBodies.push(body);
      answerLikeJev(body, res, 0.95);
    });
    try {
      const hugePrompt = `how does selectRoute work? ${'x'.repeat(12_000)}`;
      const res = await runPipeline(input('sess-huge', hugePrompt), baseConfig(inspectServer.url, { graphRoot: dir }));
      assert.strictEqual(res.outcome, 'ok');
      assert.ok(seenBodies.length >= 2);
      for (const b of seenBodies) {
        assert.ok(typeof b.state === 'string' && b.state.length <= 4_020, `state too large: ${b.state.length}`);
        for (const q of Object.values<any>(b.questions ?? {})) {
          if (typeof q.instructions === 'string') {
            assert.ok(q.instructions.length < 1_000, `question instructions repeated prompt: ${q.instructions.length}`);
          }
        }
      }
      const run = lastRun('sess-huge');
      assert.strictEqual(run.prompt_chars, hugePrompt.length);
    } finally {
      inspectServer.server.close();
    }
  });

  it('supports Antigravity / agy PreInvocation payloads via transcriptPath and skips tool-continuation steps', async () => {
    const transcriptFile = join(dir, 'transcript.jsonl');
    writeFileSync(
      transcriptFile,
      [
        JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: '<USER_REQUEST>\nhow does selectRoute apply the risk veto?\n</USER_REQUEST>' }),
        JSON.stringify({ step_index: 1, type: 'EPHEMERAL_MESSAGE', content: 'some reminder' }),
      ].join('\n') + '\n',
      'utf8',
    );

    const agyRes = await runPipeline(
      {
        conversationId: 'agy-conv-1',
        workspacePaths: [join(dir, 'proj')],
        transcriptPath: transcriptFile,
      },
      baseConfig(ok.url, { graphRoot: dir }),
    );
    assert.strictEqual(agyRes.outcome, 'ok');
    const parsed = JSON.parse(agyRes.stdout);
    assert.ok(Array.isArray(parsed.injectSteps));
    assert.match(parsed.injectSteps[0].ephemeralMessage, /\[grafo\] proj › /);

    // Tool continuation step (last non-ephemeral step is PLANNER_RESPONSE): must skip without calling JEV
    writeFileSync(
      transcriptFile,
      [
        JSON.stringify({ step_index: 0, type: 'USER_INPUT', content: '<USER_REQUEST>\nhello\n</USER_REQUEST>' }),
        JSON.stringify({ step_index: 1, type: 'PLANNER_RESPONSE', content: 'calling tool' }),
      ].join('\n') + '\n',
      'utf8',
    );
    const contRes = await runPipeline(
      {
        conversationId: 'agy-conv-1',
        workspacePaths: [join(dir, 'proj')],
        transcriptPath: transcriptFile,
      },
      baseConfig('http://127.0.0.1:1/v1/systemone', { graphRoot: dir }),
    );
    assert.strictEqual(contRes.outcome, 'skipped_origin');
    assert.strictEqual(contRes.stdout, '{}');
  });
});

describe('context-log', () => {
  const run = {
    sessionId: 'sess-log-test',
    promptHash: 'a'.repeat(64),
    promptChars: 11,
    mode: 'inject' as const,
    outcome: 'ok' as const,
    pipelineVersion: 'hash123',
    injected: true,
    injectedChars: 50,
    latencyMs: 12,
    error: null,
  };

  it('writes a run and its candidates in one transaction, and rolls both back on a bad score', () => {
    const db = new DatabaseSync(':memory:');
    for (const sql of MIGRATIONS) db.exec(sql);
    const runId = recordContextRun(db, run, [
      { type: 'memory', path: 'doc1.md', content: 'good content', p: 0.85, injected: true },
      { type: 'memory', path: 'doc2.md', content: 'good content 2', p: 0.5, injected: false },
    ]);
    assert.ok(runId > 0);
    assert.strictEqual(db.prepare('SELECT count(*) AS n FROM context_candidates WHERE run_id = ?').get(runId)!['n'], 2);

    assert.throws(
      () => recordContextRun(db, { ...run, sessionId: 'sess-rolled-back' }, [{ type: 'memory', path: 'x.md', content: 'bad', p: 1.5, injected: true }]),
      /constraint/i,
    );
    assert.strictEqual(db.prepare('SELECT * FROM context_runs WHERE session_id = ?').get('sess-rolled-back'), undefined);
    db.close();
  });

  it('migrates a new database to the current layout', () => {
    const dir = mkdtempSync(join(tmpdir(), 'jev-ctx-mig-new-'));
    try {
      const db = openTelemetryDb(join(dir, 't.db'));
      const cols = (db.prepare('PRAGMA table_info(context_runs)').all() as { name: string }[]).map((c) => c.name);
      for (const c of ['prompt_hash', 'prompt_chars', 'error', 'latency_ms']) assert.ok(cols.includes(c), c);
      assert.ok(!cols.includes('prompt'));
      assert.ok(recordContextRun(db, run, []) > 0);
      db.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it('migrates an older database: before the context tables, and with the plain-text prompt column', () => {
    const dir = mkdtempSync(join(tmpdir(), 'jev-ctx-mig-old-'));
    try {
      // Before phase 5: router_logs only.
      const beforeFile = join(dir, 'before.db');
      const before = new DatabaseSync(beforeFile);
      for (const sql of MIGRATIONS.slice(0, CONTEXT_MIGRATION)) before.exec(sql);
      before.exec(`PRAGMA user_version = ${CONTEXT_MIGRATION}`);
      before.exec(`INSERT INTO router_logs (final_provider, route_reason, outcome, latency_ms) VALUES ('openai', 'classified', 'ok', 5)`);
      before.close();
      const migrated = openTelemetryDb(beforeFile);
      assert.strictEqual((migrated.prepare('SELECT count(*) AS n FROM router_logs').get() as { n: number }).n, 1);
      assert.ok(recordContextRun(migrated, run, []) > 0);
      migrated.close();

      // The 62d6389 layout, which stored the prompt text: the column goes, the row stays.
      const v5File = join(dir, 'v5.db');
      const v5 = new DatabaseSync(v5File);
      for (const sql of MIGRATIONS.slice(0, CONTEXT_MIGRATION + 1)) v5.exec(sql);
      v5.exec(`PRAGMA user_version = ${CONTEXT_MIGRATION + 1}`);
      v5.exec(
        `INSERT INTO context_runs (session_id, prompt, mode, outcome, pipeline_version) VALUES ('old', 'my secret prompt', 'inject', 'jev_error', 'v')`,
      );
      v5.close();
      const fixed = openTelemetryDb(v5File);
      const row = fixed.prepare(`SELECT * FROM context_runs WHERE session_id = 'old'`).get() as Record<string, unknown>;
      assert.strictEqual(row['outcome'], 'jev_error');
      assert.ok(!('prompt' in row), 'plain-text prompts are dropped');
      fixed.close();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe('context-cli.e2e', () => {
  let jev: { server: Server; url: string };
  let dir: string;
  let envFile: string;
  let dbFile: string;

  before(async () => {
    jev = await listen((req, res) => void jevLike(req, res, 0.98));
    dir = mkdtempSync(join(tmpdir(), 'jev-ctx-cli-'));
    writeGraph(dir, 'proj');
    dbFile = join(dir, 'telemetry.db');
    envFile = join(dir, 'hook.env');
    writeFileSync(envFile, [`TYPESAFE_API_KEY=mock_api_key`, `JEV_API_URL=${jev.url}`, `CONTEXT_GRAPH_ROOT=${dir}`].join('\n'), 'utf8');
  });

  after(() => {
    jev.server.close();
    rmSync(dir, { recursive: true, force: true });
  });

  function hook(stdin: string, env: Record<string, string> = {}): Promise<{ code: number | null; stdout: string; stderr: string }> {
    return new Promise((done) => {
      const proc = spawn(process.execPath, ['--import', 'tsx', resolve('src/cli.ts'), 'context', '--env', envFile], {
        env: { ...process.env, TELEMETRY_DB_PATH: dbFile, ...env },
      });
      let stdout = '';
      let stderr = '';
      proc.stdout.on('data', (c) => (stdout += c));
      proc.stderr.on('data', (c) => (stderr += c));
      proc.on('close', (code) => done({ code, stdout, stderr }));
      proc.stdin.end(stdin);
    });
  }
  const payload = (session: string) => JSON.stringify({ prompt: 'how does selectRoute apply the risk veto?', session_id: session, cwd: join(dir, 'proj') });

  it('inject: prints only the hook JSON, with the relevant context', async () => {
    const r = await hook(payload('cli-inject'), { CONTEXT_MODE: 'inject' });
    assert.strictEqual(r.code, 0, r.stderr);
    const out = JSON.parse(r.stdout);
    assert.strictEqual(out.hookSpecificOutput.hookEventName, 'UserPromptSubmit');
    assert.match(out.hookSpecificOutput.additionalContext, /selectRoute/);
  });

  it('shadow with JEV ok: empty stdout, exit 0, the run is recorded', async () => {
    const r = await hook(payload('cli-shadow'), { CONTEXT_MODE: 'shadow' });
    assert.strictEqual(r.code, 0);
    assert.strictEqual(r.stdout, '');
    const db = new DatabaseSync(dbFile);
    const row = db.prepare(`SELECT outcome, mode FROM context_runs WHERE session_id = 'cli-shadow'`).get() as Record<string, unknown>;
    db.close();
    assert.deepStrictEqual({ ...row }, { outcome: 'ok', mode: 'shadow' });
  });

  it('JEV down: empty stdout, exit 0, and stderr says why', async () => {
    const r = await hook(payload('cli-down'), { CONTEXT_MODE: 'inject', JEV_API_URL: 'http://127.0.0.1:1/v1/systemone' });
    assert.strictEqual(r.code, 0);
    assert.strictEqual(r.stdout, '');
    assert.match(r.stderr, /jev_error: jev: /);
    assert.ok(!r.stderr.includes('mock_api_key'), 'the key never reaches stderr');
  });

  it('garbage on stdin: empty stdout, exit 0', async () => {
    for (const junk of ['not json', '', '{"prompt": 42}']) {
      const r = await hook(junk);
      assert.strictEqual(r.code, 0, junk);
      assert.strictEqual(r.stdout, '', junk);
    }
  });

  describe('529 and hanging requests tests', () => {
    let fake529: { server: Server; url: string };
    let reqs: IncomingMessage[] = [];
    let resolveWait: () => void;
    let waitPromise = new Promise<void>((r) => (resolveWait = r));
    let answered = false;

    before(async () => {
      fake529 = await listen(async (req, res) => {
        reqs.push(req);
        if (reqs.length >= 2) resolveWait();
        await waitPromise;
        if (!answered) {
          answered = true;
          res.writeHead(529, { 'content-type': 'application/json' });
          res.end(JSON.stringify({ error: 'Overloaded' }));
        } else {
          await new Promise(() => {}); // hang forever
        }
      });
      writeGraph(dir, 'proj');
      // A different graph (identical ones are loaded once), so rankCommunities sends two requests in parallel.
      writeGraph(dir, 'other', JSON.stringify({ ...GRAPH, nodes: GRAPH.nodes.map((n) => ({ ...n, label: `other${n.label}` })) }));
    });

    afterEach(() => {
      reqs = [];
      answered = false;
      waitPromise = new Promise<void>((r) => (resolveWait = r));
    });

    after(() => {
      fake529.server.closeAllConnections();
      fake529.server.close();
    });

    it('pipeline cancels all parallel in-flight JEV requests when one fails (529)', async () => {
      // Record into the test database, never into the real one.
      process.env['TELEMETRY_DB_PATH'] = dbFile;
      let res: Awaited<ReturnType<typeof runPipeline>>;
      try {
        res = await runPipeline(
          { prompt: 'hello', session_id: 'pipeline-529', cwd: join(dir, 'proj') },
          baseConfig(fake529.url, { graphRoot: dir, timeoutMs: 5000 })
        );
      } finally {
        delete process.env['TELEMETRY_DB_PATH'];
      }
      assert.strictEqual(res.outcome, 'jev_error');
      assert.match(res.error ?? '', /JEV HTTP 529/);

      const unansweredReqs = reqs.slice(1);
      assert.ok(unansweredReqs.length > 0);
      
      const closes = unansweredReqs.map((req) => 
        new Promise<void>((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('Socket did not close in time')), 1000);
          req.socket.once('close', () => {
            clearTimeout(timeout);
            resolve();
          });
        })
      );
      await Promise.all(closes);
    });

    it('cli e2e: 529 plus hanging fake exits under 3000ms', async () => {
      const start = performance.now();
      const r = await hook(payload('cli-529'), { CONTEXT_MODE: 'inject', JEV_API_URL: fake529.url });
      const elapsed = performance.now() - start;
      assert.strictEqual(r.code, 0);
      assert.strictEqual(r.stdout, '');
      assert.ok(elapsed < 3000);
    });
  });

  describe('timeouts and watchdog', () => {
    let fakeHanging: { server: Server; url: string };
    before(async () => {
      fakeHanging = await listen(async (req, res) => {
        await new Promise(() => {}); // never answers
      });
    });
    after(() => {
      fakeHanging.server.closeAllConnections();
      fakeHanging.server.close();
    });

    it('cli e2e: never answers, with long timeout and short budget', async () => {
      const start = performance.now();
      const r = await hook(payload('cli-timeout'), {
        CONTEXT_MODE: 'inject',
        JEV_API_URL: fakeHanging.url,
        CONTEXT_TIMEOUT_MS: '20000',
        CONTEXT_HOOK_BUDGET_MS: '2000'
      });
      const elapsed = performance.now() - start;
      assert.strictEqual(r.code, 0);
      assert.strictEqual(r.stdout, '');
      assert.ok(elapsed < 3500);

      const db = new DatabaseSync(dbFile);
      const row = db.prepare(`SELECT outcome FROM context_runs WHERE session_id = 'cli-timeout' ORDER BY id DESC LIMIT 1`).get() as { outcome: string };
      db.close();
      assert.strictEqual(row?.outcome, 'timeout');
    });

    it('cli e2e: watchdog kicks in when stdin is held open', async () => {
      const start = performance.now();
      
      const r = await new Promise<{ code: number | null; stdout: string; stderr: string }>((done) => {
        const proc = spawn(process.execPath, ['--import', 'tsx', resolve('src/cli.ts'), 'context', '--env', envFile], {
          env: { ...process.env, TELEMETRY_DB_PATH: dbFile, CONTEXT_HOOK_BUDGET_MS: '1000' },
        });
        let stdout = '';
        let stderr = '';
        proc.stdout.on('data', (c) => (stdout += c));
        proc.stderr.on('data', (c) => (stderr += c));
        proc.on('close', (code) => done({ code, stdout, stderr }));
        // never end stdin
      });

      const elapsed = performance.now() - start;
      assert.strictEqual(r.code, 0);
      assert.strictEqual(r.stdout, '');
      assert.ok(elapsed < 3000);
      assert.match(r.stderr, /hook budget of 1000 ms exceeded/);
    });
  });
});

describe('isProjectDir', () => {
  let tempHome: string;

  before(() => {
    tempHome = mkdtempSync(join(tmpdir(), 'jev-ctx-home-'));
  });

  after(() => {
    rmSync(tempHome, { recursive: true, force: true });
  });

  it('correctly identifies project directories and skips others', () => {
    const projDir = join(tempHome, 'my-proj');
    const subDir = join(projDir, 'src', 'sub');
    mkdirSync(subDir, { recursive: true });
    
    // no marker: false
    assert.strictEqual(isProjectDir(subDir, tempHome), false);

    // marker in cwd: true
    writeFileSync(join(subDir, '.git'), '');
    assert.strictEqual(isProjectDir(subDir, tempHome), true);
    rmSync(join(subDir, '.git'));

    // marker only in ancestor below home: true
    writeFileSync(join(projDir, 'package.json'), '{}');
    assert.strictEqual(isProjectDir(subDir, tempHome), true);

    // a dir with no marker: false
    const otherDir = join(tempHome, 'other');
    mkdirSync(otherDir);
    assert.strictEqual(isProjectDir(otherDir, tempHome), false);

    // home itself: false
    assert.strictEqual(isProjectDir(tempHome, tempHome), false);

    // missing path: false
    assert.strictEqual(isProjectDir(join(tempHome, 'does-not-exist'), tempHome), false);

    // filesystem root: false
    const root = parse(tempHome).root;
    assert.strictEqual(isProjectDir(root, tempHome), false);
  });
});

describe('pipeline skipped_cwd', () => {
  let tempDir: string;
  let dbFile: string;

  before(() => {
    tempDir = mkdtempSync(join(tmpdir(), 'jev-ctx-skipped-cwd-'));
    dbFile = join(tempDir, 'telemetry.db');
    process.env['TELEMETRY_DB_PATH'] = dbFile;
    openTelemetryDb(dbFile).close();
  });

  after(() => {
    delete process.env['TELEMETRY_DB_PATH'];
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('yields skipped_cwd outcome and zero requests when cwd is not a project', async () => {
    let reqs = 0;
    const fake = await listen((req, res) => {
      reqs++;
      res.end();
    });

    const cwd = join(tempDir, 'no-marker-dir');
    mkdirSync(cwd);

    try {
      const res = await runPipeline(
        { prompt: 'hello', session_id: 'sess-skipped-cwd', cwd },
        baseConfig(fake.url)
      );
      assert.strictEqual(res.outcome, 'skipped_cwd');
      assert.strictEqual(res.stdout, '');
      assert.strictEqual(reqs, 0);

      const db = new DatabaseSync(dbFile);
      const row = db.prepare(`SELECT outcome FROM context_runs WHERE session_id = 'sess-skipped-cwd'`).get() as { outcome: string };
      db.close();
      assert.strictEqual(row?.outcome, 'skipped_cwd');
    } finally {
      fake.server.close();
    }
  });
});
