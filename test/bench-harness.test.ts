import assert from 'node:assert/strict';
import { createServer, type IncomingMessage, type Server } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, beforeEach, describe, it } from 'node:test';
import { OpenAICompatibleProvider } from '../src/providers/openai/provider.js';
import { runTrial, summarize, TASKS, TOOLS, Workspace } from '../scripts/bench/harness.js';

type Scripted = { status: number; body: string } | { chunks: object[] };

const toolCall = (name: string, args: string, id = 'call_1') => ({
  chunks: [
    { choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id, type: 'function', function: { name, arguments: args } }] } }] },
    { choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] },
    { choices: [], usage: { prompt_tokens: 100, completion_tokens: 20 } },
  ],
});

const text = (content: string) => ({
  chunks: [
    { choices: [{ index: 0, delta: { content } }] },
    { choices: [{ index: 0, delta: {}, finish_reason: 'stop' }] },
    { choices: [], usage: { prompt_tokens: 150, completion_tokens: 5 } },
  ],
});

const task = (id: string) => TASKS.find((t) => t.id === id)!;
const options = { maxSteps: 8, padKb: 1, maxTokens: 1024, stepTimeoutMs: 5_000 };

async function readJson(req: IncomingMessage): Promise<any> {
  const chunks: Buffer[] = [];
  for await (const c of req) chunks.push(c as Buffer);
  return JSON.parse(Buffer.concat(chunks).toString('utf8'));
}

describe('bench harness through OpenAICompatibleProvider', () => {
  let server: Server;
  let provider: OpenAICompatibleProvider;
  let script: Scripted[] = [];
  let requests: any[] = [];

  before(async () => {
    server = createServer(async (req, res) => {
      requests.push(await readJson(req));
      const next = script.shift() ?? { status: 500, body: 'script exhausted' };
      if ('status' in next) {
        res.writeHead(next.status).end(next.body);
        return;
      }
      res.writeHead(200, { 'content-type': 'text/event-stream' });
      for (const c of next.chunks) res.write(`data: ${JSON.stringify(c)}\n\n`);
      res.end('data: [DONE]\n\n');
    });
    await new Promise<void>((r) => server.listen(0, '127.0.0.1', r));
    const { port } = server.address() as AddressInfo;
    provider = new OpenAICompatibleProvider({
      baseUrl: `http://127.0.0.1:${port}/v1`,
      apiKey: 'test',
      model: 'gpt-oss:20b',
      maxOutputTokens: 4096,
      timeoutMs: 5_000,
    });
  });
  after(() => server.close());
  beforeEach(() => {
    script = [];
    requests = [];
  });

  it('scores a Read → Edit → answer loop as success and replays tool results', async () => {
    script = [
      toolCall('Read', '{"file_path":"/workspace/math.ts"}'),
      toolCall('Edit', '{"file_path":"/workspace/math.ts","old_string":"return a - b;","new_string":"return a + b;"}', 'call_2'),
      text('Fixed add.'),
    ];
    const r = await runTrial(provider, task('fix-bug'), options);

    assert.equal(r.outcome, 'success', r.detail);
    assert.equal(r.steps, 3);
    assert.deepEqual(r.toolCalls, ['Read', 'Edit']);
    assert.equal(r.inputTokens, 350);
    // The model saw strict tool schemas and, on step 2, the Read output as a tool message.
    const read = requests[0].tools.find((t: any) => t.function.name === 'Read');
    assert.equal(read.function.parameters.additionalProperties, false);
    assert.equal(requests[0].model, 'gpt-oss:20b');
    const toolMsg = requests[1].messages.find((m: any) => m.role === 'tool');
    assert.match(toolMsg.content, /2→ {2}return a - b;/);
  });

  it('counts recoverable tool errors and still judges the final result', async () => {
    script = [
      toolCall('ReadFile', '{"path":"math.ts"}'),
      toolCall('Read', '{"file_path":"math.ts"}'),
      toolCall('Read', '{"file_path":"/workspace/math.ts"}'),
      toolCall('Edit', '{"file_path":"/workspace/math.ts","old_string":"     2→  return a - b;","new_string":"  return a + b;"}'),
      text('Done.'),
    ];
    const r = await runTrial(provider, task('fix-bug'), options);

    assert.equal(r.outcome, 'wrong_result');
    assert.equal(r.counters.hallucinatedTool, 1);
    assert.equal(r.counters.schemaViolation, 1);
    assert.equal(r.counters.editMiss, 1);
    assert.equal(r.toolErrors.length, 3);
    assert.match(r.toolErrors[1] ?? '', /^Read\(\{"file_path":"math.ts"\}\): InputValidationError: .*absolute path/);
  });

  // Tool calls are buffered to the end, so invalid JSON breaks the stream before any block
  // reaches the client: the provider reports it unavailable and the router can still fail over.
  it('reports invalid tool JSON before any content as a fail-over, not a stream_error', async () => {
    script = [toolCall('Read', '{"file_path": "/workspace/math.ts"')];
    const r = await runTrial(provider, task('fix-bug'), options);
    assert.equal(r.outcome, 'fallback');
    assert.match(r.detail ?? '', /invalid JSON/);
  });

  it('reports a refused request as fallback', async () => {
    script = [{ status: 503, body: 'overloaded' }];
    const r = await runTrial(provider, task('read-answer'), options);
    assert.equal(r.outcome, 'fallback');
    assert.match(r.detail ?? '', /HTTP 503/);
  });

  it('stops at the step budget', async () => {
    script = Array.from({ length: 3 }, () => toolCall('Glob', '{"pattern":"**/*.json"}'));
    const r = await runTrial(provider, task('read-answer'), { ...options, maxSteps: 3 });
    assert.equal(r.outcome, 'max_steps');
  });
});

describe('Workspace', () => {
  it('enforces read-before-edit and unique matches like Claude Code', () => {
    const ws = new Workspace({ 'a.ts': 'x\nx\n' });
    assert.equal(ws.run('Edit', { file_path: '/workspace/a.ts', old_string: 'x', new_string: 'y' }).isError, true);
    assert.equal(ws.counters.editWithoutRead, 1);
    ws.run('Read', { file_path: '/workspace/a.ts' });
    assert.match(ws.run('Edit', { file_path: '/workspace/a.ts', old_string: 'x', new_string: 'y' }).content, /Found 2 matches/);
    ws.run('Edit', { file_path: '/workspace/a.ts', old_string: 'x', new_string: 'y', replace_all: true });
    assert.equal(ws.file('a.ts'), 'y\ny\n');
  });

  it('globs and rejects unknown parameters', () => {
    const ws = new Workspace({ 'src/a.ts': '', 'b.json': '' });
    assert.equal(ws.run('Glob', { pattern: '**/*.ts' }).content, '/workspace/src/a.ts');
    assert.equal(ws.run('Read', { file_path: '/workspace/b.json', encoding: 'utf8' }).isError, true);
    assert.equal(ws.counters.schemaViolation, 1);
  });

  it('reads from the first line on offset 0, like Claude Code', () => {
    const ws = new Workspace({ 'a.ts': 'one\ntwo' });
    const out = ws.run('Read', { file_path: '/workspace/a.ts', offset: 0, limit: 1 });
    assert.equal(out.isError, false);
    assert.match(out.content, /^ +1→one$/);
    assert.equal(ws.counters.schemaViolation, 0);
  });

  it('exposes every tool with a JSON schema', () => {
    assert.deepEqual(TOOLS.map((t) => t.name), ['Read', 'Edit', 'Write', 'Glob', 'Grep', 'Bash']);
    assert.ok(TOOLS.every((t) => t.input_schema['type'] === 'object' && !('$schema' in t.input_schema)));
  });
});

describe('summarize', () => {
  it('separates provider failures from wrong work', () => {
    const base = { toolErrors: [], detail: undefined, steps: 1, toolCalls: [], latencyMs: 10, inputTokens: 0, outputTokens: 0 };
    const counters = { hallucinatedTool: 0, schemaViolation: 1, fileNotFound: 0, editWithoutRead: 0, editMiss: 0, disallowedTool: 0 };
    const s = summarize('m', [
      { ...base, task: 'a', outcome: 'success', counters },
      { ...base, task: 'a', outcome: 'stream_error', counters },
      { ...base, task: 'b', outcome: 'wrong_result', counters },
      { ...base, task: 'b', outcome: 'fallback', counters },
    ]);
    assert.equal(s.successRate, 0.25);
    assert.equal(s.failureRate, 0.5);
    assert.equal(s.counters.schemaViolation, 4);
    assert.deepEqual(s.byTask['b']?.outcomes, { wrong_result: 1, fallback: 1 });
  });
});
