import { meterAnthropicBody } from '../src/telemetry/usage-meter.js';
import { jevRequestBody, parseJevTextOnly, parseJevRisk, parseJevRiskScores } from '../src/classifier/jev-classifier.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { GeminiCliProvider } from '../src/providers/gemini/provider.js';
import { stat } from 'node:fs/promises';
import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { runAgy } from '../src/providers/gemini/agy-runner.js';
import { loadConfig, expandEnv } from '../src/config.js';
import { MessagesBody } from '../src/routing/messages-body.js';
import { hasImageDocumentOrToolChoice } from '../src/routing/messages-body.js';
import { geminiEligible } from '../src/domain/policy.js';
import { renderGeminiPrompt } from '../src/providers/gemini/prompt.js';
import { sseEvent } from '../src/providers/openai/sse.js';
import { CircuitBreaker } from '../src/providers/breaker.js';
import { ensureAgyHome } from '../src/providers/gemini/isolation.js';
import { rm, readFile, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { mkdtemp } from 'node:fs/promises';

describe('Gemini tier unit tests', () => {

  describe('Config', () => {
    it('defaults: gemini is undefined', () => {
      const c = loadConfig({ CHEAP_API_KEY: 'k', CLASSIFIER: 'heuristic' });
      assert.equal(c.gemini, undefined);
    });

    it('GEMINI_TIER=on gives correct defaults', () => {
      const c = loadConfig({ CHEAP_API_KEY: 'k', CLASSIFIER: 'heuristic', GEMINI_TIER: 'on', JEV_ROUTER_HOME: '/jev-home' });
      assert.ok(c.gemini);
      assert.equal(c.gemini.model, 'gemini-3.8-flash-high');
      assert.equal(c.gemini.timeoutMs, 60000);
      assert.equal(c.gemini.minTextOnly, 0.8);
      assert.equal(c.gemini.pressureMinTextOnly, 0.6);
      assert.equal(c.gemini.fromPrimary, true);
      assert.equal(c.gemini.maxConcurrency, 1);
      assert.equal(c.gemini.maxPromptChars, 200000);
      assert.equal(c.gemini.breakerFailures, 3);
      assert.equal(c.gemini.breakerCooldownMs, 300000);
      assert.equal(c.gemini.home, join('/jev-home', 'agy-home'));
    });

    it('expandEnv helper', () => {
      assert.equal(expandEnv('%LOCALAPPDATA%\\agy', { LOCALAPPDATA: 'C:\\App' }), 'C:\\App\\agy');
    });

    it('rejects bad config bounds', () => {
      assert.throws(() => loadConfig({ CHEAP_API_KEY: 'k', GEMINI_TIER: 'on', GEMINI_TIER_MAX_CONCURRENCY: '0' }));
      assert.throws(() => loadConfig({ CHEAP_API_KEY: 'k', GEMINI_TIER: 'on', GEMINI_TIER_MAX_CONCURRENCY: '5' }));
      assert.throws(() => loadConfig({ CHEAP_API_KEY: 'k', GEMINI_TIER: 'maybe' as any }));
    });
  });

  describe('Classifier', () => {
    it('jevRequestBody textOnly parameter', () => {
      const b1 = jevRequestBody('model1', 'hi' as any) as any;
      assert.equal(b1.questions.text_answer_suffices, undefined);
      
      const b2 = jevRequestBody('model1', 'hi' as any, true) as any;
      assert.equal(b2.questions.text_answer_suffices?.type, 'noul');
    });

    it('parseJevTextOnly', () => {
      assert.equal(parseJevTextOnly({} as any), undefined);
      assert.equal(parseJevTextOnly({ answers: {} } as any), undefined);
      assert.equal(parseJevTextOnly({ answers: { text_answer_suffices: { noul: 0.95 } } } as any), 0.95);
    });

    it('parseJevRisk with text_answer_suffices', () => {
      const payload = {
        answers: {
          text_answer_suffices: { noul: 0.99 },
          security_sensitive: { noul: 0.1 },
          destructive_or_production: { noul: 0.1 },
          requires_inspection: { noul: 0.1 }
        }
      } as any;
      assert.equal(parseJevRisk(payload), 0.1);
    });

    it('parseJevRiskScores keeps each risk Noul by question', () => {
      const payload = { answers: { security_sensitive: { noul: 0.1 }, destructive_or_production: { noul: 0.2 }, requires_inspection: { noul: 0.4 } } };
      assert.deepEqual(parseJevRiskScores(payload), { security_sensitive: 0.1, destructive_or_production: 0.2, requires_inspection: 0.4 });
      // A missing risk answer must fail the call (toward primary), never read as zero risk.
      assert.throws(() => parseJevRiskScores({ answers: { security_sensitive: { noul: 0.1 }, destructive_or_production: { noul: 0.2 } } }));
    });
  });

  describe('Policy/router - geminiEligible', () => {
    it('geminiEligible truth table', () => {
      const opts = { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6 };
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1 }, 'primary', { ...opts, enabled: false }, 'none'), false);
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1, textOnly: 0.9 }, 'trivial', opts, 'none'), false);
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1, risk: 0.6, textOnly: 0.9 }, 'primary', opts, 'none'), false);
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1 }, 'primary', opts, 'none'), false);
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1, textOnly: 0.79 }, 'primary', opts, 'none'), false);
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1, textOnly: 0.8 }, 'primary', opts, 'none'), true);
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1, textOnly: 0.65 }, 'primary', opts, 'pressure'), true);

      // Cheap route off: Gemini takes trivial/standard work itself, tools included, no text-only bar.
      const replacing = { ...opts, replacesCheap: true };
      assert.equal(geminiEligible({ simple: 0.95, standard: 0.05, structural: 0 }, 'trivial', replacing, 'none'), true);
      assert.equal(geminiEligible({ simple: 0.4, standard: 0.5, structural: 0.1, textOnly: 0.1 }, 'standard', replacing, 'none'), true);
      // Structural work still needs the text-only bar to leave Claude.
      assert.equal(geminiEligible({ simple: 0, standard: 0, structural: 1, textOnly: 0.1 }, 'primary', replacing, 'none'), false);
      assert.equal(geminiEligible({ simple: 0.95, standard: 0, structural: 0 }, 'trivial', { ...replacing, enabled: false }, 'none'), false);

      // Tool-enabled mode (minTextOnly=0): allows requires_inspection >= 0.5 and low textOnly, blocks only hardRisk >= 0.5
      const toolOpts = { enabled: true, minTextOnly: 0, pressureMinTextOnly: 0 };
      assert.equal(
        geminiEligible(
          {
            simple: 0.05,
            standard: 0.15,
            structural: 0.8,
            risk: 0.95,
            riskScores: { security_sensitive: 0.05, destructive_or_production: 0.05, requires_inspection: 0.95 },
            textOnly: 0.02,
          },
          'primary',
          toolOpts,
          'none',
        ),
        true,
      );
      assert.equal(
        geminiEligible(
          {
            simple: 0.05,
            standard: 0.15,
            structural: 0.8,
            risk: 0.92,
            riskScores: { security_sensitive: 0.92, destructive_or_production: 0.05, requires_inspection: 0.5 },
            textOnly: 0.02,
          },
          'primary',
          toolOpts,
          'none',
        ),
        false,
      );
    });
  });

  describe('messages-body hasImageDocumentOrToolChoice', () => {
    it('detects tools choice and images', () => {
      assert.equal(hasImageDocumentOrToolChoice({ messages: [{ role: 'user', content: [{ type: 'image', source: {} as any }] }] } as any), true);
      assert.equal(hasImageDocumentOrToolChoice({ messages: [{ role: 'user', content: 'hello' }], tool_choice: { type: 'any' } } as any), true);
      assert.equal(hasImageDocumentOrToolChoice({ messages: [{ role: 'user', content: 'hello' }] } as any), false);
    });
  });

  describe('Runner (fake spawn)', () => {
    it('success', async () => {
      let spawnCwd: string = '';
      let spawnArgs: string[] = [];
      let spawnEnv: any;
      let stdinData = '';

      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { stdinData += c; cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 100;
      
      const spawnFn = (bin: string, args: string[], opts: any) => {
        spawnArgs = args;
        spawnCwd = opts.cwd;
        spawnEnv = opts.env;
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'hello world', usage: { input_tokens: 10, output_tokens: 2 } } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 10);
        return fakeChild;
      };

      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: true, text: 'hello world', usage: { inputTokens: 10, outputTokens: 2 } });
      assert.ok(spawnArgs.includes('--input-format'));
      assert.ok(spawnArgs.includes('stream-json'));
      assert.ok(spawnArgs[spawnArgs.length - 1] === '--print=');
      assert.ok(!spawnArgs.includes('--dangerously-skip-permissions'));
      assert.ok(!spawnArgs.includes('--add-dir'));
      assert.equal(spawnEnv.USERPROFILE, '/tmp/home');
      assert.equal(spawnEnv.HOME, '/tmp/home');
      
      const parsed = JSON.parse(stdinData.trim());
      assert.equal(parsed.event, 'user');
      assert.equal(parsed.message.content, 'hello');
      
      // Check temp cwd is gone
      await assert.rejects(async () => await readFile(spawnCwd));
    });

    it('tool step -> kill and failure', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 101;
      
      const spawnFn = (bin: string, args: string[], opts: any) => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'step_update', step_update: { step_type: 'tool', tool_name: 'write_to_file' } }) + '\n'));
          setTimeout(() => fakeChild.emit('close', 1), 10);
        }, 10);
        return fakeChild;
      };

      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'tool attempt: write_to_file' });
    });

    it('hang -> timeout', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 102;
      
      const spawnFn = () => fakeChild;

      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 50, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'timeout' });
    });
    it('non-zero exit', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 103;
      const spawnFn = () => {
        setTimeout(() => fakeChild.emit('close', 1), 5);
        return fakeChild;
      };
      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'exit 1' });
    });

    it('status ERROR', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 104;
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'ERROR', error: 'boom' } }) + '\n'));
          fakeChild.emit('close', 1);
        }, 5);
        return fakeChild;
      };
      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'status ERROR: boom' });
    });

    it('no result event', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 105;
      const spawnFn = () => {
        setTimeout(() => fakeChild.emit('close', 0), 5);
        return fakeChild;
      };
      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'no result event' });
    });

    it('whitespace response', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 106;
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: '   ' } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 5);
        return fakeChild;
      };
      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'empty response' });
    });

    it('abort signal', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 107;
      const spawnFn = () => fakeChild;
      const ac = new AbortController();
      const p = runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home', signal: ac.signal }, spawnFn as any);
      ac.abort();
      const res = await p;
      assert.deepEqual(res, { ok: false, reason: 'aborted' });
    });

    it('chunk boundaries', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 108;
      const spawnFn = () => {
        setTimeout(() => {
          const str = JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'hello world', usage: { input_tokens: 1, output_tokens: 1 } } }) + '\r\n';
          fakeChild.stdout.emit('data', Buffer.from(str.slice(0, 15)));
          setTimeout(() => {
            fakeChild.stdout.emit('data', Buffer.from(str.slice(15)));
            fakeChild.emit('close', 0);
          }, 5);
        }, 5);
        return fakeChild;
      };
      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: true, text: 'hello world', usage: { inputTokens: 1, outputTokens: 1 } });
    });

    it('cleanup deletes only brain/<id>', async () => {
      const home = await mkdtemp(join(tmpdir(), 'cleanup-home-'));
      const id = 'abc-123';
      const brainDir = join(home, '.gemini', 'antigravity-cli', 'brain');
      const annDir = join(home, '.gemini', 'antigravity-cli', 'annotations');
      const convDir = join(home, '.gemini', 'antigravity-cli', 'conversations');
      await mkdir(join(brainDir, id), { recursive: true });
      await mkdir(join(brainDir, 'other'), { recursive: true });
      await mkdir(annDir, { recursive: true });
      await writeFile(join(annDir, id + '.pbtxt'), 'test');
      await writeFile(join(annDir, 'other.pbtxt'), 'test');
      await mkdir(convDir, { recursive: true });
      await writeFile(join(convDir, id + '.txt'), 'test');
      await writeFile(join(convDir, 'other.txt'), 'test');

      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 109;
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'conversation_start', conversation_id: id }) + '\n'));
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'ok' } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 5);
        return fakeChild;
      };
      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home }, spawnFn as any);
      
      const brainExists = await stat(join(brainDir, id)).catch(() => null);
      assert.ok(!brainExists);
      const otherBrain = await stat(join(brainDir, 'other')).catch(() => null);
      assert.ok(otherBrain);
      const annExists = await stat(join(annDir, id + '.pbtxt')).catch(() => null);
      assert.ok(!annExists);
      const otherAnn = await stat(join(annDir, 'other.pbtxt')).catch(() => null);
      assert.ok(otherAnn);
      const convExists = await stat(join(convDir, id + '.txt')).catch(() => null);
      assert.ok(!convExists);
      const otherConv = await stat(join(convDir, 'other.txt')).catch(() => null);
      assert.ok(otherConv);

      await rm(home, { recursive: true, force: true });
    });

    it('strips API keys from env', async () => {
      let spawnEnv: any;
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 110;
      const spawnFn = (bin: string, args: string[], opts: any) => {
        spawnEnv = opts.env;
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'ok' } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 5);
        return fakeChild;
      };

      process.env.GEMINI_API_KEY = 'x';
      process.env.google_api_key = 'y';
      try {
        await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
        assert.ok(!('GEMINI_API_KEY' in spawnEnv));
        assert.ok(!('google_api_key' in spawnEnv));
        assert.ok(!('GOOGLE_API_KEY' in spawnEnv));
      } finally {
        delete process.env.GEMINI_API_KEY;
        delete process.env.google_api_key;
      }
    });


    it('spawn emits error ENOENT', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new EventEmitter();
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      // no pid
      
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.emit('error', new Error('ENOENT'));
        }, 5);
        return fakeChild;
      };

      const res = await runAgy('hello', { bin: 'agy', model: 'm1', timeoutMs: 1000, home: '/tmp/home' }, spawnFn as any);
      assert.deepEqual(res, { ok: false, reason: 'spawn failed: ENOENT' });
    });
  });

  describe('Prompt translation', () => {
    it('no body.system, reminders capped and deduplicated, oldest dropped', () => {
      const body: MessagesBody = {
        model: 'm',
        messages: [
          { role: 'user', content: 'first' },
          { role: 'assistant', content: 'ack' },
          { role: 'user', content: '<system-reminder>r1</system-reminder>second' },
          { role: 'user', content: '<system-reminder>r1</system-reminder>third' },
        ],
        system: [{ type: 'text', text: 'IGNORE THIS' }]
      } as any;

      const prompt = renderGeminiPrompt(body, 200000);
      assert.ok(prompt);
      assert.ok(!prompt.includes('IGNORE THIS'));
      assert.ok(prompt.includes('System reminders:\n<system-reminder>r1</system-reminder>'));
      assert.equal(prompt.match(/<system-reminder>r1<\/system-reminder>/g)?.length, 1);
      assert.ok(prompt.includes('### User\nfirst'));
      assert.ok(prompt.includes('### User\nsecond'));
    });

    it('latest message alone too big -> undefined', () => {
      const body: MessagesBody = {
        model: 'm',
        messages: [{ role: 'user', content: 'very long message' }],
      } as any;
      const prompt = renderGeminiPrompt(body, 10);
      assert.equal(prompt, undefined);
    });

    it('tool_result rendering', () => {
      const body: MessagesBody = {
        model: 'm',
        messages: [{ role: 'user', content: [{ type: 'tool_result', tool_use_id: '1', content: [{ type: 'text', text: 'result text' }] }] }],
      } as any;
      const prompt = renderGeminiPrompt(body, 200000);
      assert.ok(prompt!.includes('[tool result: result text]'));
    });

    it('renders Available Client Tools and extracts <env> from body.system when tools are provided', () => {
      const body: MessagesBody = {
        model: 'm',
        system: [{ type: 'text', text: 'LONG SYSTEM PROMPT\n<env>\nWorking directory: C:\\Dev\\JEV-APP-PC\n</env>\nMORE SYSTEM' }],
        tools: [
          {
            name: 'Read',
            description: 'Reads a file from the local filesystem.',
            input_schema: {
              type: 'object',
              properties: { file_path: { type: 'string' }, limit: { type: 'number' } },
              required: ['file_path'],
            },
          },
        ],
        messages: [{ role: 'user', content: 'read package.json' }],
      } as any;
      const prompt = renderGeminiPrompt(body, 200000);
      assert.ok(prompt);
      assert.ok(!prompt.includes('LONG SYSTEM PROMPT'));
      assert.ok(prompt.includes('<env>\nWorking directory: C:\\Dev\\JEV-APP-PC\n</env>'));
      assert.ok(prompt.includes('Available Client Tools'));
      assert.ok(prompt.includes('- Read: Reads a file from the local filesystem. | input: {"type":"object","properties":{"file_path":{"type":"string"},"limit":{"type":"number"}},"required":["file_path"]}'));
    });

    it('renders nested tool schemas whole, minus prose (regression: TodoWrite items were flattened to "array")', () => {
      const body = {
        tools: [
          {
            name: 'TodoWrite',
            description: 'Writes the todo list.',
            input_schema: {
              $schema: 'http://json-schema.org/draft-07/schema#',
              type: 'object',
              additionalProperties: false,
              properties: {
                todos: {
                  type: 'array',
                  description: 'The updated todo list',
                  items: {
                    type: 'object',
                    properties: {
                      content: { type: 'string' },
                      status: { type: 'string', enum: ['pending', 'in_progress', 'completed'] },
                      description: { type: 'string', description: 'a field named description' },
                    },
                    required: ['content', 'status'],
                  },
                },
              },
              required: ['todos'],
            },
          },
        ],
        messages: [{ role: 'user', content: 'plan it' }],
      } as any;
      const prompt = renderGeminiPrompt(body, 200000)!;
      assert.ok(
        prompt.includes(
          '- TodoWrite: Writes the todo list. | input: {"type":"object","properties":{"todos":{"type":"array","items":{"type":"object","properties":{"content":{"type":"string"},"status":{"type":"string","enum":["pending","in_progress","completed"]},"description":{"type":"string"}},"required":["content","status"]}}},"required":["todos"]}',
        ),
      );
      assert.ok(!prompt.includes('$schema') && !prompt.includes('The updated todo list'));
    });
  });

  describe('Breaker', () => {
    it('state machine', () => {
      const b = new CircuitBreaker(3, 1000, 1);
      assert.equal(b.state, 'closed');
      assert.ok(b.acquire());
      b.release('failure'); // 1
      assert.equal(b.state, 'closed');
      assert.ok(b.acquire());
      b.release('failure'); // 2
      assert.ok(b.acquire());
      b.release('failure'); // 3
      assert.equal(b.state, 'open');
      assert.equal(b.acquire(), false);
      
      // Let's pretend cooldown passed (hack the time)
      (b as any).lastFailureTime = 0; 
      assert.equal(b.state, 'half-open');
      assert.ok(b.acquire());
      assert.equal(b.acquire(), false); // Only one in half open
      b.release('success');
      assert.equal(b.state, 'closed');
    });
  });

  describe('Isolation', () => {
    it('ensureAgyHome writes settings', async () => {
      const home = await mkdtemp(join(tmpdir(), 'agy-home-'));
      await ensureAgyHome(home);
      const conf = await readFile(join(home, '.gemini', 'antigravity-cli', 'settings.json'), 'utf8');
      assert.ok(conf.includes('"allowNonWorkspaceAccess": false'));
      assert.ok(conf.includes('"command(*)"'));
      await rm(home, { recursive: true, force: true });
    });
  });
        describe('Provider / translation', () => {
    it('non-streaming translation', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 200;
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: 'hello', usage: { input_tokens: 5, output_tokens: 6 } } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 5);
        return fakeChild;
      };

      const p = new GeminiCliProvider({
        bin: 'agy', model: 'model1', timeoutMs: 1000, home: '/tmp/home', maxPromptChars: 1000, spawnFn
      } as any);
      
      const req = { body: { model: 'other', messages: [{ role: 'user', content: 'hi' }] } } as any;
      const res = await p.send(req);
      
      assert.equal(res.kind, 'response');
      const responseRes = res as any;
      
      const metered = meterAnthropicBody(responseRes.body, 'application/json', undefined);
      
      let sentStr = '';
      for await (const chunk of metered.body) {
        sentStr += chunk.toString();
      }
      await metered.settled;
      assert.deepEqual(metered.usage(), { tokensIn: 5, tokensOut: 6, cacheReadTokens: 0, cacheWriteTokens: 0 });
      
      const body = JSON.parse(sentStr);
      assert.equal(body.type, 'message');
      assert.equal(body.role, 'assistant');
      assert.equal(body.model, 'model1');
      assert.deepEqual(body.content, [{ type: 'text', text: 'hello' }]);
      assert.equal(body.stop_reason, 'end_turn');
      assert.deepEqual(body.usage, { input_tokens: 5, output_tokens: 6 });
      assert.ok(body.id.startsWith('msg_jev_'));
    });

    it('streaming translation', async () => {
      const longText = 'a'.repeat(5000);
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 201;
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'result', result: { status: 'SUCCESS', response: longText, usage: { input_tokens: 5, output_tokens: 6 } } }) + '\n'));
          fakeChild.emit('close', 0);
        }, 5);
        return fakeChild;
      };

      const p = new GeminiCliProvider({
        bin: 'agy', model: 'model1', timeoutMs: 1000, home: '/tmp/home', maxPromptChars: 1000, spawnFn
      } as any);
      
      const req = { body: { model: 'other', messages: [{ role: 'user', content: 'hi' }], stream: true } } as any;
      const res = await p.send(req);
      
      assert.equal(res.kind, 'response');
      const streamRes = res as any;
      assert.equal(streamRes.headers['cache-control'], 'no-cache');
      assert.equal(streamRes.headers['content-type'], 'text/event-stream');
      
      const metered = meterAnthropicBody(streamRes.body, 'text/event-stream', undefined);
      
      let sentStr = '';
      for await (const chunk of metered.body) {
        sentStr += chunk.toString();
      }
      await metered.settled;
      assert.deepEqual(metered.usage(), { tokensIn: 5, tokensOut: 6, cacheReadTokens: 0, cacheWriteTokens: 0 });
      
      const events = [];
      const lines = sentStr.split('\n');
      let currentData = '';
      let currentEvent = '';
      for (const line of lines) {
        if (line.startsWith('event: ')) currentEvent = line.slice(7).trim();
        else if (line.startsWith('data: ')) currentData += line.slice(6).trim();
        else if (line === '') {
          if (currentEvent && currentData) {
            events.push({ event: currentEvent, data: JSON.parse(currentData) });
            currentEvent = '';
            currentData = '';
          }
        }
      }

      assert.equal(events[0]?.event, 'message_start');
      assert.deepEqual(events[0]?.data?.message?.content, []);
      assert.equal(events[0]?.data?.message?.stop_reason, null);
      assert.equal(events[0]?.data?.message?.model, 'model1');
      assert.equal(events[1]?.event, 'content_block_start');
      assert.equal(events[2]?.event, 'content_block_delta');
      
      let reconstructed = '';
      let stopReason = null;
      let msgStop = false;
      for (const ev of events) {
        if (ev.event === 'content_block_delta') reconstructed += ev.data.delta.text;
        if (ev.event === 'message_delta') stopReason = ev.data.delta.stop_reason;
        if (ev.event === 'message_stop') msgStop = true;
      }
      assert.equal(reconstructed, longText);
      assert.equal(stopReason, 'end_turn');
      assert.ok(msgStop);
    });

    it('translates <jev_tool_use> into Anthropic tool_use blocks in both JSON and SSE modes', async () => {
      const responseWithTools =
        'Vou ler o arquivo.\n<jev_tool_use>{"name":"Read","input":{"file_path":"package.json"}}</jev_tool_use>';
      const makeSpawn = () => () => {
        const fakeChild = new EventEmitter() as any;
        fakeChild.stdin = new Writable({ write(_c, _e, cb) { cb(); } });
        fakeChild.stdout = new EventEmitter();
        fakeChild.stderr = new EventEmitter();
        fakeChild.pid = 205;
        setTimeout(() => {
          fakeChild.stdout.emit(
            'data',
            Buffer.from(
              JSON.stringify({
                event: 'result',
                result: { status: 'SUCCESS', response: responseWithTools, usage: { input_tokens: 12, output_tokens: 18 } },
              }) + '\n',
            ),
          );
          fakeChild.emit('close', 0);
        }, 5);
        return fakeChild;
      };

      const pJson = new GeminiCliProvider({
        bin: 'agy', model: 'model1', timeoutMs: 1000, home: '/tmp/home', maxPromptChars: 5000, spawnFn: makeSpawn()
      } as any);
      const resJson = await pJson.send({ body: { model: 'other', messages: [{ role: 'user', content: 'read' }] } } as any);
      assert.equal(resJson.kind, 'response');
      const meteredJson = meterAnthropicBody((resJson as any).body, 'application/json', undefined);
      let jsonStr = '';
      for await (const chunk of meteredJson.body) jsonStr += chunk.toString();
      await meteredJson.settled;
      assert.equal(meteredJson.toolCalls(), 1);
      const parsedJson = JSON.parse(jsonStr);
      assert.equal(parsedJson.stop_reason, 'tool_use');
      assert.equal(parsedJson.content.length, 2);
      assert.deepEqual(parsedJson.content[0], { type: 'text', text: 'Vou ler o arquivo.' });
      assert.equal(parsedJson.content[1].type, 'tool_use');
      assert.equal(parsedJson.content[1].name, 'Read');
      assert.deepEqual(parsedJson.content[1].input, { file_path: 'package.json' });

      const pSse = new GeminiCliProvider({
        bin: 'agy', model: 'model1', timeoutMs: 1000, home: '/tmp/home', maxPromptChars: 5000, spawnFn: makeSpawn()
      } as any);
      const resSse = await pSse.send({ body: { model: 'other', messages: [{ role: 'user', content: 'read' }], stream: true } } as any);
      assert.equal(resSse.kind, 'response');
      const meteredSse = meterAnthropicBody((resSse as any).body, 'text/event-stream', undefined);
      let sseStr = '';
      for await (const chunk of meteredSse.body) sseStr += chunk.toString();
      await meteredSse.settled;
      assert.equal(meteredSse.toolCalls(), 1);
      assert.ok(sseStr.includes('"type":"tool_use"'));
      assert.ok(sseStr.includes('"stop_reason":"tool_use"'));
    });

    it('JEV_NEEDS_TOOLS and prompt over budget', async () => {
      const fakeChild = new EventEmitter() as any;
      fakeChild.stdin = new Writable({ write(c, e, cb) { cb(); } });
      fakeChild.stdout = new EventEmitter();
      fakeChild.stderr = new EventEmitter();
      fakeChild.pid = 202;
      const spawnFn = () => {
        setTimeout(() => {
          fakeChild.stdout.emit('data', Buffer.from(JSON.stringify({ event: 'step_update', step_update: { step_type: 'tool', tool_name: 'write_to_file' } }) + '\n'));
          setTimeout(() => fakeChild.emit('close', 1), 5);
        }, 5);
        return fakeChild;
      };

      const p = new GeminiCliProvider({
        bin: 'agy', model: 'model1', timeoutMs: 1000, home: '/tmp/home', maxPromptChars: 1000, spawnFn
      } as any);
      
      let res = await p.send({ body: { messages: [{ role: 'user', content: 'hi' }] } } as any);
      assert.deepEqual(res, { kind: 'unavailable', reason: 'tool attempt: write_to_file' });

      // over budget (doesn't even call spawnFn)
      res = await p.send({ body: { messages: [{ role: 'user', content: 'x'.repeat(2000) }] } } as any);
      assert.deepEqual(res, { kind: 'unavailable', reason: 'transcript budget exceeded' });
    });
  });
});
