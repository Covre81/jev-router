import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { Writable } from 'node:stream';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import {
  CURSOR_HARMLESS_TOOLS,
  isHarmlessCursorTool,
  runCursorAgent,
  toolKind,
} from '../src/providers/cursor/agent-runner.js';

function fakeChild(script: (child: any) => void) {
  const child = new EventEmitter() as any;
  child.pid = 99999;
  child.stdin = new Writable({ write(_c, _e, cb) { cb(); } });
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  setTimeout(() => script(child), 5);
  return child;
}

async function fakeCursorRoot() {
  const root = join(tmpdir(), `cursor-test-harmless-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const ver = join(root, 'versions', '2026.04.01-abc');
  await mkdir(ver, { recursive: true });
  await writeFile(join(ver, 'node.exe'), 'FAKE');
  await writeFile(join(ver, 'index.js'), 'FAKE');
  return { root, ver };
}

const line = (o: unknown) => Buffer.from(JSON.stringify(o) + '\n');

describe('CURSOR_HARMLESS_TOOLS allowlist and isHarmlessCursorTool', () => {
  it('allowlist exports expected harmless tools without side effects outside session state', () => {
    const expected = [
      'update todos',
      'todo write',
      'read todos',
      'todo read',
      'get mcp tools',
      'list mcp tools',
    ];
    assert.equal(CURSOR_HARMLESS_TOOLS.size, 6);
    for (const tool of expected) {
      assert.equal(CURSOR_HARMLESS_TOOLS.has(tool), true, `missing allowed tool: ${tool}`);
    }
    // Strict blocklist check: none of these should ever be in CURSOR_HARMLESS_TOOLS
    const forbidden = [
      'shell',
      'run',
      'terminal',
      'edit',
      'write',
      'delete',
      'apply patch',
      'mcp',
      'call mcp tool',
      'web search',
      'fetch',
      'read',
      'grep',
      'glob',
    ];
    for (const tool of forbidden) {
      assert.equal(CURSOR_HARMLESS_TOOLS.has(tool), false, `forbidden tool in allowlist: ${tool}`);
    }
  });

  describe('isHarmlessCursorTool tool_call keys matching', () => {
    it('allows todo-list tools (updateTodosToolCall, todoWriteToolCall, readTodosToolCall, todoReadToolCall)', () => {
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { updateTodosToolCall: {} } }), true);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { todoWriteToolCall: {} } }), true);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { readTodosToolCall: {} } }), true);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { todoReadToolCall: {} } }), true);
    });

    it('allows MCP discovery tools (getMcpToolsToolCall, listMcpToolsToolCall)', () => {
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { getMcpToolsToolCall: {} } }), true);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { listMcpToolsToolCall: {} } }), true);
    });

    it('normalizes case-insensitively and handles whitespace / raw name properties', () => {
      assert.equal(isHarmlessCursorTool({ name: 'UPDATE TODOS' }), true);
      assert.equal(isHarmlessCursorTool({ tool_name: 'Get MCP Tools' }), true);
      assert.equal(isHarmlessCursorTool({ toolName: '  todo write  ' }), true);
      assert.equal(isHarmlessCursorTool({ kind: 'list_mcp_tools' }), true);
      assert.equal(isHarmlessCursorTool({ name: 'read_todos' }), true);
    });

    it('enforces exact-match check: substrings and supersets must NOT sneak through', () => {
      assert.equal(isHarmlessCursorTool({ name: 'update todos and run' }), false);
      assert.equal(isHarmlessCursorTool({ name: 'run update todos' }), false);
      assert.equal(isHarmlessCursorTool({ name: 'get mcp tools and call' }), false);
      assert.equal(isHarmlessCursorTool({ name: 'todo write code' }), false);
      assert.equal(isHarmlessCursorTool({ tool_call: { updateTodosAndRunToolCall: {} } }), false);
    });

    it('blocks dangerous/side-effect tools explicitly', () => {
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { shellToolCall: { args: { command: 'ls' } } } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { terminalToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { runToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { editToolCall: { args: { path: 'a.ts' } } } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { writeToolCall: { args: { path: 'a.ts' } } } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { deleteToolCall: { args: { path: 'a.ts' } } } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { applyPatchToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { mcpToolCall: { args: { name: 'db' } } } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { callMcpToolToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { webSearchToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { fetchToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { readToolCall: { args: { path: 'prompt.md' } } } }), false); // read is handled by isPromptFileRead, not harmless tool
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { readToolCall: { args: { path: 'secret.txt' } } } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { grepToolCall: {} } }), false);
      assert.equal(isHarmlessCursorTool({ type: 'tool_call', subtype: 'started', tool_call: { globToolCall: {} } }), false);
    });

    it('returns false for null, undefined, primitives and empty objects', () => {
      assert.equal(isHarmlessCursorTool(null), false);
      assert.equal(isHarmlessCursorTool(undefined), false);
      assert.equal(isHarmlessCursorTool(''), false);
      assert.equal(isHarmlessCursorTool(123), false);
      assert.equal(isHarmlessCursorTool({}), false);
    });
  });

  describe('runCursorAgent integration with harmless and blocked tools', () => {
    it('updateTodosToolCall started -> not a miss and run succeeds with final result text', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { updateTodosToolCall: { args: { todos: ['step 1', 'step 2'] } } },
            }));
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'completed',
              tool_call: { updateTodosToolCall: {} },
            }));
            child.stdout.emit('data', line({
              type: 'result',
              subtype: 'success',
              is_error: false,
              result: 'completed all todos and solved task',
              usage: { inputTokens: 42, outputTokens: 84 },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('solve task', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, true);
        if (res.ok) {
          assert.equal(res.text, 'completed all todos and solved task');
          assert.equal(res.usage.inputTokens, 42);
          assert.equal(res.usage.outputTokens, 84);
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('getMcpToolsToolCall started -> not a miss and run succeeds with final result text', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { getMcpToolsToolCall: {} },
            }));
            child.stdout.emit('data', line({
              type: 'result',
              subtype: 'success',
              is_error: false,
              result: 'listed tools successfully',
              usage: { inputTokens: 10, outputTokens: 20 },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('list tools', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, true);
        if (res.ok) {
          assert.equal(res.text, 'listed tools successfully');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('shellToolCall -> still fails with tool attempt: shell', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { shellToolCall: { args: { command: 'rm -rf /' } } },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('prompt', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'tool attempt: shell');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('editToolCall -> still fails with tool attempt: edit', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { editToolCall: { args: { path: 'src/main.ts' } } },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('prompt', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'tool attempt: edit');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('mcpToolCall (call mcp tool) -> still fails with tool attempt: mcp', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { mcpToolCall: { args: { tool: 'call' } } },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('prompt', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'tool attempt: mcp');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('readToolCall of another file -> still fails with tool attempt: read', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { readToolCall: { args: { path: 'C:/secret/.env' } } },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('prompt', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'tool attempt: read');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('exact-match check: "update todos and run" -> still fails with tool attempt failure', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              name: 'update todos and run',
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('prompt', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'tool attempt: update todos and run');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });

    it('allowed tool started but child exits without result event -> fails with no result event', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const spawnFn = () =>
          fakeChild((child) => {
            child.stdout.emit('data', line({
              type: 'tool_call',
              subtype: 'started',
              tool_call: { updateTodosToolCall: {} },
            }));
            child.emit('close', 0);
          });

        const res = await runCursorAgent('prompt', { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);
        assert.equal(res.ok, false);
        if (!res.ok) {
          assert.equal(res.reason, 'no result event');
        }
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });
  });
});
