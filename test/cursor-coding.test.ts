import assert from 'node:assert/strict';
import { EventEmitter } from 'node:events';
import { existsSync, readFileSync } from 'node:fs';
import { readFile, rm, mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { describe, it } from 'node:test';
import { loadConfig } from '../src/config.js';
import {
  CURSOR_CODING_MAX_TEXT_ONLY,
  CURSOR_CODING_MIN_MASS,
  isCodingTurn,
} from '../src/domain/policy.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';
import { runCursorAgent } from '../src/providers/cursor/agent-runner.js';
import { CursorCliProvider } from '../src/providers/cursor/provider.js';

function fakeChild(onSetup?: (child: any) => void) {
  const child = new EventEmitter() as any;
  child.pid = 99999;
  child.stdin = new EventEmitter();
  child.stdin.write = () => true;
  child.stdin.end = () => {};
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  if (onSetup) setTimeout(() => onSetup(child), 5);
  return child;
}

async function fakeCursorRoot() {
  const root = join(tmpdir(), `cursor-test-coding-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  const ver = join(root, 'versions', '2026.04.01-abc');
  await mkdir(ver, { recursive: true });
  await writeFile(join(ver, 'node.exe'), 'FAKE');
  await writeFile(join(ver, 'index.js'), 'FAKE');
  return { root, ver };
}

describe('Cursor coding inside Argon tier', () => {
  describe('isCodingTurn table', () => {
    it('returns true when !hardRiskVeto, textOnly < 0.5, and standard+structural >= 0.5', () => {
      const d = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.3, // mass = 0.6 >= 0.5
        textOnly: 0.1,   // < 0.5
      };
      assert.equal(isCodingTurn(d), true);
    });

    it('returns false when textOnly >= 0.5', () => {
      const dBoundary = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.3,
        textOnly: CURSOR_CODING_MAX_TEXT_ONLY, // 0.5
      };
      assert.equal(isCodingTurn(dBoundary), false);

      const dHigh = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.3,
        textOnly: 0.8,
      };
      assert.equal(isCodingTurn(dHigh), false);
    });

    it('returns false when standard + structural mass < 0.5', () => {
      const d = {
        simple: 0.6,
        standard: 0.2,
        structural: 0.2, // mass = 0.4 < 0.5
        textOnly: 0.1,
      };
      assert.equal(isCodingTurn(d), false);
    });

    it('returns false on hard risk veto (security_sensitive >= 0.5)', () => {
      const d = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.3,
        textOnly: 0.1,
        riskScores: {
          security_sensitive: 0.6,
          destructive_or_production: 0.1,
          requires_inspection: 0.2,
        },
      };
      assert.equal(isCodingTurn(d), false);
    });

    it('returns false on hard risk veto (destructive_or_production >= 0.5)', () => {
      const d = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.3,
        textOnly: 0.1,
        riskScores: {
          security_sensitive: 0.1,
          destructive_or_production: 0.8,
          requires_inspection: 0.2,
        },
      };
      assert.equal(isCodingTurn(d), false);
    });

    it('returns false on legacy risk >= 0.5 without riskScores', () => {
      const d = {
        simple: 0.1,
        standard: 0.3,
        structural: 0.3,
        textOnly: 0.1,
        risk: 0.7,
      };
      assert.equal(isCodingTurn(d), false);
    });
  });

  describe('routing entering Argon tier', () => {
    function setupRouter(classification: any, options: any = {}) {
      const classifier = {
        name: 'test',
        classify: async () => classification,
      };
      const store = new TtlLruStore<any>(100, 60_000);
      const router = new Router(classifier as any, store, {
        policy: { minCheapProbability: 0.9, standardEnabled: false, argonEnabled: true },
        primaryClasses: new Set(['auxiliary']),
        cheapContextTokens: 100_000,
        classifierMaxChars: 4000,
        classifierTimeoutMs: 1000,
        cursorEnabled: options.cursorEnabled ?? true,
        cursorCoding: options.cursorCoding ?? true,
        argonPolicy: {
          enabled: true,
          modelLow: 'gemini-4-argon-low',
          modelMedium: 'gemini-4-argon-medium',
          modelHigh: 'gemini-4-argon-high',
        },
        ...options,
      } as any);
      return { router, store };
    }

    const req = (content = 'write code for function foo') => ({
      body: { messages: [{ role: 'user', content }] },
      rawByteLength: 100,
      sessionId: 's-coding',
      agentId: undefined,
      requestClass: undefined,
      contextCompacted: false,
    } as any);

    it('coding turn entering Argon tier selects Cursor first', async () => {
      const codingDist = {
        simple: 0.1,
        standard: 0.4,
        structural: 0.5,
        textOnly: 0.1,
        risk: 0.1,
      };
      const { router } = setupRouter(codingDist);
      const dec = await router.decide(req());
      assert.equal(dec.route, 'cursor');
      assert.equal(dec.tier, 'cursor');
      assert.equal(dec.reason, 'cursor:coding');
    });

    it('non-coding turn entering Argon tier routes to Argon directly', async () => {
      const nonCodingDist = {
        simple: 0.1,
        standard: 0.4,
        structural: 0.5,
        textOnly: 0.8, // high textOnly -> not coding
        risk: 0.1,
      };
      const { router } = setupRouter(nonCodingDist);
      const dec = await router.decide(req('explain the architecture'));
      assert.equal(dec.route, 'argon');
      assert.equal(dec.tier, 'argon');
      assert.equal(dec.reason, 'argon:high'); // structural 0.5 >= 0.45 -> high
    });

    it('coding turn routes to Argon when cursorCoding is disabled', async () => {
      const codingDist = {
        simple: 0.1,
        standard: 0.4,
        structural: 0.5,
        textOnly: 0.1,
        risk: 0.1,
      };
      const { router } = setupRouter(codingDist, { cursorCoding: false });
      const dec = await router.decide(req());
      assert.equal(dec.route, 'argon');
      assert.equal(dec.tier, 'argon');
      assert.equal(dec.reason, 'argon:high');
    });

    it('coding turn routes to Argon when cursor is not enabled', async () => {
      const codingDist = {
        simple: 0.1,
        standard: 0.4,
        structural: 0.5,
        textOnly: 0.1,
        risk: 0.1,
      };
      const { router } = setupRouter(codingDist, { cursorEnabled: false });
      const dec = await router.decide(req());
      assert.equal(dec.route, 'argon');
      assert.equal(dec.tier, 'argon');
      assert.equal(dec.reason, 'argon:high');
    });
  });

  describe('temp-file delivery and cleanup', () => {
    it('writes prompt to prompt.md in fresh temp dir, passes --workspace and removes dir on exit', async () => {
      const { root } = await fakeCursorRoot();
      let capturedArgs: string[] = [];
      let capturedWorkspaceDir: string | undefined;
      let promptFileContent: string | undefined;

      try {
        const spawnFn = (bin: string, args: string[]) => {
          capturedArgs = args;
          const wsIdx = args.indexOf('--workspace');
          if (wsIdx !== -1) {
            capturedWorkspaceDir = args[wsIdx + 1];
            // Synchronously inspect before child exits
            if (capturedWorkspaceDir && existsSync(join(capturedWorkspaceDir, 'prompt.md'))) {
              promptFileContent = readFileSync(
                join(capturedWorkspaceDir, 'prompt.md'),
                'utf8',
              );
            }
          }
          return fakeChild((child) => {
            const line = JSON.stringify({
              type: 'result',
              subtype: 'success',
              is_error: false,
              result: 'coded successfully',
              usage: { inputTokens: 50, outputTokens: 25 },
            }) + '\n';
            child.stdout.emit('data', Buffer.from(line));
            child.emit('close', 0);
          });
        };

        const testPrompt = 'Write a fibonacci function in TypeScript with memoization.';
        const res = await runCursorAgent(testPrompt, { bin: root, model: 'auto', timeoutMs: 2000 }, spawnFn as any);

        assert.equal(res.ok, true);
        if (res.ok) {
          assert.equal(res.text, 'coded successfully');
        }

        // Verify argv delivery format
        assert.ok(capturedArgs.includes('--workspace'));
        assert.ok(capturedWorkspaceDir);
        assert.equal(
          capturedArgs.at(-1),
          'Read the file prompt.md in the workspace and follow it exactly. Output only your answer.',
        );

        // Verify prompt was written to prompt.md
        assert.equal(promptFileContent, testPrompt);

        // Verify workspace directory was cleaned up in finally block
        assert.equal(existsSync(capturedWorkspaceDir!), false);
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });
  });

  describe('prompt length cap', () => {
    it('default CURSOR_TIER_MAX_PROMPT_CHARS is 200000 in config', () => {
      const c = loadConfig({
        CLASSIFIER: 'heuristic',
        CURSOR_TIER: 'on',
        LOCALAPPDATA: 'C:\\Users\\x\\AppData\\Local',
      });
      assert.ok(c.cursor);
      assert.equal(c.cursor.maxPromptChars, 200_000);
    });

    it('CursorCliProvider rejects transcripts exceeding maxPromptChars with transcript budget exceeded', async () => {
      const { root } = await fakeCursorRoot();
      try {
        const p = new CursorCliProvider({
          bin: root,
          model: 'auto',
          timeoutMs: 1000,
          maxPromptChars: 200_000,
        } as any);

        const hugeMessage = 'a'.repeat(200_005);
        const res = await p.send({
          body: {
            model: 'claude-3-5-sonnet',
            messages: [{ role: 'user', content: hugeMessage }],
          },
        } as any);

        assert.deepEqual(res, {
          kind: 'unavailable',
          reason: 'transcript budget exceeded',
        });
      } finally {
        await rm(root, { recursive: true, force: true });
      }
    });
  });
});

describe('Argon tier vs legacy Gemini text-only gate (quota pressure)', () => {
  const medium = { simple: 0.3, standard: 0.5, structural: 0.2, textOnly: 0.9, risk: 0.1 };
  const coding = { simple: 0.2, standard: 0.6, structural: 0.2, textOnly: 0.1, risk: 0.1 };
  const hardRisk = {
    ...medium,
    riskScores: { security_sensitive: 0.9, destructive_or_production: 0.1, requires_inspection: 0.1 },
  };

  function build(dist: any, argon: boolean) {
    const classifier = { name: 'test', classify: async () => dist };
    return new Router(classifier as any, new TtlLruStore<any>(100, 60_000), {
      // Mirrors serve.ts: Argon owns the middle band, so the standard-mass free band is off.
      policy: { minCheapProbability: 0.9, minStandardProbability: 0.75, standardEnabled: !argon, argonEnabled: argon },
      primaryClasses: new Set(['auxiliary']),
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota: { level: () => 'pressure' },
      pressurePolicy: { minCheapProbability: 0.9, minStandardProbability: 0.6 },
      geminiPolicy: { enabled: true, minTextOnly: 0.8, pressureMinTextOnly: 0.6, replacesCheap: true },
      geminiFromPrimary: false,
      cursorEnabled: true,
      cursorCoding: true,
      argonPolicy: { enabled: argon, modelLow: 'l', modelMedium: 'm', modelHigh: 'h' },
    } as any);
  }

  const req = {
    body: { messages: [{ role: 'user', content: 'x' }] },
    rawByteLength: 100,
    sessionId: 's-pressure',
    agentId: undefined,
    requestClass: undefined,
    contextCompacted: false,
  } as any;

  it('medium text-only turn under pressure -> argon:medium, not Gemini', async () => {
    const dec = await build(medium, true).decide(req);
    assert.equal(dec.route, 'argon');
    assert.equal(dec.reason, 'argon:medium');
    assert.equal(dec.argonProfile, 'medium');
  });

  it('coding turn under pressure -> cursor:coding, not Gemini', async () => {
    const dec = await build(coding, true).decide(req);
    assert.equal(dec.route, 'cursor');
    assert.equal(dec.reason, 'cursor:coding');
  });

  it('hard risk under pressure -> primary', async () => {
    const dec = await build(hardRisk, true).decide(req);
    assert.equal(dec.route, 'primary');
  });

  it('simple turn still goes to Flash', async () => {
    const dec = await build({ simple: 0.95, standard: 0.04, structural: 0.01, textOnly: 0.9, risk: 0 }, true).decide(req);
    assert.equal(dec.route, 'free');
  });

  it('ARGON_TIER off keeps the legacy gemini:text-only gate', async () => {
    const dec = await build({ simple: 0.1, standard: 0.4, structural: 0.5, textOnly: 0.9, risk: 0.1 }, false).decide(req);
    assert.equal(dec.route, 'free');
    assert.equal(dec.reason, 'gemini:text-only');
  });
});
