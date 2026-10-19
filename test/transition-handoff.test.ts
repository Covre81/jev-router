import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import {
  extractWorkingDirFromSystem,
  summarizeRecentTranscript,
  buildTransitionHandoff,
  withTransitionHandoff,
} from '../src/context/transition-handoff.js';
import { renderGeminiPrompt } from '../src/providers/gemini/prompt.js';
import { toChatCompletion } from '../src/providers/openai/translate-request.js';
import { Router } from '../src/routing/router.js';
import { TtlLruStore } from '../src/routing/session-store.js';

describe('Model transition context handoff (TDD)', () => {
  it('extracts Working directory from <env> block in body.system', () => {
    const sys = [
      {
        type: 'text',
        text: 'Header\n<env>\nWorking directory: C:\\Dev\\JEV-APP-PC\nPlatform: win32\n</env>',
      },
    ];
    assert.equal(extractWorkingDirFromSystem(sys), 'C:\\Dev\\JEV-APP-PC');
    assert.equal(extractWorkingDirFromSystem('no env here'), undefined);
  });

  it('summarizes the last 4 turns and extracts touched files from tool_use blocks', () => {
    const messages: any[] = [
      { role: 'user', content: 'turno 1 muito antigo que deve ficar de fora' },
      { role: 'assistant', content: 'resposta 1 antiga' },
      { role: 'user', content: 'turno 2 explorar config' },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'Lendo src/config.ts' },
          { type: 'tool_use', id: 't1', name: 'Read', input: { file_path: 'src/config.ts' } },
        ],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 't1', content: 'GEMINI_TIER=on' }],
      },
      {
        role: 'assistant',
        content: [
          { type: 'text', text: 'Atualizando src/domain/policy.ts' },
          {
            type: 'tool_use',
            id: 't2',
            name: 'Edit',
            input: { file_path: 'src/domain/policy.ts', old_string: 'a', new_string: 'b' },
          },
        ],
      },
      {
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: 't2', content: 'File updated' }],
      },
      { role: 'assistant', content: 'Política atualizada com sucesso.' },
      { role: 'user', content: '<system-reminder>ignore</system-reminder>Agora revise a segurança da mudança' },
    ];

    const summary = summarizeRecentTranscript(messages, 4);
    assert.ok(!summary.text.includes('turno 1 muito antigo'));
    assert.ok(summary.filesTouched.includes('src/config.ts'));
    assert.ok(summary.filesTouched.includes('src/domain/policy.ts'));
    assert.ok(summary.text.includes('src/config.ts'));
    assert.ok(summary.text.includes('src/domain/policy.ts'));
    assert.ok(summary.text.includes('Política atualizada com sucesso.'));
  });

  it('builds <session_handoff> combining transcript summary + ai-memory and triggers background write', async () => {
    const dir = await mkdtemp(join(tmpdir(), 'jev-handoff-'));
    try {
      await writeFile(join(dir, '.ai-memory.toml'), 'workspace = "personal"\nproject = "jev-router"\n');

      const recordedCalls: { args: string[]; stdin?: string | undefined }[] = [];
      const fakeExec = async (_bin: string, args: string[], stdin?: string) => {
        recordedCalls.push({ args, stdin });
        if (args[0] === 'read-page') {
          return JSON.stringify({
            path: 'working/jev-session-handoff.md',
            title: 'JEV Session Handoff',
            body: 'Decisão anterior: manter Claude apenas para hardRisk >= 0.5.',
          });
        }
        if (args[0] === 'search') {
          return JSON.stringify([
            {
              path: 'architecture/routing.md',
              title: 'Escada de roteamento',
              snippet: 'Ordem: gpt-oss:20b-cloud -> gemma4:31b-cloud -> gemini-3.8-flash-high -> Claude.',
            },
          ]);
        }
        return '{}';
      };

      const body: any = {
        model: 'claude-opus-4-6',
        system: [{ type: 'text', text: `<env>\nWorking directory: ${dir}\n</env>` }],
        messages: [
          { role: 'user', content: 'Implemente suporte a tools no Gemini' },
          {
            role: 'assistant',
            content: [
              { type: 'text', text: 'Editei src/providers/gemini/provider.ts' },
              { type: 'tool_use', id: 'u1', name: 'Edit', input: { file_path: 'src/providers/gemini/provider.ts' } },
            ],
          },
          { role: 'user', content: [{ type: 'tool_result', tool_use_id: 'u1', content: 'ok' }] },
          { role: 'assistant', content: 'Concluído no provider.' },
          { role: 'user', content: 'Agora audite a segurança do token de proxy' },
        ],
      };

      const handoff = await buildTransitionHandoff({
        body,
        previousTier: 'gemini',
        currentTier: 'primary',
        aiMemoryBin: 'ai-memory',
        timeoutMs: 450,
        execFn: fakeExec,
      });

      assert.ok(handoff);
      assert.ok(handoff.startsWith('<session_handoff>'));
      assert.ok(handoff.endsWith('</session_handoff>'));
      assert.ok(handoff.includes('gemini -> primary'));
      assert.ok(handoff.includes('src/providers/gemini/provider.ts'));
      assert.ok(handoff.includes('Decisão anterior: manter Claude apenas para hardRisk >= 0.5.'));
      assert.ok(handoff.includes('Escada de roteamento'));
      assert.ok(handoff.length <= 5000);

      // Wait a tick for the non-blocking background write-page call
      await new Promise((r) => setTimeout(r, 20));
      const writeCall = recordedCalls.find((c) => c.args[0] === 'write-page');
      assert.ok(writeCall, 'expected background write-page call to ai-memory');
      assert.ok(writeCall.args.includes('working/jev-session-handoff.md'));
      assert.ok(writeCall.stdin?.includes('gemini -> primary'));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('contextual injection per provider: preserves Anthropic cache prefix, injects in Gemini preamble, and in OpenAI system', () => {
    const handoffBlock = '<session_handoff>\nModel transition: trivial -> gemini\n### Active Session State\n- Files touched: src/cli.ts\n</session_handoff>';
    const originalBody: any = {
      model: 'claude-opus-4-6',
      system: [
        { type: 'text', text: 'CORE SYSTEM PROMPT', cache_control: { type: 'ephemeral' } },
        { type: 'text', text: '<env>\nWorking directory: C:\\Dev\\JEV-APP-PC\n</env>' },
      ],
      messages: [{ role: 'user', content: 'continue' }],
    };

    const injectedBody = withTransitionHandoff(originalBody, handoffBlock);

    // 1. Anthropic: original cached system block is untouched at index 0; handoff is appended
    assert.deepEqual((injectedBody as any).system[0], {
      type: 'text',
      text: 'CORE SYSTEM PROMPT',
      cache_control: { type: 'ephemeral' },
    });
    assert.equal((injectedBody as any).system.at(-1).text, handoffBlock);

    // 2. Gemini: renderGeminiPrompt includes <session_handoff> in preamble but NOT 'CORE SYSTEM PROMPT'
    const geminiPrompt = renderGeminiPrompt(injectedBody, 200_000)!;
    assert.ok(!geminiPrompt.includes('CORE SYSTEM PROMPT'));
    assert.ok(geminiPrompt.includes('<session_handoff>'));
    assert.ok(geminiPrompt.includes('Model transition: trivial -> gemini'));

    // 3. OpenAI (Gemma / GPT-OSS): toChatCompletion places <session_handoff> in the initial system message
    const chatReq = toChatCompletion(injectedBody, { model: 'gemma4:31b-cloud', maxOutputTokens: 4096 });
    assert.equal(chatReq.messages[0]?.role, 'system');
    assert.ok(chatReq.messages[0]?.content.includes('<session_handoff>'));
  });

  it('Router detects one-shot model transitions (both escalating and de-escalating) and does not re-trigger on same-tier turns', async () => {
    const scripts = [
      // Turn 1: trivial
      { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.05 }, textOnly: 0.95 },
      // Turn 2: escalates to standard -> transition trivial -> standard!
      { simple: 0.2, standard: 0.65, structural: 0.15, risk: 0.8, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.8 }, textOnly: 0.1 },
      // Turn 3: stays on standard -> NO transition!
      { simple: 0.25, standard: 0.65, structural: 0.1, risk: 0.8, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.8 }, textOnly: 0.1 },
      // Turn 4: escalates to gemini -> transition standard -> gemini!
      { simple: 0.05, standard: 0.1, structural: 0.85, risk: 0.9, riskScores: { security_sensitive: 0.05, destructive_or_production: 0.05, requires_inspection: 0.9 }, textOnly: 0.05 },
      // Turn 5: de-escalates back to standard -> transition gemini -> standard!
      { simple: 0.95, standard: 0.03, structural: 0.02, risk: 0.05, riskScores: { security_sensitive: 0.01, destructive_or_production: 0.01, requires_inspection: 0.05 }, textOnly: 0.9 },
      // Turn 6: escalates to primary (Claude) due to hardRisk -> transition standard -> primary!
      { simple: 0.05, standard: 0.1, structural: 0.85, risk: 0.95, riskScores: { security_sensitive: 0.95, destructive_or_production: 0.1, requires_inspection: 0.9 }, textOnly: 0.05 },
    ];

    const classifier = { name: 'test', classify: async () => scripts.shift()! };
    const store = new TtlLruStore<any>(100, 60_000);
    const router = new Router(classifier as any, store, {
      policy: { minCheapProbability: 0.9, standardRoute: 'primary', standardEnabled: true, minStandardProbability: 0.75 },
      primaryClasses: new Set(['auxiliary']),
      cheapContextTokens: 100_000,
      standardContextTokens: 250_000,
      geminiPolicy: { enabled: true, minTextOnly: 0, pressureMinTextOnly: 0 },
      geminiFromPrimary: true,
      classifierMaxChars: 4000,
      classifierTimeoutMs: 1000,
      quota: { level: () => 'none' },
    });

    const mkReq = (messages: any[]) => ({
      body: { model: 'claude-opus-4-6', messages } as any,
      rawByteLength: 100,
      sessionId: 's-trans',
      agentId: undefined,
      requestClass: undefined,
      contextCompacted: false,
    });

    // Turn 1 (fresh): trivial -> no transition
    const m1 = [{ role: 'user', content: 'ola' }];
    const d1 = await router.decide(mkReq(m1));
    assert.equal(d1.tier, 'trivial');
    assert.equal(d1.modelTransition, undefined);

    // Turn 2: trivial -> standard (transition!)
    const m2 = [...m1, { role: 'assistant', content: 'oi' }, { role: 'user', content: 'leia o arquivo x' }];
    const d2 = await router.decide(mkReq(m2));
    assert.equal(d2.tier, 'standard');
    assert.equal(d2.modelTransition, true);
    assert.equal(d2.previousTier, 'trivial');

    // Turn 3: standard -> standard (same model, NO transition!)
    const m3 = [...m2, { role: 'assistant', content: 'lido' }, { role: 'user', content: 'leia o arquivo y' }];
    const d3 = await router.decide(mkReq(m3));
    assert.equal(d3.tier, 'standard');
    assert.equal(d3.modelTransition, undefined);

    // Turn 4: standard -> gemini (transition!)
    const m4 = [...m3, { role: 'assistant', content: 'lido y' }, { role: 'user', content: 'refatore toda a arquitetura' }];
    const d4 = await router.decide(mkReq(m4));
    assert.equal(d4.tier, 'gemini');
    assert.equal(d4.modelTransition, true);
    assert.equal(d4.previousTier, 'standard');

    // Turn 4b: tool_result continuation on gemini (same model, NO transition!)
    const m4Tool = [
      ...m4,
      { role: 'assistant', content: [{ type: 'tool_use', id: '1', name: 'Read', input: {} }] },
      { role: 'user', content: [{ type: 'tool_result', tool_use_id: '1', content: 'ok' }] },
    ];
    const d4Tool = await router.decide(mkReq(m4Tool));
    assert.equal(d4Tool.tier, 'gemini');
    assert.equal(d4Tool.modelTransition, undefined);

    // Turn 5: gemini -> standard (de-escalation transition!)
    const m5 = [...m4Tool, { role: 'assistant', content: 'pronto' }, { role: 'user', content: 'obrigado, entendi' }];
    const d5 = await router.decide(mkReq(m5));
    assert.equal(d5.tier, 'standard');
    assert.equal(d5.modelTransition, true);
    assert.equal(d5.previousTier, 'gemini');

    // Turn 6: standard -> primary (escalation to Claude at the last moment -> transition!)
    const m6 = [...m5, { role: 'assistant', content: 'de nada' }, { role: 'user', content: 'rotacione as chaves de producao e altere o IAM' }];
    const d6 = await router.decide(mkReq(m6));
    assert.equal(d6.tier, 'primary');
    assert.equal(d6.modelTransition, true);
    assert.equal(d6.previousTier, 'standard');
  });
});
