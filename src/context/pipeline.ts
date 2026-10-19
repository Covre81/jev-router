import { createHash } from 'node:crypto';
import { existsSync, readFileSync } from 'node:fs';
import { z } from 'zod';
import { getMachineOrigin } from './origin.js';
import { searchMemory, type MemoryHit } from './memory-source.js';
import { getGraphSources, type GraphData, type GraphCommunity } from './graph-source.js';
import { isProjectDir } from './project-dir.js';
import { JevClient } from './jev-client.js';
import { rankCommunities, STAGE1_INSTRUCTIONS, type SelectedCommunity } from './relevance.js';
import { scoreCandidates, STAGE2_INSTRUCTIONS, type Candidate, type ScoredCandidate } from './select.js';
import { recordContextRun, type ContextCandidateInput, type ContextOutcome } from './log.js';
import { openTelemetryDb } from '../telemetry/db.js';
import { defaultTelemetryDbPath } from '../paths.js';

/** Cap the prompt sent to JEV System One so large pasted logs never trigger HTTP 400. */
export const MAX_JEV_PROMPT_CHARS = 4_000;

// Schema for Claude Code UserPromptSubmit stdin
const ClaudeStdinSchema = z.looseObject({
  prompt: z.string(),
  session_id: z.string(),
  cwd: z.string(),
  transcript_path: z.string().optional(),
});

// Schema for Antigravity / agy PreInvocation stdin
const AntigravityStdinSchema = z.looseObject({
  conversationId: z.string().min(1),
  workspacePaths: z.array(z.string()).optional(),
  transcriptPath: z.string().optional(),
  cwd: z.string().optional(),
  prompt: z.string().optional(),
});

export type ContextInput = z.infer<typeof ClaudeStdinSchema>;

/**
 * Reads the latest non-ephemeral step from an Antigravity/agy transcript.jsonl.
 * Returns the user prompt only when the latest turn is a fresh USER_INPUT step
 * (skipping tool-call continuation invocations).
 */
export function extractAntigravityPrompt(transcriptPath: string): string | undefined {
  try {
    if (!existsSync(transcriptPath)) return undefined;
    const raw = readFileSync(transcriptPath, 'utf8');
    const lines = raw.split(/\r?\n/);
    for (let i = lines.length - 1; i >= 0; i--) {
      const line = lines[i]?.trim();
      if (!line) continue;
      let step: any;
      try {
        step = JSON.parse(line);
      } catch {
        continue;
      }
      const type = step?.type;
      if (type === 'EPHEMERAL_MESSAGE' || type === 'CHECKPOINT') continue;
      if (type !== 'USER_INPUT') return undefined;
      const content = typeof step?.content === 'string' ? step.content : '';
      const match = /<USER_REQUEST>([\s\S]*?)<\/USER_REQUEST>/i.exec(content);
      const extracted = (match ? match[1]! : content).trim();
      return extracted.length > 0 ? extracted : undefined;
    }
    return undefined;
  } catch {
    return undefined;
  }
}

export interface PipelineConfig {
  readonly mode: 'inject' | 'shadow' | 'off';
  readonly lCurrent: number;
  readonly lOther: number;
  readonly maxItems: number;
  readonly maxChars: number;
  readonly timeoutMs: number;
  readonly graphRoot?: string | undefined;
  readonly aiMemoryBin: string;
  readonly apiKey?: string | undefined;
  readonly apiUrl: string;
  readonly model: string;
}

export interface PipelineResult {
  readonly outcome: ContextOutcome;
  readonly stdout: string;
  /** Why the outcome is not ok; never carries the key or the prompt. */
  readonly error?: string;
}

/** Diagnostic text kept short: it lands in stderr and in context_runs.error. */
const reasonOf = (err: unknown): string => (err instanceof Error ? err.message : String(err)).replace(/\s+/g, ' ').slice(0, 300);

export async function runPipeline(
  rawInput: unknown,
  config: PipelineConfig
): Promise<PipelineResult> {
  const startedAt = performance.now();
  // 1. Validate stdin (Claude Code UserPromptSubmit or Antigravity/agy PreInvocation)
  let input: ContextInput;
  let isAntigravity = false;
  let emptyStdout = '';

  const parsedClaude = ClaudeStdinSchema.safeParse(rawInput);
  if (parsedClaude.success) {
    input = parsedClaude.data;
  } else {
    const parsedAgy = AntigravityStdinSchema.safeParse(rawInput);
    if (!parsedAgy.success) {
      // Stdin inválido: sai sem gravar (as per instructions)
      throw new Error('invalid_stdin');
    }
    isAntigravity = true;
    emptyStdout = '{}';
    const cwd = parsedAgy.data.cwd ?? parsedAgy.data.workspacePaths?.[0] ?? '';
    const prompt =
      parsedAgy.data.prompt ??
      (parsedAgy.data.transcriptPath ? extractAntigravityPrompt(parsedAgy.data.transcriptPath) : undefined);
    if (!prompt || !cwd) {
      return { outcome: 'skipped_origin', stdout: emptyStdout };
    }
    input = {
      prompt,
      session_id: parsedAgy.data.conversationId,
      cwd,
      transcript_path: parsedAgy.data.transcriptPath,
    };
  }

  // Labels go stale when the questions or the knobs change: hash what JEV is actually asked.
  const pipelineVersionInput = JSON.stringify({
    lCurrent: config.lCurrent,
    lOther: config.lOther,
    maxItems: config.maxItems,
    maxChars: config.maxChars,
    stage1Question: STAGE1_INSTRUCTIONS,
    stage2Question: STAGE2_INSTRUCTIONS,
  });
  const pipelineVersion = createHash('sha256').update(pipelineVersionInput).digest('hex').slice(0, 12);

  /** One row per run; a database problem only costs the row, never the prompt. */
  const record = (
    outcome: ContextOutcome,
    error: string | undefined,
    candidates: readonly ContextCandidateInput[] = [],
    injectedChars = 0,
  ): void => {
    try {
      const dbPath = process.env['TELEMETRY_DB_PATH'] ?? defaultTelemetryDbPath();
      if (!existsSync(dirname(dbPath))) return;
      const db = openTelemetryDb(dbPath);
      try {
        recordContextRun(
          db,
          {
            sessionId: input.session_id,
            promptHash: createHash('sha256').update(input.prompt).digest('hex'),
            promptChars: input.prompt.length,
            mode: config.mode,
            outcome,
            pipelineVersion,
            injected: injectedChars > 0,
            injectedChars,
            latencyMs: Math.round(performance.now() - startedAt),
            error: error ?? null,
          },
          candidates,
        );
      } finally {
        db.close();
      }
    } catch (err) {
      process.stderr.write(`[context] SQLite failure: ${reasonOf(err)}\n`);
    }
  };

  // 2. Check for machine origin
  const machineOrigin = getMachineOrigin(input.prompt);
  if (machineOrigin) {
    record('skipped_origin', machineOrigin);
    return { outcome: 'skipped_origin', stdout: emptyStdout };
  }

  if (config.mode === 'off') {
    return { outcome: 'ok', stdout: emptyStdout };
  }

  if (!isProjectDir(input.cwd)) {
    record('skipped_cwd', 'cwd is not a project');
    return { outcome: 'skipped_cwd', stdout: emptyStdout };
  }

  // 3. Check for API key configuration
  if (!config.apiKey) {
    const error = 'TYPESAFE_API_KEY is not set';
    record('config_error', error);
    return { outcome: 'config_error', stdout: emptyStdout, error };
  }

  // Setup AbortController for global timeout
  const controller = new AbortController();
  let timedOut = false;
  const timeoutId = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, config.timeoutMs);

  let finalOutcome: 'ok' | 'partial' | 'source_error' | 'jev_error' | 'timeout' = 'ok';
  const errors: string[] = [];
  let memoryHits: readonly MemoryHit[] = [];
  let graphSources: readonly GraphData[] = [];
  let memoryFailed = false;
  let graphFailed = false;

  // 4. Fetch memory and graph sources in parallel
  try {
    const memoryPromise = (async () => {
      try {
        return await searchMemory({
          bin: config.aiMemoryBin,
          cwd: input.cwd,
          prompt: input.prompt,
          timeoutMs: config.timeoutMs,
          signal: controller.signal,
        });
      } catch (err) {
        memoryFailed = true;
        errors.push(`memory: ${reasonOf(err)}`);
        return [];
      }
    })();

    const graphPromise = (async () => {
      try {
        if (!config.graphRoot) return [];
        return getGraphSources(config.graphRoot, input.cwd);
      } catch (err) {
        graphFailed = true;
        errors.push(`graph: ${reasonOf(err)}`);
        return [];
      }
    })();

    const [memResult, graphResult] = await Promise.all([memoryPromise, graphPromise]);
    memoryHits = memResult;
    graphSources = graphResult;

    if (memoryFailed && graphFailed) {
      finalOutcome = 'source_error';
    } else if (memoryFailed || graphFailed) {
      finalOutcome = 'partial';
    }
  } catch (err: any) {
    if (err.name === 'AbortError' || timedOut) {
      finalOutcome = 'timeout';
    } else {
      finalOutcome = 'source_error';
      errors.push(`sources: ${reasonOf(err)}`);
    }
  }

  const allCandidates: ScoredCandidate[] = [];
  const evaluatedCandidates: ContextCandidateInput[] = [];

  if (finalOutcome !== 'timeout' && finalOutcome !== 'source_error') {
    const client = new JevClient({ apiUrl: config.apiUrl, apiKey: config.apiKey });
    const jevPrompt = input.prompt.slice(0, MAX_JEV_PROMPT_CHARS);

    try {
      // 5. JEV Stage 1 Choice (rank communities)
      let selectedCommunities: readonly SelectedCommunity[] = [];
      if (graphSources.length > 0) {
        selectedCommunities = await rankCommunities({
          client,
          model: config.model,
          prompt: jevPrompt,
          graphs: graphSources,
          signal: controller.signal,
        });
      }

      // 6. Build Candidate List
      const candidatesList: Candidate[] = [];
      let nextId = 0;

      // Add Memory Candidates
      for (const hit of memoryHits) {
        candidatesList.push({
          id: `c${nextId++}`,
          type: 'memory',
          path: hit.path,
          title: hit.title,
          content: hit.snippet,
          isCurrent: true, // Memory is always current project
          project: '', // Filled in later or omitted for memory
        });
      }

      // Add Graph Candidates (up to 10 nodes of top 2 selected communities of each graph)
      for (const sel of selectedCommunities) {
        const topNodes = sel.community.topNodes.slice(0, 10);
        for (const node of topNodes) {
          candidatesList.push({
            id: `c${nextId++}`,
            type: 'graph',
            path: sel.community.id,
            content: node.label,
            isCurrent: sel.graph.isCurrent,
            project: sel.graph.project,
            extra: {
              communityId: sel.community.id,
              nodeId: node.id,
              nodeLabel: node.label,
            },
          });
        }
      }

      // 7. JEV Stage 2 Nouls (score candidates)
      const scoredCandidates = await scoreCandidates({
        client,
        model: config.model,
        prompt: jevPrompt,
        candidates: candidatesList,
        signal: controller.signal,
      });

      allCandidates.push(...scoredCandidates);
    } catch (err: any) {
      if (err.name === 'AbortError' || timedOut) {
        finalOutcome = 'timeout';
      } else {
        // HTTP status + API reason (JevHttpError), schema path (JevSchemaError) or a network error.
        finalOutcome = 'jev_error';
        errors.push(`jev: ${reasonOf(err)}`);
      }
    }
  }

  clearTimeout(timeoutId);
  if (!controller.signal.aborted) {
    controller.abort();
  }

  // If timeout was triggered during execution, override outcome
  if (timedOut) {
    finalOutcome = 'timeout';
    if (!errors.some(e => e.startsWith('timeout after'))) {
      errors.push(`timeout after ${config.timeoutMs} ms`);
    }
  }
  const error = errors.length > 0 ? errors.join('; ') : undefined;

  // 8. Thresholding, Grouping, and Filtering
  const selectedMemoryItems: ScoredCandidate[] = [];
  const selectedGraphNodes: ScoredCandidate[] = [];

  for (const cand of allCandidates) {
    const threshold = cand.isCurrent ? config.lCurrent : config.lOther;
    const isSelected = cand.p >= threshold;

    if (isSelected) {
      if (cand.type === 'memory') {
        selectedMemoryItems.push(cand);
      } else {
        selectedGraphNodes.push(cand);
      }
    }
  }

  // Group graph nodes by project + community
  const groupedGraphItems: {
    project: string;
    communityId: string;
    labels: string[];
    maxP: number;
  }[] = [];

  for (const node of selectedGraphNodes) {
    const commId = node.extra?.communityId || '';
    const proj = node.project;
    let group = groupedGraphItems.find((g) => g.project === proj && g.communityId === commId);
    if (!group) {
      group = {
        project: proj,
        communityId: commId,
        labels: [],
        maxP: 0,
      };
      groupedGraphItems.push(group);
    }
    group.labels.push(node.content);
    if (node.p > group.maxP) {
      group.maxP = node.p;
    }
  }

  // Format final list of items with their scored P
  interface FormattedItem {
    readonly text: string;
    readonly p: number;
    readonly rawItems: readonly ScoredCandidate[];
  }

  const finalFormattedList: FormattedItem[] = [];

  // Memory items
  for (const mem of selectedMemoryItems) {
    finalFormattedList.push({
      text: `[memória] ${mem.title || mem.path} (${mem.path}): ${mem.content}`,
      p: mem.p,
      rawItems: [mem],
    });
  }

  // Grouped graph items
  for (const group of groupedGraphItems) {
    // Unique and sort labels
    const uniqueLabels = Array.from(new Set(group.labels)).join(', ');
    finalFormattedList.push({
      text: `[grafo] ${group.project} › ${group.communityId}: ${uniqueLabels}`,
      p: group.maxP,
      rawItems: selectedGraphNodes.filter(
        (n) => n.project === group.project && n.extra?.communityId === group.communityId
      ),
    });
  }

  // Sort final items by p descending
  finalFormattedList.sort((a, b) => b.p - a.p);

  // Apply maximum items and character limit
  const finalSelectedStrings: string[] = [];
  const injectedCandidateIds = new Set<string>();
  let currentChars = 0;

  for (const item of finalFormattedList) {
    if (finalSelectedStrings.length >= config.maxItems) break;
    const additionalLength = item.text.length + (finalSelectedStrings.length > 0 ? 1 : 0);
    if (currentChars + additionalLength > config.maxChars) break;

    finalSelectedStrings.push(item.text);
    currentChars += additionalLength;
    for (const raw of item.rawItems) {
      injectedCandidateIds.add(raw.id);
    }
  }

  const injectedContextStr = finalSelectedStrings.join('\n');
  const hasInjected = finalSelectedStrings.length > 0;

  // Build list of evaluated candidates for DB log
  for (const cand of allCandidates) {
    evaluatedCandidates.push({
      type: cand.type,
      path: cand.path,
      title: cand.title || null,
      content: cand.content,
      p: cand.p,
      injected: injectedCandidateIds.has(cand.id),
    });
  }

  // 9. Db Logging
  record(finalOutcome, error, evaluatedCandidates, hasInjected ? injectedContextStr.length : 0);

  // 10. Stdout Output
  let stdoutResult = emptyStdout;
  if (config.mode === 'inject' && hasInjected) {
    stdoutResult = isAntigravity
      ? JSON.stringify({
          injectSteps: [
            {
              ephemeralMessage: injectedContextStr,
            },
          ],
        })
      : JSON.stringify({
          hookSpecificOutput: {
            hookEventName: 'UserPromptSubmit',
            additionalContext: injectedContextStr,
          },
        });
  }

  return {
    outcome: finalOutcome,
    stdout: stdoutResult,
    ...(error ? { error } : {}),
  };
}

function dirname(path: string): string {
  return path.substring(0, Math.max(path.lastIndexOf('/'), path.lastIndexOf('\\')));
}
