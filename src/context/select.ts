import { z } from 'zod';
import { JevSchemaError, parseJev, type JevClient } from './jev-client.js';

export const STAGE2_INSTRUCTIONS = 'Is the following project context item highly relevant to understanding, explaining, or answering this prompt?';

const JevNoulResponseSchema = z.looseObject({
  answers: z.record(
    z.string(),
    z.looseObject({
      noul: z.number().min(0).max(1),
    })
  ),
});

export interface Candidate {
  readonly id: string; // e.g. "c0", "c1"
  readonly type: 'memory' | 'graph';
  readonly path: string; // memory path or community id
  readonly title?: string; // memory title, or undefined for graph
  readonly content: string; // memory snippet, or node label/id for graph
  readonly isCurrent: boolean;
  readonly project: string;
  readonly extra?: {
    readonly communityId: string;
    readonly nodeId: string;
    readonly nodeLabel: string;
  };
}

export interface ScoredCandidate extends Candidate {
  readonly p: number; // Relevance probability from JEV
}

/**
 * Batches candidates into groups of up to 30 and scores them in parallel via JEV.
 */
export async function scoreCandidates(options: {
  readonly client: JevClient;
  readonly model: string;
  readonly prompt: string;
  readonly candidates: readonly Candidate[];
  readonly signal: AbortSignal;
}): Promise<readonly ScoredCandidate[]> {
  if (options.candidates.length === 0) return [];

  // Group candidates into batches of up to 30
  const batches: Candidate[][] = [];
  for (let i = 0; i < options.candidates.length; i += 30) {
    batches.push(options.candidates.slice(i, i + 30));
  }

  // Run all batch requests in parallel
  const scoredBatches = await Promise.all(
    batches.map(async (batch) => {
      const questions: Record<string, any> = {};
      for (const cand of batch) {
        let itemDesc = '';
        if (cand.type === 'memory') {
          itemDesc = `Memory wiki document: ${cand.title || cand.path} (path: ${cand.path})\nContent:\n${cand.content}`;
        } else {
          itemDesc = `Code element/file in project '${cand.project}' (Community ${cand.extra?.communityId}):\nName/Label: ${cand.content}`;
        }

        questions[cand.id] = {
          type: 'noul',
          instructions: `${STAGE2_INSTRUCTIONS}\nContext Item:\n${itemDesc}`,
        };
      }

      const requestBody = {
        model: options.model,
        state: `User prompt:\n${options.prompt}`,
        questions,
      };

      const rawPayload = await options.client.postSystemOne(requestBody, options.signal);
      const { answers } = parseJev(JevNoulResponseSchema, rawPayload);
      return batch.map((cand): ScoredCandidate => {
        const ans = answers[cand.id];
        if (!ans) throw new JevSchemaError(`schema: answers.${cand.id}: missing`);
        return { ...cand, p: ans.noul };
      });
    })
  );

  return scoredBatches.flat();
}
