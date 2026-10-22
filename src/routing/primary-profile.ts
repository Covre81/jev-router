import type { PrimaryProfile } from '../domain/policy.js';

const DEFAULT_MODEL = 'claude-opus-5-5';
const DEFAULT_EFFORT = 'medium';

/**
 * Rewrites model and effort of a primary request, or returns undefined to
 * forward the original bytes. Only the client default (Opus 5.5 at medium, not
 * fast mode) is touched: anything else is an explicit /model, /effort or /fast.
 * Opus 5.5 and Sonnet 5.5 share the 128K output cap, so max_tokens carries over.
 */
export function withPrimaryProfile(raw: Buffer, profile: PrimaryProfile): Buffer | undefined {
  let body: Record<string, unknown>;
  try {
    body = JSON.parse(raw.toString('utf8')) as Record<string, unknown>;
  } catch {
    return undefined;
  }
  if (typeof body !== 'object' || body === null) return undefined;
  const config = body['output_config'] as { effort?: unknown } | undefined;
  if (body['model'] !== DEFAULT_MODEL || config?.effort !== DEFAULT_EFFORT || body['speed'] !== undefined) return undefined;
  if (profile.model === DEFAULT_MODEL && profile.effort === DEFAULT_EFFORT) return undefined;
  body['model'] = profile.model;
  body['output_config'] = { ...config, effort: profile.effort };
  return Buffer.from(JSON.stringify(body));
}
