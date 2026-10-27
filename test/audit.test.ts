import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { auditFailure, jevDecisionOf, type ExchangeContext } from '../src/telemetry/audit.js';
import type { NewRouterLog } from '../src/telemetry/schema.js';

describe('jevDecisionOf', () => {
  it('keeps textOnly and each risk Noul, not just their max', () => {
    const riskScores = { security_sensitive: 0.1, destructive_or_production: 0.2, requires_inspection: 0.4 };
    const d = jevDecisionOf({
      route: 'cheap', reason: 'classified', conversationKey: 'k',
      distribution: { simple: 1, standard: 0, structural: 0, risk: 0.4, riskScores, textOnly: 0.7 },
    });
    assert.deepEqual(d?.riskScores, riskScores);
    assert.equal(d?.textOnly, 0.7);
    assert.equal(jevDecisionOf({ route: 'cheap', reason: 'classified', conversationKey: 'k', distribution: { simple: 1, standard: 0, structural: 0 } })?.textOnly, null);
  });
});

describe('auditFailure', () => {
  it('keeps the cause of a 502 and pins it on the attempt that failed', () => {
    const rows: NewRouterLog[] = [];
    const sink = { record: (r: NewRouterLog) => void rows.push(r), close: async () => {} };
    const ctx: ExchangeContext = {
      startedAt: performance.now(), startedAtMs: Date.now(), humanText: 'x', requestClass: 'main',
      model: 'm', requestedModel: 'm', toolsOffered: 0, upstream: { failure: 'retried: network' },
    };
    auditFailure(sink, { route: 'cheap', reason: 'classified', conversationKey: 'k' }, ctx, 'network: ECONNREFUSED');
    // Failed over to Claude, which failed too: the cheap retry's record is not this attempt's cause.
    auditFailure(sink, { route: 'primary', reason: 'failover:gemini-unavailable', conversationKey: 'k' }, ctx, 'network: getaddrinfo ENOTFOUND');
    auditFailure(sink, { route: 'primary', reason: 'sticky', conversationKey: 'k' }, { ...ctx, upstream: undefined }, `HTTP 500: ${'x'.repeat(500)}`);
    assert.deepEqual(rows.map((r) => r.upstreamFailure?.slice(0, 32)), ['retried: network', 'network: getaddrinfo ENOTFOUND', 'HTTP 500: ' + 'x'.repeat(22)]);
    assert.equal(rows[2]!.upstreamFailure!.length, 200);
  });
});
