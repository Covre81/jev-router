import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { openTelemetryDb } from '../src/telemetry/db.js';
import { computeCacheMisses, computeStats, renderStats } from '../src/telemetry/stats.js';
import { pricingFromEnv } from '../src/telemetry/pricing.js';
import { meterAnthropicBody } from '../src/telemetry/usage-meter.js';
import { isAcknowledgement } from '../src/telemetry/audit.js';
import { needsReadFirstHint, withReadFirstHint } from '../src/routing/read-first-hint.js';
import { MessagesBody } from '../src/routing/messages-body.js';
import { Readable } from 'node:stream';

describe('Stats and Hint improvements', () => {
  it('1) cache-miss causes', () => {
    const db = openTelemetryDb(':memory:');
    const pricing = pricingFromEnv({});
    
    db.exec(`
      INSERT INTO router_logs (final_provider, route_reason, max_tokens, tokens_in, cache_write_tokens, created_at, session_id, request_class, requested_model, system_hash, outcome, latency_ms) VALUES 
      ('anthropic', 'reason', 100, 200000, 200000, 1000000, 's1', 'main', 'claude-sonnet-5', 'sys1', 'ok', 100), -- 1: first Anthropic main row -> first-turn-of-session
      ('openai', 'reason', 100, 200000, 0, 1001000, 's1', 'main', 'claude-sonnet-5', 'sys1', 'ok', 100), -- cheap main row
      ('anthropic', 'reason', 100, 200000, 200000, 1002000, 's1', 'main', 'claude-sonnet-5', 'sys1', 'ok', 100), -- 2: after cheap main -> return-from-cheap
      ('anthropic', 'failover:cheap-unavailable', 100, 200000, 200000, 1003000, 's1', 'main', 'claude-sonnet-5', 'sys1', 'ok', 100), -- 3: failover -> failover-from-cheap
      ('anthropic', 'reason', 100, 200000, 200000, 1004000, 's1', 'auxiliary', 'claude-opus-5-5', 'sys1', 'ok', 100), -- 4: first auxiliary
      ('anthropic', 'reason', 100, 200000, 200000, 1005000, 's1', 'auxiliary', 'claude-sonnet-5', 'sys1', 'ok', 100), -- 5: auxiliary model-switch -> model-switch
      ('anthropic', 'reason', 1, 200000, 200000, 1006000, 's2', 'main', 'claude-sonnet-5', 'sys1', 'ok', 100), -- probe row, ignored
      ('anthropic', 'reason', 100, 200000, 200000, 1007000, 's3', 'auxiliary', 'claude-opus-5-5', 'sys1', 'ok', 100), -- title call
      ('anthropic', 'reason', 100, 200000, 200000, 1008000, 's3', 'main', 'claude-sonnet-5', 'sys2', 'ok', 100), -- 6: main row after auxiliary -> first-turn-of-session (not system-changed)
      ('anthropic', 'reason', 100, 200000, 200000, 1009000, 's3', 'main', 'claude-sonnet-5', 'sys3', 'ok', 100) -- 7: system-changed
    `);

    const stats = computeCacheMisses(db, pricing, { minWrite: 150000 });
    const causes = stats.misses.map((m: any) => m.cause);
    
    assert.deepEqual(causes, [
      'first-turn-of-session', 
      'return-from-cheap',
      'failover-from-cheap',
      'first-turn-of-session',
      'model-switch',
      'first-turn-of-session',
      'first-turn-of-session',
      'system-changed'
    ]);
    db.close();
  });

  it('isAcknowledgement rules', () => {
    assert.equal(isAcknowledgement('ok'), true);
    assert.equal(isAcknowledgement('obrigado!'), true);
    assert.equal(isAcknowledgement('ficou top'), true);
    assert.equal(isAcknowledgement('valeu, ficou top'), true);
    assert.equal(isAcknowledgement('Olha, as mudancas que fiz com o grok bot ficaram top'), true);
    assert.equal(isAcknowledgement('show!'), true);
    assert.equal(isAcknowledgement('ok, now fix the bug in src/a.ts'), false);
    assert.equal(isAcknowledgement('what next?'), false);
    assert.equal(isAcknowledgement('show me the status of the build'), false);
    assert.equal(isAcknowledgement('stop the server'), false);
    assert.equal(isAcknowledgement('token count looks ok but read README.md'), false);
    assert.equal(isAcknowledgement('book'), false);
  });

  it('3) endsWithQuestion', async () => {
    const collect = async (r: Readable) => Buffer.concat(await r.toArray());
    
    // JSON body ending with ?
    const jsonBody = Buffer.from(JSON.stringify({ content: [{ type: 'text', text: 'what next?' }] }));
    const mJson = meterAnthropicBody(Readable.from([jsonBody]), 'application/json', undefined);
    await collect(mJson.body);
    await mJson.settled;
    assert.equal(mJson.endsWithQuestion(), true);

    // JSON body ending with period
    const jsonBody2 = Buffer.from(JSON.stringify({ content: [{ type: 'text', text: 'done.' }] }));
    const mJson2 = meterAnthropicBody(Readable.from([jsonBody2]), 'application/json', undefined);
    await collect(mJson2.body);
    await mJson2.settled;
    assert.equal(mJson2.endsWithQuestion(), false);

    // SSE body ending with ?
    const sse = Buffer.from(
      [
        'event: message_start',
        'data: {"type":"message_start"}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"what "}}',
        '',
        'event: content_block_delta',
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"next?"}}',
        '',
        'event: message_stop',
        'data: {"type":"message_stop"}',
        '',
        '',
      ].join('\r\n'),
    );
    const mSse = meterAnthropicBody(Readable.from([sse]), 'text/event-stream', undefined);
    await collect(mSse.body);
    await mSse.settled;
    assert.equal(mSse.endsWithQuestion(), true);
  });

  it('4) quota labels', () => {
    const db = openTelemetryDb(':memory:');
    const pricing = pricingFromEnv({});
    
    db.exec(`
      INSERT INTO router_logs (final_provider, outcome, route_reason, max_tokens, http_status, created_at, latency_ms) VALUES 
      ('anthropic', 'ok', 'failover:primary-rate-limited', 1, 429, 1000, 10),
      ('anthropic', 'ok', 'failover:primary-rate-limited', 64000, 429, 2000, 10),
      ('anthropic', 'ok', 'reason', 1, 529, 3000, 10)
    `);

    const stats = computeStats(db, { pricing });
    assert.equal(stats.quotaRouted, 1); // only the max_tokens 64000 one
    assert.equal(stats.probeRateLimited, 2); // the two max_tokens=1 probes
    
    const rendered = renderStats(stats);
    assert.ok(rendered.includes('Rate-limited session probes'));
    
    db.close();
  });

  it('5) needsReadFirstHint and withReadFirstHint', () => {
    const freshBody = (text: string, tools: boolean = true) => MessagesBody.parse({
      model: 'claude-opus-5-5',
      messages: [{ role: 'user', content: [{ type: 'text', text }] }],
      tools: tools ? [{ name: 'Read', input_schema: { type: 'object', properties: {} } }] : undefined
    });

    const secondTurnBody = MessagesBody.parse({
      model: 'claude-opus-5-5',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'hi' }] },
        { role: 'assistant', content: [{ type: 'text', text: 'hello' }] },
        { role: 'user', content: [{ type: 'text', text: 'o que falta fazer?' }] }
      ],
      tools: [{ name: 'Read', input_schema: { type: 'object', properties: {} } }]
    });

    const continuationBody = MessagesBody.parse({
      model: 'claude-opus-5-5',
      messages: [
        { role: 'user', content: [{ type: 'text', text: 'hi' }] },
        { role: 'assistant', content: [{ type: 'tool_use', id: 't1', name: 'Read', input: {} }] },
        { role: 'user', content: [{ type: 'tool_result', tool_use_id: 't1', content: 'ok' }] }
      ],
      tools: [{ name: 'Read', input_schema: { type: 'object', properties: {} } }]
    });

    assert.equal(needsReadFirstHint(freshBody('estou pensando no que fazer')), true);
    assert.equal(needsReadFirstHint(freshBody('o que falta fazer?')), true);
    assert.equal(needsReadFirstHint(freshBody('What should we do next?')), true);
    assert.equal(needsReadFirstHint(freshBody('Onde paramos?')), true);
    assert.equal(needsReadFirstHint(freshBody('fix the typo in the README')), false);
    assert.equal(needsReadFirstHint(freshBody('explain this function')), false);
    assert.equal(needsReadFirstHint(freshBody('corrija todos os testes')), false);
    assert.equal(needsReadFirstHint(freshBody('the planet')), false);
    assert.equal(needsReadFirstHint(freshBody('estou penso emo que fazer.')), true);
    
    assert.equal(needsReadFirstHint(secondTurnBody), false);
    assert.equal(needsReadFirstHint(freshBody('Onde paramos?', false)), false); // no file tools
    assert.equal(needsReadFirstHint(continuationBody), false);

    const bStringSystem = MessagesBody.parse({ model: 'x', messages: [{ role: 'user', content: 'hi' }], system: 'A' });
    const cString = structuredClone(bStringSystem);
    const resultString = withReadFirstHint(bStringSystem);
    assert.deepEqual(bStringSystem, cString);
    assert.ok(Array.isArray(resultString.messages[0]!.content));
    assert.ok((resultString.messages[0]!.content as any)[1].text.includes('<system-reminder>'));
    
    const bArraySystem = MessagesBody.parse({ model: 'x', messages: [{ role: 'user', content: 'hi' }], system: [{ type: 'text', text: 'A' }] });
    const cArray = structuredClone(bArraySystem);
    const resultArray = withReadFirstHint(bArraySystem);
    assert.deepEqual(bArraySystem, cArray);
    assert.ok(Array.isArray(resultArray.messages[0]!.content));
    assert.ok((resultArray.messages[0]!.content as any)[1].text.includes('<system-reminder>'));
    
    const bNoSystem = MessagesBody.parse({ model: 'x', messages: [{ role: 'user', content: 'hi' }] });
    const cNoSystem = structuredClone(bNoSystem);
    const resultNoSystem = withReadFirstHint(bNoSystem);
    assert.deepEqual(bNoSystem, cNoSystem);
    assert.ok(Array.isArray(resultNoSystem.messages[0]!.content));
    assert.ok((resultNoSystem.messages[0]!.content as any)[1].text.includes('<system-reminder>'));
  });
});
