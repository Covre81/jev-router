import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadConfig, removedEnvSet } from '../src/config.js';

const base = { CLASSIFIER: 'heuristic' };

describe('loadConfig security guards', () => {
  it('requires ANTHROPIC_API_KEY in inject mode', () => {
    assert.throws(() => loadConfig({ ...base, UPSTREAM_AUTH_MODE: 'inject' }), /ANTHROPIC_API_KEY/);
  });

  it('refuses a key-injecting proxy off loopback without PROXY_AUTH_TOKEN', () => {
    const inject = { ...base, UPSTREAM_AUTH_MODE: 'inject', ANTHROPIC_API_KEY: 'sk-ant-test' };
    assert.throws(() => loadConfig({ ...inject, HOST: '0.0.0.0' }), /PROXY_AUTH_TOKEN/);
    assert.equal(loadConfig({ ...inject, HOST: '0.0.0.0', PROXY_AUTH_TOKEN: 'proxy-token-0123456789' }).host, '0.0.0.0');
    assert.equal(loadConfig(inject).host, '127.0.0.1', 'loopback needs no token');
  });

  it('rejects a proxy token shorter than 16 characters', () => {
    assert.throws(() => loadConfig({ ...base, PROXY_AUTH_TOKEN: 'short' }), /PROXY_AUTH_TOKEN/);
  });
});

describe('supervisor settings', () => {
  it('puts the control port next to the listener by default', () => {
    const config = loadConfig(base);
    assert.equal(config.supervisor.enabled, true);
    assert.equal(config.supervisor.controlPort, 8788);
    assert.equal(config.supervisor.drainMs, 120_000);
    assert.equal(loadConfig({ ...base, PORT: '9000' }).supervisor.controlPort, 9001);
    assert.equal(loadConfig({ ...base, CONTROL_PORT: '7000' }).supervisor.controlPort, 7000);
  });

  it('refuses a derived control port past 65535', () => {
    assert.throws(() => loadConfig({ ...base, PORT: '65535' }), /CONTROL_PORT/);
    assert.equal(loadConfig({ ...base, PORT: '65535', CONTROL_PORT: '7000' }).supervisor.controlPort, 7000);
  });
});

describe('quota routing settings', () => {
  it('is on by default, with the 429/529 failover', () => {
    const config = loadConfig(base);
    assert.deepEqual(config.quota, { enabled: true, pressure: 0.8, critical: 0.95, minStandardProbability: 0.6 });
    assert.equal(config.router.failoverOnPrimaryRateLimit, true);
    assert.equal(loadConfig({ ...base, QUOTA_ROUTING: 'false' }).quota.enabled, false);
  });

  it('refuses a critical level below the pressure level', () => {
    assert.throws(() => loadConfig({ ...base, QUOTA_PRESSURE: '0.9', QUOTA_CRITICAL: '0.8' }), /QUOTA_CRITICAL/);
  });
});

describe('removedEnvSet', () => {
  it('lists env vars that no longer exist but are still set', () => {
    assert.deepEqual(removedEnvSet({ ROUTER_ALLOW_ESCALATION: 'false', HOST: '127.0.0.1' }), ['ROUTER_ALLOW_ESCALATION']);
    assert.deepEqual(removedEnvSet({}), []);
  });
});

describe('Ollama / cheap keys ignored if present', () => {
  it('ignores all legacy cheap env vars without error', () => {
    const legacy = {
      ...base,
      CHEAP_BASE_URL: 'http://127.0.0.1:11434/v1',
      CHEAP_ROUTE: 'on',
      CHEAP_API_KEY: 'ollama',
      CHEAP_MODEL: 'gpt-oss:20b-cloud',
      CHEAP_MAX_OUTPUT_TOKENS: '8192',
      CHEAP_CONTEXT_TOKENS: '131072',
      CHEAP_TIMEOUT_MS: '60000',
      CHEAP_MODEL_STANDARD: 'gemma4:31b-cloud',
      CHEAP_HEALTH_ENABLED: 'true',
      ROUTER_STANDARD_ROUTE: 'cheap',
      QUOTA_PRESSURE_MIN_CHEAP: '0.7',
      CHEAP_RETRY: 'true',
      CHEAP_READ_FIRST_HINT: 'true',
    };
    const config = loadConfig(legacy);
    assert.ok(config);
    assert.equal((config as any).cheap, undefined);
    assert.equal((config as any).cheapRoute, undefined);
  });
});

describe('New defaults', () => {
  it('provides expected defaults for Argon, Cursor, timeouts, and Gemini', () => {
    const config = loadConfig(base);
    assert.equal(config.argon.enabled, true);
    assert.equal(config.argon.modelLow, 'gemini-4-argon-low');
    assert.equal(config.argon.modelMedium, 'gemini-4-argon-medium');
    assert.equal(config.argon.modelHigh, 'gemini-4-argon-high');

    assert.equal(config.cursorCoding, true);
    assert.equal(config.cursorCodingMaxTextOnly, 0.5);
    assert.equal(config.cursorCodingMinMass, 0.5);

    assert.equal(config.timeouts.firstByteMs, 60_000);
    assert.equal(config.timeouts.idleMs, 60_000);
    assert.equal(config.timeouts.keepaliveIntervalMs, 15_000);

    const cursorOn = loadConfig({ ...base, CURSOR_TIER: 'on' });
    assert.equal(cursorOn.cursor?.maxPromptChars, 200_000);

    const geminiOn = loadConfig({ ...base, GEMINI_TIER: 'on' });
    assert.equal(geminiOn.gemini?.timeoutMs, 300_000);
    assert.equal(geminiOn.gemini?.model, 'gemini-3.8-flash-high');
  });
});
