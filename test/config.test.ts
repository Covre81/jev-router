import assert from 'node:assert/strict';
import { describe, it } from 'node:test';
import { loadConfig, removedEnvSet } from '../src/config.js';

const base = { CLASSIFIER: 'heuristic', CHEAP_API_KEY: 'x' };

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

  it('reads the cheap health check switches', () => {
    assert.deepEqual(loadConfig(base).cheapHealth, { enabled: true, intervalMs: 30_000 });
    assert.equal(loadConfig({ ...base, CHEAP_HEALTH_ENABLED: 'false' }).cheapHealth.enabled, false);
  });
});

describe('quota routing settings', () => {
  it('is on by default, with the 429/529 failover', () => {
    const config = loadConfig(base);
    assert.deepEqual(config.quota, { enabled: true, pressure: 0.8, critical: 0.95, minCheapProbability: 0.7, minStandardProbability: 0.6 });
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

describe('CHEAP_ROUTE', () => {
  it('is off by default and then needs no CHEAP_API_KEY', () => {
    assert.equal(loadConfig(base).cheapRoute, false);
    const { CHEAP_API_KEY: _, ...noKey } = base as Record<string, string>;
    assert.equal(loadConfig(noKey).cheapRoute, false);
  });

  it('requires CHEAP_API_KEY when on', () => {
    const { CHEAP_API_KEY: _, ...noKey } = base as Record<string, string>;
    assert.throws(() => loadConfig({ ...noKey, CHEAP_ROUTE: 'on' }), /CHEAP_API_KEY/);
    assert.equal(loadConfig({ ...base, CHEAP_ROUTE: 'on' }).cheapRoute, true);
  });
});
