import { describe, it } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, mkdir, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadConfig } from '../src/config.js';
import { resolveCursorLaunch } from '../src/providers/cursor/agent-runner.js';

// Regression: on the Vaio (Linux) every Cursor hop failed with "cursor agent not found",
// because the launcher only knew the Windows layout (%LOCALAPPDATA%\cursor-agent, node.exe).
describe('cursor launch on Linux', () => {
  it('resolves the Linux install layout (versions/<date>/node, no .exe)', async () => {
    const root = await mkdtemp(join(tmpdir(), 'jev-cursor-linux-'));
    const ver = join(root, 'versions', '2026.11.11-b912975');
    await mkdir(ver, { recursive: true });
    await writeFile(join(ver, 'node'), '');
    await writeFile(join(ver, 'index.js'), '');

    assert.deepEqual(resolveCursorLaunch(root), { nodeExe: join(ver, 'node'), indexJs: join(ver, 'index.js') });
  });

  it('defaults to the Linux installer root when CURSOR_TIER_BIN is unset', { skip: process.platform === 'win32' }, () => {
    const c = loadConfig({ CHEAP_API_KEY: 'k', CLASSIFIER: 'heuristic', CURSOR_TIER: 'on' });
    assert.equal(c.cursor?.bin, join(homedir(), '.local', 'share', 'cursor-agent'));
  });
});
