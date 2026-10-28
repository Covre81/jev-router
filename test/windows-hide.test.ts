import assert from 'node:assert/strict';
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { describe, it } from 'node:test';

// Every child process the router or its hooks start must pass `windowsHide: true`,
// otherwise Windows pops a console window over the user's work (issue: flashing terminals).
function walk(dir: string): string[] {
  return readdirSync(dir).flatMap((name) => {
    const p = join(dir, name);
    return statSync(p).isDirectory() ? walk(p) : p.endsWith('.ts') ? [p] : [];
  });
}

const CALL = /\b(spawn|spawnSync|execFile|execFileAsync|execFileSync|exec|execSync|fork)\(/g;

describe('windowsHide on child processes', () => {
  it('every child_process call in src sets windowsHide: true', () => {
    const offenders: string[] = [];
    for (const file of walk('src')) {
      const src = readFileSync(file, 'utf8');
      if (!/node:child_process|['"]child_process['"]/.test(src)) continue;
      for (const m of src.matchAll(CALL)) {
        const before = src.slice(Math.max(0, m.index - 30), m.index);
        if (/function\s+$|\.\s*$|promisify\($/.test(before)) continue; // declarations, db.exec, regex.exec
        const window = src.slice(m.index, m.index + 600);
        if (!/windowsHide:\s*true/.test(window)) {
          offenders.push(`${file}:${src.slice(0, m.index).split('\n').length}`);
        }
      }
    }
    assert.deepEqual(offenders, []);
  });
});
