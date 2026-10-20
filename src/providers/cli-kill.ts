import { spawn } from 'node:child_process';

/** Kill a CLI child process and its whole tree (shared by the Cursor and Gemini CLI runners). */
export function killTree(pid: number): void {
  if (process.platform === 'win32') {
    spawn('taskkill', ['/pid', pid.toString(), '/T', '/F'], { windowsHide: true }).on('error', () => {});
  } else {
    try { process.kill(pid, 'SIGKILL'); } catch {}
  }
}
