import path from 'node:path';
import { execSync } from 'node:child_process';

export function voiceAlreadyRunningFromLines(lines) {
  return Array.isArray(lines) && lines.some(line => /voice\.py/.test(String(line || '')));
}

export function detectVoiceAlreadyRunning() {
  try {
    if (process.platform === 'win32') {
      const cmdStr = 'powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like \'*voice.py*\' -and $_.CommandLine -notlike \'*Get-CimInstance*\' -and $_.CommandLine -notlike \'*powershell -NoProfile*\' } | Select-Object -ExpandProperty CommandLine"';
      const output = execSync(cmdStr, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
      return voiceAlreadyRunningFromLines(output.split(/\r?\n/));
    } else {
      const output = execSync('pgrep -f voice.py', { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
      return output.trim().length > 0;
    }
  } catch (e) {
    return false;
  }
}

export function planWakeListener({ platform, alreadyRunning, scriptDir }) {
  if (alreadyRunning) {
    return null;
  }
  const command = platform === 'win32' ? 'python' : 'python3';
  const cwd = path.resolve(scriptDir, '../../backend');
  const args = [path.join(cwd, 'voice.py')];
  return { command, args, cwd };
}
