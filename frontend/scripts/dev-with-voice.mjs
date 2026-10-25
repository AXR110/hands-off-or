import { spawn } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { planWakeListener, detectVoiceAlreadyRunning } from './wake-listener.mjs';

const __filename = fileURLToPath(import.meta.url);
const scriptDir = path.dirname(__filename);
const frontendDir = path.resolve(scriptDir, '..');

const alreadyRunning = process.env.HANDFREEOR_VOICE_STARTED === '1' || detectVoiceAlreadyRunning();
const plan = planWakeListener({ platform: process.platform, alreadyRunning, scriptDir });

let voiceChild = null;
if (plan) {
  voiceChild = spawn(plan.command, plan.args, {
    cwd: plan.cwd,
    stdio: 'inherit',
    windowsHide: true
  });
}

const isStart = process.argv[2] === 'start';
const mode = isStart ? 'start' : 'dev';
const nextArgvStart = isStart ? 3 : 2;
const nextArgs = process.argv.slice(nextArgvStart);

const nextBin = path.join(frontendDir, 'node_modules/next/dist/bin/next');
const nextChild = spawn(process.execPath, [nextBin, mode, ...nextArgs], {
  cwd: frontendDir,
  stdio: 'inherit'
});

nextChild.on('exit', (code) => {
  if (voiceChild) {
    voiceChild.kill();
  }
  process.exit(code);
});

function handleSignal(sig) {
  if (voiceChild) {
    voiceChild.kill(sig);
  }
  if (nextChild) {
    nextChild.kill(sig);
  }
}

process.on('SIGINT', () => handleSignal('SIGINT'));
process.on('SIGTERM', () => handleSignal('SIGTERM'));
