import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { planWakeListener } from "../scripts/wake-listener.mjs";

test('planWakeListener win32 not running', () => {
  const result = planWakeListener({ platform: "win32", alreadyRunning: false, scriptDir: "C:/app/frontend/scripts" });
  assert.strictEqual(result.command, "python");
  assert.match(result.args[0], /voice\.py$/);
  assert.match(result.cwd, /backend$/);
});

test('planWakeListener linux not running', () => {
  const result = planWakeListener({ platform: "linux", alreadyRunning: false, scriptDir: "/app/frontend/scripts" });
  assert.strictEqual(result.command, "python3");
  assert.match(result.args[0], /voice\.py$/);
});

test('planWakeListener win32 already running', () => {
  const result = planWakeListener({ platform: "win32", alreadyRunning: true, scriptDir: "/app/frontend/scripts" });
  assert.strictEqual(result, null);
});

test('package.json scripts include dev-with-voice', () => {
  const pkg = JSON.parse(fs.readFileSync('package.json', 'utf8'));
  assert.match(pkg.scripts.dev, /dev-with-voice\.mjs/);
  assert.match(pkg.scripts.start, /dev-with-voice\.mjs/);
});

test('start.sh contains voice markers', () => {
  const sh = fs.readFileSync('../start.sh', 'utf8');
  assert.match(sh, /HANDFREEOR_VOICE_STARTED/);
  assert.match(sh, /voice\.py/);
});
