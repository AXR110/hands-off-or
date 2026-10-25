import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { armManualRecording, emptySpeechLog } from '../app/speech-log-state.ts';

test('armManualRecording contract', () => {
  const empty = emptySpeechLog();
  const v1 = armManualRecording(empty);
  assert.strictEqual(v1.recording, true);
  assert.strictEqual(v1.liveText, '');
  assert.deepStrictEqual(v1.entries, empty.entries);

  const already = { ...empty, recording: true, liveText: 'add sponge' };
  const v2 = armManualRecording(already);
  assert.strictEqual(v2.recording, true);
  assert.strictEqual(v2.liveText, 'add sponge');
  assert.ok(v2 === already || (v2.recording === true && v2.liveText === 'add sponge'));

  const pagePath = path.join(process.cwd(), 'app', 'page.tsx');
  const src = fs.readFileSync(pagePath, 'utf8');
  assert.ok(src.includes('armManualRecording'), 'missing armManualRecording');
});
