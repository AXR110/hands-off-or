import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';

import { recordingFrameClass } from '../app/speech-log-state.ts';

test('recordingFrameClass returns null when not recording', () => {
  assert.strictEqual(recordingFrameClass(false), null);
});

test('recordingFrameClass returns "recording-frame" when recording', () => {
  assert.strictEqual(recordingFrameClass(true), 'recording-frame');
});

test('globals.css contains required recording-frame styles', () => {
  const css = fs.readFileSync('app/globals.css', 'utf8');
  assert.ok(css.includes('.recording-frame'), 'missing .recording-frame selector');
  assert.ok(css.includes('recording-border-pulse'), 'missing keyframes name');
  assert.ok(css.includes('inset'), 'missing inset');
  assert.ok(css.includes('pointer-events'), 'missing pointer-events');
  assert.ok(css.includes('rgba(34, 197, 94'), 'missing green color');
  const frameRule = css.split('.recording-frame')[1]?.split('}')[0] || '';
  assert.ok(!frameRule.includes('background:') && !frameRule.includes('background-color:'), 'must not set solid background fill');
});

test('page.tsx contains data-recording-frame and recordingFrameClass wiring', () => {
  const page = fs.readFileSync('app/page.tsx', 'utf8');
  assert.ok(page.includes('data-recording-frame'), 'missing data-recording-frame');
  assert.ok(page.includes('recordingFrameClass'), 'missing recordingFrameClass usage');
});
