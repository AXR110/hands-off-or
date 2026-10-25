import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

test('page.tsx contains manual record attributes', () => {
  const pagePath = path.join(process.cwd(), 'app', 'page.tsx');
  const src = fs.readFileSync(pagePath, 'utf8');
  assert.ok(src.includes('data-start-recording'), 'missing data-start-recording');
  assert.ok(src.includes('Record'), 'missing Record label');
  assert.ok(src.includes('start_recording'), 'missing start_recording string');
});
