import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';

test('layout body includes suppressHydrationWarning', () => {
  const layoutPath = path.resolve(process.cwd(), 'app/layout.tsx');
  const content = fs.readFileSync(layoutPath, 'utf8');
  assert.match(content, /<body[^>]*suppressHydrationWarning/);
});
