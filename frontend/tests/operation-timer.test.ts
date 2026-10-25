import test from "node:test";
import assert from "node:assert/strict";
import {
  idleOperationTimer,
  startOperationTimer,
  stopOperationTimer,
  operationElapsedMs,
  formatOperationDuration,
} from "../app/operation-timer.ts";

test("idle timer is zero until the first start", () => {
  const timer = idleOperationTimer();
  assert.equal(operationElapsedMs(timer, 5_000), 0);
  assert.equal(formatOperationDuration(0), "00:00:00");
});

test("the first start begins the clock and later starts do not reset it", () => {
  const started = startOperationTimer(idleOperationTimer(), 1_000);
  const again = startOperationTimer(started, 9_000);
  assert.equal(again, started);
  assert.equal(operationElapsedMs(started, 4_500), 3_500);
  assert.equal(formatOperationDuration(3_500), "00:00:03");
});

test("stopping freezes the elapsed time", () => {
  const started = startOperationTimer(idleOperationTimer(), 1_000);
  const stopped = stopOperationTimer(started, 65_000);
  assert.equal(operationElapsedMs(stopped, 120_000), 64_000);
  assert.equal(formatOperationDuration(64_000), "00:01:04");
  assert.equal(stopOperationTimer(stopped, 70_000), stopped);
  assert.equal(startOperationTimer(stopped, 80_000), stopped);
});

test("formatOperationDuration rolls into hours", () => {
  assert.equal(formatOperationDuration(3_661_000), "01:01:01");
});
