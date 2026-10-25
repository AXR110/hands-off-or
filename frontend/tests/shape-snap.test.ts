import test from "node:test";
import assert from "node:assert/strict";

import { detectCircle, detectEllipse, smoothPath } from "../components/shapeSnap.ts";
import type { NormalizedPoint } from "../components/draw-strokes.ts";

function parametricStroke(
  count: number,
  pointAt: (angle: number) => NormalizedPoint,
): NormalizedPoint[] {
  return Array.from({ length: count + 1 }, (_, index) =>
    pointAt((index / count) * Math.PI * 2),
  );
}

test("snaps a rough closed circle", () => {
  const roughCircle = parametricStroke(100, (angle) => {
    const radius = 0.2 + 0.012 * Math.sin(angle * 7) + 0.006 * Math.sin(angle * 13);
    return { x: 0.5 + radius * Math.cos(angle), y: 0.5 + radius * Math.sin(angle) };
  });
  assert.equal(detectCircle(roughCircle)?.type, "circle");
});

test("snaps an egg-shaped stroke", () => {
  const egg = parametricStroke(100, (angle) => {
    const radius = 0.2 + 0.035 * Math.cos(angle);
    return { x: 0.5 + radius * Math.cos(angle), y: 0.5 + radius * Math.sin(angle) };
  });
  assert.equal(detectCircle(egg)?.type, "circle");
});

test("snaps an almost-closed loop", () => {
  const loop = Array.from({ length: 100 }, (_, index) => {
    const angle = ((340 * index) / 99) * (Math.PI / 180);
    return { x: 0.5 + 0.2 * Math.cos(angle), y: 0.5 + 0.2 * Math.sin(angle) };
  });
  assert.equal(detectCircle(loop)?.type, "circle");
});

test("does not snap an open arc", () => {
  const arc = Array.from({ length: 70 }, (_, index) => {
    const angle = (-120 + (240 * index) / 69) * (Math.PI / 180);
    return { x: 0.5 + 0.2 * Math.cos(angle), y: 0.5 + 0.2 * Math.sin(angle) };
  });
  assert.equal(detectCircle(arc), null);
});

test("does not snap a straight line", () => {
  const line = Array.from({ length: 60 }, (_, index) => ({
    x: 0.2 + (0.6 * index) / 59,
    y: 0.5,
  }));
  assert.equal(detectCircle(line), null);
});

test("does not snap a figure-eight as a circle", () => {
  const figureEight = parametricStroke(120, (angle) => ({
    x: 0.5 + 0.25 * Math.sin(angle),
    y: 0.5 + 0.16 * Math.sin(2 * angle),
  }));
  assert.equal(detectCircle(figureEight), null);
});

test("snaps a closed axis-aligned ellipse, not a circle", () => {
  const ellipse = parametricStroke(100, (angle) => ({
    x: 0.5 + 0.28 * Math.cos(angle),
    y: 0.5 + 0.12 * Math.sin(angle),
  }));
  assert.equal(detectCircle(ellipse), null);
  assert.equal(detectEllipse(ellipse)?.type, "ellipse");
});

test("smoothPath returns a denser path with the original endpoints", () => {
  const points = [
    { x: 0, y: 0 },
    { x: 0.5, y: 1 },
    { x: 1, y: 0 },
  ];
  const result = smoothPath(points);
  assert.ok(result.length > points.length);
  assert.deepEqual(result[0], points[0]);
  assert.deepEqual(result[result.length - 1], points[2]);
});
