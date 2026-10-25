import test from "node:test";
import assert from "node:assert/strict";

import {
  appendHandStroke,
  clearImageStrokes,
  finishHandStroke,
  pointerToCanvas,
  strokesForImage,
  DRAW_LINE_WIDTH,
  DRAW_STROKE,
} from "../components/draw-strokes.ts";

const film = { left: 100, top: 50, width: 200, height: 400 };
const viewport = { width: 1000, height: 800 };

test("pointer maps into the canvas box instead of the window", () => {
  const point = pointerToCanvas(0.2, 0.25, film, viewport);
  assert.deepStrictEqual(point, { x: 0.5, y: 0.375 });
});

test("pointer outside the film is not a stroke point", () => {
  assert.equal(pointerToCanvas(0.05, 0.25, film, viewport), null);
});

test("strokes stay attached to the image that received them", () => {
  let drawings = appendHandStroke({}, "chest", "primary", { x: 0.1, y: 0.2 }, false);
  drawings = appendHandStroke(drawings, "chest", "primary", { x: 0.4, y: 0.5 }, true);
  drawings = appendHandStroke(drawings, "hand", "primary", { x: 0.8, y: 0.9 }, false);

  assert.deepStrictEqual(strokesForImage(drawings, "chest"), [
    {
      hand: "primary",
      type: "freehand",
      points: [
        { x: 0.1, y: 0.2 },
        { x: 0.4, y: 0.5 },
      ],
    },
  ]);
  assert.deepStrictEqual(strokesForImage(drawings, "hand"), [
    { hand: "primary", type: "freehand", points: [{ x: 0.8, y: 0.9 }] },
  ]);
  assert.deepStrictEqual(strokesForImage(drawings, "teeth"), []);
  assert.equal(DRAW_STROKE, "#FF0000");
  assert.equal(DRAW_LINE_WIDTH, 4);
});

test("clearing one image leaves the other image's strokes", () => {
  let drawings = appendHandStroke({}, "chest", "primary", { x: 0.2, y: 0.3 }, false);
  drawings = appendHandStroke(drawings, "hand", "primary", { x: 0.6, y: 0.7 }, false);
  drawings = clearImageStrokes(drawings, "chest");
  assert.deepStrictEqual(strokesForImage(drawings, "chest"), []);
  assert.equal(strokesForImage(drawings, "hand").length, 1);
});

test("two pinching hands keep separate paths", () => {
  let drawings = appendHandStroke({}, "chest", "primary", { x: 0.2, y: 0.2 }, false);
  drawings = appendHandStroke(drawings, "chest", "secondary", { x: 0.7, y: 0.7 }, false);
  drawings = appendHandStroke(drawings, "chest", "primary", { x: 0.3, y: 0.2 }, true);

  const strokes = strokesForImage(drawings, "chest");
  assert.deepStrictEqual(strokes[0].points, [
    { x: 0.2, y: 0.2 },
    { x: 0.3, y: 0.2 },
  ]);
  assert.equal(strokes[0].type, "freehand");
  assert.deepStrictEqual(strokes[1], {
    hand: "secondary",
    type: "freehand",
    points: [{ x: 0.7, y: 0.7 }],
  });
});

test("nearby telemetry points are dropped", () => {
  let drawings = appendHandStroke({}, "chest", "primary", { x: 0.2, y: 0.2 }, false);
  const unchanged = appendHandStroke(
    drawings,
    "chest",
    "primary",
    { x: 0.201, y: 0.2 },
    true,
  );
  assert.equal(unchanged, drawings);
  drawings = appendHandStroke(drawings, "chest", "primary", { x: 0.22, y: 0.2 }, true);
  assert.equal(strokesForImage(drawings, "chest")[0].points.length, 2);
});

test("finished circles are tagged and retain an animation source path", () => {
  const circle = Array.from({ length: 80 }, (_, index) => {
    const angle = (index / 80) * Math.PI * 2;
    return { x: 0.5 + 0.2 * Math.cos(angle), y: 0.5 + 0.2 * Math.sin(angle) };
  });
  let drawings = appendHandStroke({}, "chest", "primary", circle[0], false);
  for (const point of circle.slice(1)) {
    drawings = appendHandStroke(drawings, "chest", "primary", point, true);
  }
  drawings = finishHandStroke(drawings, "chest", "primary", 1, 100);
  const stroke = strokesForImage(drawings, "chest")[0];
  assert.equal(stroke.type, "circle");
  assert.ok(stroke.snapFrom);
  assert.equal(stroke.snapStartedAt, 100);
});
