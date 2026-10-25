import {
  detectCircle,
  detectEllipse,
  ENABLE_ELLIPSE_SNAP,
  MIN_POINT_DISTANCE,
  resamplePath,
  smoothPath,
  SNAP_ANIMATION_MS,
} from "./shapeSnap.ts";

export const DRAW_STROKE = "#FF0000";
export const DRAW_LINE_WIDTH = 4;

export type NormalizedPoint = { x: number; y: number };
export type DrawHand = "primary" | "secondary";
export type DrawStrokeType = "freehand" | "circle" | "ellipse";
export type DrawStroke = {
  hand: DrawHand;
  type: DrawStrokeType;
  points: NormalizedPoint[];
  snapFrom?: NormalizedPoint[];
  snapStartedAt?: number;
};
export type StrokeMap = Record<string, DrawStroke[]>;

export type CanvasBox = { left: number; top: number; width: number; height: number };
export type Viewport = { width: number; height: number };

/** Map a normalized window pointer onto the canvas box. Outside the film is null. */
export function pointerToCanvas(
  x: number,
  y: number,
  rect: CanvasBox,
  viewport: Viewport,
): NormalizedPoint | null {
  if (rect.width <= 0 || rect.height <= 0 || viewport.width <= 0 || viewport.height <= 0) {
    return null;
  }
  const localX = (x * viewport.width - rect.left) / rect.width;
  const localY = (y * viewport.height - rect.top) / rect.height;
  if (localX < 0 || localX > 1 || localY < 0 || localY > 1) return null;
  return { x: localX, y: localY };
}

/** Continue this hand's path, or start a new one. Other images are left untouched. */
export function appendHandStroke(
  drawings: StrokeMap,
  imageId: string,
  hand: DrawHand,
  point: NormalizedPoint,
  continuing: boolean,
  aspectRatio = 1,
): StrokeMap {
  const strokes = drawings[imageId] ?? [];
  if (!continuing) {
    return { ...drawings, [imageId]: [...strokes, { hand, type: "freehand", points: [point] }] };
  }
  for (let index = strokes.length - 1; index >= 0; index -= 1) {
    if (strokes[index].hand !== hand) continue;
    const previous = strokes[index].points[strokes[index].points.length - 1];
    if (
      previous &&
      Math.hypot((point.x - previous.x) * aspectRatio, point.y - previous.y) <
        MIN_POINT_DISTANCE
    ) {
      return drawings;
    }
    const next = strokes.slice();
    next[index] = { ...strokes[index], points: [...strokes[index].points, point] };
    return { ...drawings, [imageId]: next };
  }
  return { ...drawings, [imageId]: [...strokes, { hand, type: "freehand", points: [point] }] };
}

export function finishHandStroke(
  drawings: StrokeMap,
  imageId: string,
  hand: DrawHand,
  aspectRatio: number,
  now: number,
): StrokeMap {
  const strokes = drawings[imageId] ?? [];
  for (let index = strokes.length - 1; index >= 0; index -= 1) {
    const stroke = strokes[index];
    if (stroke.hand !== hand || stroke.type !== "freehand") continue;
    const circle = detectCircle(stroke.points, aspectRatio);
    const ellipse =
      circle == null && ENABLE_ELLIPSE_SNAP
        ? detectEllipse(stroke.points, aspectRatio)
        : null;
    const shape = circle ?? ellipse;
    if (!shape) return drawings;
    const next = strokes.slice();
    next[index] = {
      hand,
      type: shape.type,
      points: shape.points,
      snapFrom: resamplePath(smoothPath(stroke.points), shape.points.length, aspectRatio),
      snapStartedAt: now,
    };
    return { ...drawings, [imageId]: next };
  }
  return drawings;
}

export function strokesForImage(drawings: StrokeMap, imageId: string): DrawStroke[] {
  return drawings[imageId] ?? [];
}

/** Drop every stroke saved for one image. Other images stay as they are. */
export function clearImageStrokes(drawings: StrokeMap, imageId: string): StrokeMap {
  if (drawings[imageId] == null) return drawings;
  const next = { ...drawings };
  delete next[imageId];
  return next;
}

export function paintStrokes(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  strokes: DrawStroke[],
) {
  ctx.clearRect(0, 0, width, height);
  ctx.strokeStyle = DRAW_STROKE;
  ctx.lineWidth = DRAW_LINE_WIDTH;
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  const now = performance.now();
  for (const stroke of strokes) {
    if (stroke.points.length === 0) continue;
    const animationProgress =
      stroke.snapFrom && stroke.snapStartedAt != null
        ? Math.max(0, Math.min(1, (now - stroke.snapStartedAt) / SNAP_ANIMATION_MS))
        : 1;
    const points =
      stroke.snapFrom && animationProgress < 1
        ? stroke.points.map((point, index) => {
            const from = stroke.snapFrom?.[index];
            return from
              ? {
                  x: from.x * (1 - animationProgress) + point.x * animationProgress,
                  y: from.y * (1 - animationProgress) + point.y * animationProgress,
                }
              : point;
          })
        : stroke.points;
    if (stroke.type !== "freehand" && animationProgress >= 1) {
      const xs = stroke.points.map((point) => point.x * width);
      const ys = stroke.points.map((point) => point.y * height);
      const left = Math.min(...xs);
      const right = Math.max(...xs);
      const top = Math.min(...ys);
      const bottom = Math.max(...ys);
      ctx.beginPath();
      ctx.ellipse(
        (left + right) / 2,
        (top + bottom) / 2,
        (right - left) / 2,
        (bottom - top) / 2,
        0,
        0,
        Math.PI * 2,
      );
      ctx.stroke();
      continue;
    }
    const smoothPoints = smoothPath(points);
    ctx.beginPath();
    smoothPoints.forEach((point, index) => {
      const px = point.x * width;
      const py = point.y * height;
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    if (smoothPoints.length === 1) {
      const only = smoothPoints[0];
      ctx.lineTo(only.x * width + 0.01, only.y * height);
    }
    ctx.stroke();
  }
}
