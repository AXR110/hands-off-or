export const DRAW_STROKE = "#FF0000";
export const DRAW_LINE_WIDTH = 4;

export type NormalizedPoint = { x: number; y: number };
export type DrawHand = "primary" | "secondary";
export type DrawStroke = { hand: DrawHand; points: NormalizedPoint[] };
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
): StrokeMap {
  const strokes = drawings[imageId] ?? [];
  if (!continuing) {
    return { ...drawings, [imageId]: [...strokes, { hand, points: [point] }] };
  }
  for (let index = strokes.length - 1; index >= 0; index -= 1) {
    if (strokes[index].hand !== hand) continue;
    const next = strokes.slice();
    next[index] = { hand, points: [...strokes[index].points, point] };
    return { ...drawings, [imageId]: next };
  }
  return { ...drawings, [imageId]: [...strokes, { hand, points: [point] }] };
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
  for (const stroke of strokes) {
    if (stroke.points.length === 0) continue;
    ctx.beginPath();
    stroke.points.forEach((point, index) => {
      const px = point.x * width;
      const py = point.y * height;
      if (index === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    if (stroke.points.length === 1) {
      const only = stroke.points[0];
      ctx.lineTo(only.x * width + 0.01, only.y * height);
    }
    ctx.stroke();
  }
}
