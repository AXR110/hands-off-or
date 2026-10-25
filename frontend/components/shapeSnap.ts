import type { NormalizedPoint } from "./draw-strokes";

export const MIN_POINT_DISTANCE = 0.003;
export const MIN_POINTS = 10;
export const MIN_STROKE_LENGTH = 0.08;
export const RESAMPLE_COUNT = 64;
export const TRIM_RATIO = 0.07;
export const CLOSURE_RATIO = 0.4;
export const CIRCLE_TOLERANCE = 0.28;
export const MIN_SWEEP_DEGREES = 260;
export const CIRCULARITY_WEIGHT = 0.4;
export const CLOSURE_WEIGHT = 0.2;
export const SWEEP_WEIGHT = 0.4;
export const SNAP_SCORE_THRESHOLD = 0.61;
export const ELLIPSE_TOLERANCE = 0.18;
export const ENABLE_ELLIPSE_SNAP = true;
export const SNAP_ANIMATION_MS = 150;

export type SnappedShape = {
  type: "circle" | "ellipse";
  points: NormalizedPoint[];
};

type ImagePoint = { x: number; y: number };

function toImageSpace(point: NormalizedPoint, aspectRatio: number): ImagePoint {
  return { x: point.x * aspectRatio, y: point.y };
}

function fromImageSpace(point: ImagePoint, aspectRatio: number): NormalizedPoint {
  return { x: point.x / aspectRatio, y: point.y };
}

function distance(a: ImagePoint, b: ImagePoint) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function clamp(value: number, min: number, max: number) {
  return Math.max(min, Math.min(max, value));
}

function fitKasaCircle(points: ImagePoint[]) {
  let xx = 0;
  let xy = 0;
  let x = 0;
  let yy = 0;
  let y = 0;
  let xz = 0;
  let yz = 0;
  let z = 0;

  for (const point of points) {
    const squared = point.x * point.x + point.y * point.y;
    xx += point.x * point.x;
    xy += point.x * point.y;
    x += point.x;
    yy += point.y * point.y;
    y += point.y;
    xz -= point.x * squared;
    yz -= point.y * squared;
    z -= squared;
  }

  const matrix = [
    [xx, xy, x, xz],
    [xy, yy, y, yz],
    [x, y, points.length, z],
  ];
  for (let column = 0; column < 3; column += 1) {
    let pivot = column;
    for (let row = column + 1; row < 3; row += 1) {
      if (Math.abs(matrix[row][column]) > Math.abs(matrix[pivot][column])) pivot = row;
    }
    if (Math.abs(matrix[pivot][column]) < Number.EPSILON) return null;
    [matrix[column], matrix[pivot]] = [matrix[pivot], matrix[column]];

    const divisor = matrix[column][column];
    for (let entry = column; entry < 4; entry += 1) {
      matrix[column][entry] /= divisor;
    }
    for (let row = 0; row < 3; row += 1) {
      if (row === column) continue;
      const factor = matrix[row][column];
      for (let entry = column; entry < 4; entry += 1) {
        matrix[row][entry] -= factor * matrix[column][entry];
      }
    }
  }

  const center = { x: -matrix[0][3] / 2, y: -matrix[1][3] / 2 };
  const radiusSquared = center.x * center.x + center.y * center.y - matrix[2][3];
  if (!Number.isFinite(radiusSquared) || radiusSquared <= 0) return null;
  return { center, radius: Math.sqrt(radiusSquared) };
}

function pathProperties(points: NormalizedPoint[], aspectRatio: number) {
  if (points.length < MIN_POINTS || aspectRatio <= 0) return null;
  const imagePoints = points.map((point) => toImageSpace(point, aspectRatio));
  let length = 0;
  let sweep = 0;
  for (let index = 1; index < imagePoints.length; index += 1) {
    length += distance(imagePoints[index - 1], imagePoints[index]);
  }
  const first = imagePoints[0];
  const last = imagePoints[imagePoints.length - 1];
  const closed = length >= MIN_STROKE_LENGTH && distance(first, last) < CLOSURE_RATIO * length;
  if (!closed) return { imagePoints, length, closed, sweep };

  const center = imagePoints.reduce(
    (sum, point) => ({ x: sum.x + point.x / imagePoints.length, y: sum.y + point.y / imagePoints.length }),
    { x: 0, y: 0 },
  );
  for (let index = 1; index < imagePoints.length; index += 1) {
    const previousAngle = Math.atan2(
      imagePoints[index - 1].y - center.y,
      imagePoints[index - 1].x - center.x,
    );
    const angle = Math.atan2(imagePoints[index].y - center.y, imagePoints[index].x - center.x);
    const delta = Math.atan2(Math.sin(angle - previousAngle), Math.cos(angle - previousAngle));
    sweep += Math.abs(delta);
  }
  return { imagePoints, length, closed, sweep };
}

function sampleShape(
  type: SnappedShape["type"],
  center: ImagePoint,
  radiusX: number,
  radiusY: number,
  pointCount: number,
  aspectRatio: number,
): SnappedShape {
  const points = Array.from({ length: pointCount }, (_, index) => {
    const angle = (index / pointCount) * Math.PI * 2;
    return fromImageSpace(
      {
        x: center.x + radiusX * Math.cos(angle),
        y: center.y + radiusY * Math.sin(angle),
      },
      aspectRatio,
    );
  });
  return { type, points };
}

function resampleCount(points: NormalizedPoint[]) {
  return Math.max(32, Math.min(128, points.length));
}

export function smoothPath(points: NormalizedPoint[]): NormalizedPoint[] {
  if (points.length < 3) return points.slice();
  const smoothed: NormalizedPoint[] = [points[0]];
  for (let index = 1; index < points.length - 1; index += 1) {
    const control = points[index];
    const start = smoothed[smoothed.length - 1];
    const end = {
      x: (control.x + points[index + 1].x) / 2,
      y: (control.y + points[index + 1].y) / 2,
    };
    const subdivisions = 6;
    for (let step = 1; step <= subdivisions; step += 1) {
      const t = step / subdivisions;
      const inverse = 1 - t;
      smoothed.push({
        x: inverse * inverse * start.x + 2 * inverse * t * control.x + t * t * end.x,
        y: inverse * inverse * start.y + 2 * inverse * t * control.y + t * t * end.y,
      });
    }
  }
  smoothed.push(points[points.length - 1]);
  return smoothed;
}

export function resamplePath(
  points: NormalizedPoint[],
  count: number,
  aspectRatio = 1,
): NormalizedPoint[] {
  if (points.length < 2 || count < 2 || aspectRatio <= 0) return points.slice();
  const imagePoints = points.map((point) => toImageSpace(point, aspectRatio));
  const distances = [0];
  for (let index = 1; index < imagePoints.length; index += 1) {
    distances.push(distances[index - 1] + distance(imagePoints[index - 1], imagePoints[index]));
  }
  const total = distances[distances.length - 1];
  if (total <= 0) return Array.from({ length: count }, () => points[0]);

  const result: NormalizedPoint[] = [];
  let segment = 1;
  for (let index = 0; index < count; index += 1) {
    const target = (total * index) / (count - 1);
    while (segment < distances.length - 1 && distances[segment] < target) segment += 1;
    const segmentStart = distances[segment - 1];
    const segmentLength = distances[segment] - segmentStart;
    const ratio = segmentLength <= 0 ? 0 : (target - segmentStart) / segmentLength;
    result.push(
      fromImageSpace(
        {
          x: imagePoints[segment - 1].x + (imagePoints[segment].x - imagePoints[segment - 1].x) * ratio,
          y: imagePoints[segment - 1].y + (imagePoints[segment].y - imagePoints[segment - 1].y) * ratio,
        },
        aspectRatio,
      ),
    );
  }
  return result;
}

export function detectCircle(
  points: NormalizedPoint[],
  aspectRatio = 1,
): SnappedShape | null {
  if (points.length < MIN_POINTS || aspectRatio <= 0) return null;
  const resampled = resamplePath(points, RESAMPLE_COUNT, aspectRatio);
  const imagePoints = resampled.map((point) => toImageSpace(point, aspectRatio));
  let length = 0;
  for (let index = 1; index < imagePoints.length; index += 1) {
    length += distance(imagePoints[index - 1], imagePoints[index]);
  }
  if (length < MIN_STROKE_LENGTH) return null;

  const trimCount = Math.floor(imagePoints.length * TRIM_RATIO);
  const fittingPoints = imagePoints.slice(trimCount, imagePoints.length - trimCount);
  const fitted = fitKasaCircle(fittingPoints);
  if (!fitted) return null;

  const { center, radius } = fitted;
  const radii = fittingPoints.map((point) => distance(point, center));
  const rms = Math.sqrt(
    radii.reduce((sum, value) => sum + (value - radius) ** 2, 0) / radii.length,
  );
  const circularityError = rms / radius;

  let sweepRadians = 0;
  for (let index = 1; index < fittingPoints.length; index += 1) {
    const previousAngle = Math.atan2(
      fittingPoints[index - 1].y - center.y,
      fittingPoints[index - 1].x - center.x,
    );
    const angle = Math.atan2(
      fittingPoints[index].y - center.y,
      fittingPoints[index].x - center.x,
    );
    const delta = Math.atan2(Math.sin(angle - previousAngle), Math.cos(angle - previousAngle));
    sweepRadians += delta;
  }
  const sweepDegrees = (Math.abs(sweepRadians) * 180) / Math.PI;
  const closureGapRatio =
    distance(imagePoints[0], imagePoints[imagePoints.length - 1]) / length;
  const circularityScore = clamp(1 - circularityError / CIRCLE_TOLERANCE, 0, 1);
  const closureScore = clamp(1 - closureGapRatio / CLOSURE_RATIO, 0, 1);
  const sweepScore =
    sweepDegrees < MIN_SWEEP_DEGREES
      ? 0
      : clamp((sweepDegrees - MIN_SWEEP_DEGREES) / (360 - MIN_SWEEP_DEGREES), 0, 1);
  const score =
    CIRCULARITY_WEIGHT * circularityScore +
    CLOSURE_WEIGHT * closureScore +
    SWEEP_WEIGHT * sweepScore;
  if (score <= SNAP_SCORE_THRESHOLD) return null;

  return sampleShape("circle", center, radius, radius, resampleCount(points), aspectRatio);
}

export function detectEllipse(
  points: NormalizedPoint[],
  aspectRatio = 1,
): SnappedShape | null {
  const path = pathProperties(points, aspectRatio);
  if (!path || !path.closed || path.sweep < (300 * Math.PI) / 180) return null;
  const xs = path.imagePoints.map((point) => point.x);
  const ys = path.imagePoints.map((point) => point.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const radiusX = (maxX - minX) / 2;
  const radiusY = (maxY - minY) / 2;
  if (radiusX <= 0 || radiusY <= 0) return null;
  const center = { x: (minX + maxX) / 2, y: (minY + maxY) / 2 };
  const radialErrors = path.imagePoints.map((point) =>
    Math.hypot((point.x - center.x) / radiusX, (point.y - center.y) / radiusY) - 1,
  );
  const rms = Math.sqrt(
    radialErrors.reduce((sum, value) => sum + value ** 2, 0) / radialErrors.length,
  );
  if (rms >= ELLIPSE_TOLERANCE) return null;
  return sampleShape("ellipse", center, radiusX, radiusY, resampleCount(points), aspectRatio);
}
