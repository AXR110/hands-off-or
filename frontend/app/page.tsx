"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const BRIDGE_URL = process.env.NEXT_PUBLIC_BRIDGE_URL ?? "ws://localhost:8000/ws";
const LOG_LIMIT = 50;
const KNOWN_TOOLS = ["set_pointer", "set_pinch"];
const HAND_DETECTION_TIMEOUT_MS = 500;
const EDGE_WARNING_THRESHOLD = 0.06;

const STUDIES = [
  { id: "chest", label: "Chest", detail: "PA chest", src: "/studies/chest.png" },
  { id: "hand", label: "Hand", detail: "Hand", src: "/studies/hand.png" },
  { id: "teeth", label: "Teeth", detail: "Dental", src: "/studies/teeth.png" },
] as const;

type StudyId = (typeof STUDIES)[number]["id"];

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ZOOM_START_PX = 48;
const DOUBLE_PINCH_MS = 650;
const DOUBLE_PINCH_SLOP_PX = 140;
// A flick while one pinch is held. Slow drifts and sideways motion stay put.
const SWIPE_WINDOW_MS = 320;
const SWIPE_MIN_MS = 80;
const SWIPE_MIN_SAMPLES = 3;
const SWIPE_MIN_SPEED = 0.45;
const SWIPE_AXIS_RATIO = 1.5;
// One film per flick. Pinch tracking can drop for a frame mid-swipe; wait until it stays open.
const SWIPE_RELEASE_MS = 320;

type Point = { x: number; y: number };

type ZoomGesture = {
  baseline: number;
  scale: number;
  anchorX: number;
  anchorY: number;
};

type PinchMark = {
  at: number;
  x: number;
  y: number;
};

type SwipeSample = {
  x: number;
  y: number;
  at: number;
};

type ImageLayout = {
  centerX: number;
  centerY: number;
  width: number;
  height: number;
};

function clamp(value: number, min: number, max: number) {
  return Math.min(max, Math.max(min, value));
}

function clampZoom(scale: number) {
  return clamp(scale, MIN_ZOOM, MAX_ZOOM);
}

// Keep the pinched spot fixed. Only stop the film from sliding fully out of view.
function clampAxis(offset: number, center: number, scaledSize: number, start: number, end: number) {
  const overlap = Math.min(64, scaledSize, end - start);
  const lower = start + overlap - center - scaledSize / 2;
  const upper = end - overlap - center + scaledSize / 2;
  return clamp(offset, Math.min(lower, upper), Math.max(lower, upper));
}

function imageLayout(image: HTMLElement): ImageLayout | null {
  const width = image.offsetWidth;
  const height = image.offsetHeight;
  const parent = image.offsetParent;
  if (!(parent instanceof HTMLElement) || width < 1 || height < 1) return null;
  const parentRect = parent.getBoundingClientRect();
  return {
    centerX: parentRect.left + parent.clientLeft + image.offsetLeft + width / 2,
    centerY: parentRect.top + parent.clientTop + image.offsetTop + height / 2,
    width,
    height,
  };
}

function focalInStage(x: number, y: number, x2: number, y2: number, stage: DOMRect): Point {
  return {
    x: clamp(((x + x2) / 2) * window.innerWidth, stage.left, stage.right),
    y: clamp(((y + y2) / 2) * window.innerHeight, stage.top, stage.bottom),
  };
}

// Scale around the content point under the pinch, and follow that point if it moves.
function zoomAtFocal(
  gesture: ZoomGesture,
  focal: Point,
  distance: number,
  layout: ImageLayout,
  stage: DOMRect,
) {
  const scale = clampZoom(gesture.scale * (distance / gesture.baseline));
  if (scale <= MIN_ZOOM) return { scale: MIN_ZOOM, x: 0, y: 0 };
  return {
    scale,
    x: clampAxis(
      focal.x - layout.centerX - gesture.anchorX * scale,
      layout.centerX,
      layout.width * scale,
      stage.left,
      stage.right,
    ),
    y: clampAxis(
      focal.y - layout.centerY - gesture.anchorY * scale,
      layout.centerY,
      layout.height * scale,
      stage.top,
      stage.bottom,
    ),
  };
}

function studyAtPoint(x: number, y: number): StudyId | null {
  const element = document.elementFromPoint(x * window.innerWidth, y * window.innerHeight);
  const id = element?.closest("[data-study]")?.getAttribute("data-study");
  if (id === "chest" || id === "hand" || id === "teeth") return id;
  return null;
}

function pointerOverStage(x: number, y: number, stage: DOMRect | undefined) {
  if (!stage || stage.width <= 0 || stage.height <= 0) return false;
  const px = x * window.innerWidth;
  const py = y * window.innerHeight;
  return px >= stage.left && px <= stage.right && py >= stage.top && py <= stage.bottom;
}

// 1 = finger traveled up, so show the next film. -1 = finger traveled down, so go back.
function classifyVerticalSwipe(
  samples: SwipeSample[],
  viewportWidth: number,
  viewportHeight: number,
): -1 | 1 | null {
  if (samples.length < SWIPE_MIN_SAMPLES || viewportWidth < 1 || viewportHeight < 1) return null;
  const minTravel = Math.max(72, viewportHeight * 0.08);
  let bestSpeed = 0;
  let best: -1 | 1 | null = null;

  for (let start = 0; start < samples.length - 1; start += 1) {
    const from = samples[start];
    const to = samples[samples.length - 1];
    const elapsed = to.at - from.at;
    if (elapsed < SWIPE_MIN_MS || elapsed > SWIPE_WINDOW_MS) continue;
    const dx = (to.x - from.x) * viewportWidth;
    const dy = (to.y - from.y) * viewportHeight;
    const absX = Math.abs(dx);
    const absY = Math.abs(dy);
    if (absY < minTravel || absY < absX * SWIPE_AXIS_RATIO) continue;
    const speed = absY / elapsed;
    if (speed < SWIPE_MIN_SPEED || speed <= bestSpeed) continue;
    if (!swipePathAgrees(samples, start)) continue;
    bestSpeed = speed;
    best = dy < 0 ? 1 : -1;
  }
  return best;
}

function swipePathAgrees(samples: SwipeSample[], start: number) {
  const sign = Math.sign(samples[samples.length - 1].y - samples[start].y);
  if (sign === 0) return false;
  let agreeing = 0;
  let steps = 0;
  let largest = 0;
  let total = 0;
  for (let index = start + 1; index < samples.length; index += 1) {
    const step = samples[index].y - samples[index - 1].y;
    if (Math.abs(step) < 0.003) continue;
    steps += 1;
    const magnitude = Math.abs(step);
    total += magnitude;
    if (magnitude > largest) largest = magnitude;
    if (Math.sign(step) === sign) agreeing += 1;
  }
  // One jumped sample is tracking noise. A flick keeps moving for several frames.
  if (steps < 2 || agreeing / steps < 0.7 || total <= 0) return false;
  return largest / total <= 0.8;
}

type ConnectionStatus = "connecting" | "online" | "reconnecting";

// Mirrors $defs/telemetryEvent and $defs/stateSnapshot in backend/shared_schema.json.
type Telemetry = {
  type: "telemetry";
  x: number;
  y: number;
  pinch_active: boolean;
  source: string;
  seq: number;
  client_ts: number | null;
  client_seq: number | null;
  server_ts: number;
  x2?: number | null;
  y2?: number | null;
  pinch2_active?: boolean | null;
};

// Mirrors $defs/functionCall.
type FunctionCallFrame = {
  type: "function_call";
  call_id: string;
  name: string;
  args: Record<string, unknown>;
  origin?: "client" | "gemini" | "server";
  server_ts?: number;
};

type StateSnapshotFrame = {
  type: "state_snapshot";
  telemetry: Telemetry | null;
};

type ToolCall = {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  origin: string;
  at: number;
};

function useBridge() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [handDetected, setHandDetected] = useState(false);
  const [calls, setCalls] = useState<ToolCall[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const seenCalls = useRef(new Set<string>());

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let handDetectionTimer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    let disposed = false;

    const url = new URL(BRIDGE_URL);
    url.searchParams.set("role", "viewer");
    url.searchParams.set("client_id", `web-${Math.random().toString(36).slice(2, 8)}`);

    const updateTelemetry = (frame: Telemetry) => {
      setTelemetry(frame);
      if (handDetectionTimer) clearTimeout(handDetectionTimer);
      const age = Math.max(0, Date.now() - frame.server_ts);
      const freshFor = Math.max(0, HAND_DETECTION_TIMEOUT_MS - age);
      setHandDetected(freshFor > 0);
      if (freshFor > 0) {
        handDetectionTimer = setTimeout(() => {
          setHandDetected(false);
          handDetectionTimer = null;
        }, freshFor);
      }
    };

    const recordCall = (frame: FunctionCallFrame) => {
      if (seenCalls.current.has(frame.call_id)) return;
      seenCalls.current.add(frame.call_id);
      setCounts((previous) => ({
        ...previous,
        [frame.name]: (previous[frame.name] ?? 0) + 1,
      }));
      setCalls((previous) =>
        [
          {
            callId: frame.call_id,
            name: frame.name,
            args: frame.args ?? {},
            origin: frame.origin ?? "gemini",
            at: frame.server_ts ?? Date.now(),
          },
          ...previous,
        ].slice(0, LOG_LIMIT),
      );
    };

    const connect = () => {
      if (disposed) return;
      setStatus("connecting");
      socket = new WebSocket(url.toString());

      socket.onopen = () => {
        attempts = 0;
        setStatus("online");
      };

      socket.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        let frame: Telemetry | FunctionCallFrame | StateSnapshotFrame;
        try {
          frame = JSON.parse(event.data);
        } catch {
          return;
        }
        if (frame.type === "telemetry") {
          updateTelemetry(frame);
        } else if (frame.type === "state_snapshot") {
          if (frame.telemetry) updateTelemetry(frame.telemetry);
        } else if (frame.type === "function_call") {
          recordCall(frame);
        }
      };

      socket.onclose = () => {
        if (disposed) return;
        setStatus("reconnecting");
        const delay = Math.min(5000, 500 * 2 ** attempts);
        attempts += 1;
        retryTimer = setTimeout(connect, delay);
      };

      socket.onerror = () => socket?.close();
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (handDetectionTimer) clearTimeout(handDetectionTimer);
      socket?.close();
    };
  }, []);

  return { status, telemetry, handDetected, calls, counts };
}

function formatClock(timestamp: number) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function PointerCursor({
  x,
  y,
  pinched,
  tone = "sky",
}: {
  x: number;
  y: number;
  pinched: boolean;
  tone?: "sky" | "amber";
}) {
  const open =
    tone === "amber"
      ? "h-9 w-9 border-2 border-amber-300/90 bg-amber-300/10 shadow-[0_0_28px_rgba(251,191,36,0.45)]"
      : "h-9 w-9 border-2 border-sky-400/90 bg-sky-400/10 shadow-[0_0_28px_rgba(56,189,248,0.55)]";
  const closed =
    tone === "amber"
      ? "h-3 w-3 border border-amber-100 bg-amber-300 shadow-[0_0_18px_4px_rgba(251,191,36,0.8)]"
      : "h-3 w-3 border border-sky-200 bg-sky-300 shadow-[0_0_18px_4px_rgba(125,211,252,0.85)]";

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed z-50 transition-[left,top] duration-100 ease-linear"
      style={{ left: `${x * 100}%`, top: `${y * 100}%` }}
    >
      <div
        className={`-translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-150 ease-out ${
          pinched ? closed : open
        }`}
      />
    </div>
  );
}

export default function SurgicalDashboard() {
  const { status, telemetry, handDetected, calls, counts } = useBridge();
  const [selected, setSelected] = useState<StudyId>("chest");
  const [scale, setScale] = useState(MIN_ZOOM);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [hovered, setHovered] = useState<StudyId | null>(null);
  const scaleRef = useRef(MIN_ZOOM);
  const panRef = useRef<Point>({ x: 0, y: 0 });
  const zoomGesture = useRef<ZoomGesture | null>(null);
  const pinchEdge = useRef({ primary: false, secondary: false });
  const stageRef = useRef<HTMLDivElement>(null);
  const imageRef = useRef<HTMLImageElement>(null);
  const lastPrimaryPinch = useRef<PinchMark | null>(null);
  const lastSecondaryPinch = useRef<PinchMark | null>(null);
  const selectedRef = useRef(selected);
  const swipeGesture = useRef<{
    samples: SwipeSample[];
    fired: boolean;
    idleSince: number | null;
  } | null>(null);
  selectedRef.current = selected;

  const toolNames = useMemo(() => {
    const observed = Object.keys(counts).filter((name) => !KNOWN_TOOLS.includes(name));
    return [...KNOWN_TOOLS, ...observed];
  }, [counts]);
  const totalCalls = useMemo(
    () => Object.values(counts).reduce((sum, value) => sum + value, 0),
    [counts],
  );
  const maxCount = Math.max(1, ...toolNames.map((name) => counts[name] ?? 0));
  const pinchActive = telemetry?.pinch_active ?? false;
  const secondHand =
    telemetry != null &&
    typeof telemetry.x2 === "number" &&
    typeof telemetry.y2 === "number" &&
    typeof telemetry.pinch2_active === "boolean";
  const secondPinch = Boolean(secondHand && telemetry?.pinch2_active);
  const study = STUDIES.find((item) => item.id === selected) ?? STUDIES[0];
  const zoomPercent = Math.round(scale * 100);
  const pointerPositions: Point[] = telemetry
    ? [
        { x: telemetry.x, y: telemetry.y },
        ...(typeof telemetry.x2 === "number" && typeof telemetry.y2 === "number"
          ? [{ x: telemetry.x2, y: telemetry.y2 }]
          : []),
      ]
    : [];
  const cursorNearEdge =
    handDetected &&
    pointerPositions.some(
      ({ x, y }) =>
        x <= EDGE_WARNING_THRESHOLD ||
        x >= 1 - EDGE_WARNING_THRESHOLD ||
        y <= EDGE_WARNING_THRESHOLD ||
        y >= 1 - EDGE_WARNING_THRESHOLD,
    );
  const frameFade = cursorNearEdge ? "rgba(248, 113, 113, 0.22)" : "rgba(56, 189, 248, 0.18)";
  const frameEdge = cursorNearEdge ? "rgba(248, 113, 113, 0.32)" : "rgba(125, 211, 252, 0.28)";

  const focusStudy = (id: StudyId | null) => {
    if (!id) return;
    setSelected(id);
    zoomGesture.current = null;
    scaleRef.current = MIN_ZOOM;
    panRef.current = { x: 0, y: 0 };
    setScale(MIN_ZOOM);
    setPan({ x: 0, y: 0 });
  };

  const showView = (next: { scale: number; x: number; y: number }) => {
    const nextPan = { x: next.x, y: next.y };
    scaleRef.current = next.scale;
    panRef.current = nextPan;
    setScale(next.scale);
    setPan(nextPan);
  };

  const zoomFullyAt = (x: number, y: number) => {
    const stage = stageRef.current?.getBoundingClientRect();
    const layout = imageRef.current ? imageLayout(imageRef.current) : null;
    if (!stage || !layout || stage.width <= 0 || stage.height <= 0) return false;
    const px = x * window.innerWidth;
    const py = y * window.innerHeight;
    const overStage =
      px >= stage.left && px <= stage.right && py >= stage.top && py <= stage.bottom;
    if (!overStage) return false;
    zoomGesture.current = null;
    if (scaleRef.current >= MAX_ZOOM - 0.05) {
      showView({ scale: MIN_ZOOM, x: 0, y: 0 });
      return true;
    }
    const safeScale = scaleRef.current || MIN_ZOOM;
    const focal = {
      x: clamp(px, stage.left, stage.right),
      y: clamp(py, stage.top, stage.bottom),
    };
    showView(
      zoomAtFocal(
        {
          baseline: 1,
          scale: MAX_ZOOM,
          anchorX: clamp(
            (focal.x - layout.centerX - panRef.current.x) / safeScale,
            -layout.width / 2,
            layout.width / 2,
          ),
          anchorY: clamp(
            (focal.y - layout.centerY - panRef.current.y) / safeScale,
            -layout.height / 2,
            layout.height / 2,
          ),
        },
        focal,
        1,
        layout,
        stage,
      ),
    );
    return true;
  };

  const advanceStudy = (direction: -1 | 1) => {
    const current = selectedRef.current;
    const index = STUDIES.findIndex((item) => item.id === current);
    if (index < 0) return;
    const next = STUDIES[(index + direction + STUDIES.length) % STUDIES.length];
    if (!next || next.id === current) return;
    selectedRef.current = next.id;
    lastPrimaryPinch.current = null;
    lastSecondaryPinch.current = null;
    focusStudy(next.id);
  };

  const appendSwipe = (x: number, y: number) => {
    const gesture = swipeGesture.current;
    if (!gesture || gesture.fired) return;
    const now = performance.now();
    gesture.samples.push({ x, y, at: now });
    const cutoff = now - SWIPE_WINDOW_MS;
    while (gesture.samples.length > SWIPE_MIN_SAMPLES && gesture.samples[0].at < cutoff) {
      gesture.samples.shift();
    }
    const direction = classifyVerticalSwipe(
      gesture.samples,
      window.innerWidth,
      window.innerHeight,
    );
    if (direction == null) return;
    gesture.fired = true;
    gesture.samples = [];
    advanceStudy(direction);
  };

  const onPinchDown = (x: number, y: number, mark: { current: PinchMark | null }) => {
    const now = performance.now();
    const prior = mark.current;
    mark.current = { at: now, x, y };
    const moved = prior
      ? Math.hypot((x - prior.x) * window.innerWidth, (y - prior.y) * window.innerHeight)
      : Number.POSITIVE_INFINITY;
    const isDouble =
      prior != null && now - prior.at <= DOUBLE_PINCH_MS && moved <= DOUBLE_PINCH_SLOP_PX;
    if (isDouble && zoomFullyAt(x, y)) return;
    focusStudy(studyAtPoint(x, y));
  };

  useEffect(() => {
    if (!telemetry) return;
    const x2 = telemetry.x2;
    const y2 = telemetry.y2;
    const pinch2 = telemetry.pinch2_active;
    const second =
      typeof x2 === "number" && typeof y2 === "number" && typeof pinch2 === "boolean";
    const both = telemetry.pinch_active && second && pinch2 === true;

    if (both && second) {
      const distance = Math.hypot(
        (x2 - telemetry.x) * window.innerWidth,
        (y2 - telemetry.y) * window.innerHeight,
      );
      const stage = stageRef.current?.getBoundingClientRect();
      const layout = imageRef.current ? imageLayout(imageRef.current) : null;
      if (stage && layout && stage.width > 0 && stage.height > 0) {
        const focal = focalInStage(telemetry.x, telemetry.y, x2, y2, stage);
        if (!zoomGesture.current) {
          if (distance >= ZOOM_START_PX) {
            const safeScale = scaleRef.current || MIN_ZOOM;
            zoomGesture.current = {
              baseline: distance,
              scale: safeScale,
              anchorX: clamp(
                (focal.x - layout.centerX - panRef.current.x) / safeScale,
                -layout.width / 2,
                layout.width / 2,
              ),
              anchorY: clamp(
                (focal.y - layout.centerY - panRef.current.y) / safeScale,
                -layout.height / 2,
                layout.height / 2,
              ),
            };
          }
        } else {
          showView(zoomAtFocal(zoomGesture.current, focal, distance, layout, stage));
        }
      }
    } else {
      zoomGesture.current = null;
    }

    const stage = stageRef.current?.getBoundingClientRect();
    const secondPinchActive = second && pinch2 === true;
    const anyPinch = telemetry.pinch_active || secondPinchActive;
    const primaryOnly = telemetry.pinch_active && !secondPinchActive;
    const secondaryOnly = secondPinchActive && !telemetry.pinch_active;
    const primaryDown = telemetry.pinch_active && !pinchEdge.current.primary;
    const secondaryDown = secondPinchActive && !pinchEdge.current.secondary;
    const pinchPoint = primaryOnly
      ? { x: telemetry.x, y: telemetry.y }
      : secondaryOnly && typeof x2 === "number" && typeof y2 === "number"
        ? { x: x2, y: y2 }
        : null;

    if (
      swipeGesture.current?.fired &&
      swipeGesture.current.idleSince != null &&
      performance.now() - swipeGesture.current.idleSince >= SWIPE_RELEASE_MS
    ) {
      swipeGesture.current = null;
    }

    const startedOnFilm =
      pinchPoint != null &&
      (primaryDown || secondaryDown) &&
      swipeGesture.current == null &&
      pointerOverStage(pinchPoint.x, pinchPoint.y, stage);

    // Thumbnail focus and double-pinch zoom only see a pinch that is not already a film swipe.
    if (!both && swipeGesture.current == null) {
      if (primaryDown) onPinchDown(telemetry.x, telemetry.y, lastPrimaryPinch);
      if (secondaryDown && typeof x2 === "number" && typeof y2 === "number") {
        onPinchDown(x2, y2, lastSecondaryPinch);
      }
    }

    if (both) {
      if (!swipeGesture.current?.fired) swipeGesture.current = null;
    } else if (pinchPoint && swipeGesture.current) {
      swipeGesture.current.idleSince = null;
      appendSwipe(pinchPoint.x, pinchPoint.y);
    } else if (startedOnFilm && pinchPoint) {
      swipeGesture.current = {
        samples: [{ x: pinchPoint.x, y: pinchPoint.y, at: performance.now() }],
        fired: false,
        idleSince: null,
      };
    } else if (!anyPinch && swipeGesture.current) {
      if (!swipeGesture.current.fired) {
        swipeGesture.current = null;
      } else {
        const now = performance.now();
        if (swipeGesture.current.idleSince == null) swipeGesture.current.idleSince = now;
        if (now - swipeGesture.current.idleSince >= SWIPE_RELEASE_MS) swipeGesture.current = null;
      }
    }

    pinchEdge.current = {
      primary: telemetry.pinch_active,
      secondary: Boolean(second && pinch2),
    };

    const nextHover =
      studyAtPoint(telemetry.x, telemetry.y) ?? (second ? studyAtPoint(x2, y2) : null);
    setHovered((current) => (current === nextHover ? current : nextHover));
  }, [telemetry]);

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden bg-[#05070d] text-slate-200">
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 bg-[#070b14]/95 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center rounded-lg border border-sky-500/40 bg-sky-500/10">
            <svg viewBox="0 0 24 24" className="h-4 w-4 text-sky-400" aria-hidden>
              <path
                d="M12 4v16M4 12h16"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                fill="none"
              />
            </svg>
          </span>
          <div>
            <h1 className="text-xs font-semibold tracking-[0.2em] text-slate-100 uppercase sm:text-sm">
              Touchless Surgical Assistant
            </h1>
            <p className="text-[10px] tracking-[0.18em] text-slate-500 uppercase">
              Sterile interface · telemetry bridge
            </p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          <dl className="hidden items-center gap-4 font-mono text-[11px] text-slate-500 sm:flex">
            <div className="flex items-center gap-1.5">
              <dt>X</dt>
              <dd className="w-12 text-slate-200">
                {telemetry ? telemetry.x.toFixed(3) : "—"}
              </dd>
            </div>
            <div className="flex items-center gap-1.5">
              <dt>Y</dt>
              <dd className="w-12 text-slate-200">
                {telemetry ? telemetry.y.toFixed(3) : "—"}
              </dd>
            </div>
            <span
              className={`rounded border px-2 py-0.5 tracking-[0.18em] transition-colors ${
                pinchActive
                  ? "border-sky-400/70 bg-sky-400/20 text-sky-200"
                  : "border-slate-700/80 text-slate-600"
              }`}
            >
              PINCH
            </span>
            {secondHand && (
              <span
                className={`rounded border px-2 py-0.5 tracking-[0.18em] transition-colors ${
                  secondPinch
                    ? "border-amber-300/70 bg-amber-300/20 text-amber-100"
                    : "border-slate-700/80 text-slate-600"
                }`}
              >
                PINCH 2
              </span>
            )}
          </dl>

          <span
            className={`flex items-center gap-2 rounded-full border px-3 py-1 text-[10px] font-medium tracking-[0.18em] uppercase ${
              status === "online"
                ? "border-sky-500/40 bg-sky-500/10 text-sky-300"
                : "border-amber-500/30 bg-amber-500/10 text-amber-300"
            }`}
          >
            <span className="relative flex h-1.5 w-1.5">
              {status === "online" && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400/70" />
              )}
              <span
                className={`relative inline-flex h-1.5 w-1.5 rounded-full ${
                  status === "online" ? "bg-sky-400" : "bg-amber-400"
                }`}
              />
            </span>
            {status === "online" ? "Bridge online" : status === "connecting" ? "Connecting" : "Reconnecting"}
          </span>
        </div>
      </header>

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <main className="flex min-h-0 min-w-0 flex-1 flex-col md:flex-row">
          <nav
            aria-label="Studies"
            className="flex shrink-0 gap-2 overflow-x-auto border-b border-slate-800/80 bg-[#070b14] p-3 md:w-44 md:flex-col md:overflow-y-auto md:border-r md:border-b-0"
          >
            <div className="hidden shrink-0 md:block">
              <h2 className="text-[10px] font-semibold tracking-[0.22em] text-slate-400 uppercase">
                Studies
              </h2>
              <p className="mt-1 text-[10px] leading-relaxed tracking-wider text-slate-600 uppercase">
                Pinch a film to focus it
              </p>
            </div>
            {STUDIES.map((item) => {
              const active = item.id === selected;
              const underCursor = item.id === hovered;
              return (
                <button
                  key={item.id}
                  type="button"
                  data-study={item.id}
                  aria-pressed={active}
                  onClick={() => focusStudy(item.id)}
                  className={`flex w-28 shrink-0 flex-col gap-2 rounded-lg border p-2 text-left transition-colors md:w-auto ${
                    active
                      ? "border-sky-400/80 bg-sky-400/10"
                      : underCursor
                        ? "border-sky-400/40 bg-slate-900/80"
                        : "border-slate-800 bg-slate-950/40 hover:border-slate-700"
                  }`}
                >
                  <img
                    src={item.src}
                    alt=""
                    draggable={false}
                    className="h-16 w-full rounded bg-black object-cover"
                  />
                  <span className="flex flex-col">
                    <span className="text-xs text-slate-100">{item.label}</span>
                    <span className="text-[10px] tracking-wider text-slate-500 uppercase">
                      {item.detail}
                    </span>
                  </span>
                </button>
              );
            })}
          </nav>

          <section className="flex h-full min-h-[320px] min-w-0 flex-1 flex-col overflow-hidden bg-[#060a12] md:m-3 md:rounded-xl md:border md:border-slate-800/80 md:shadow-[inset_0_0_120px_rgba(14,165,233,0.06)]">
            <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-slate-800/70 px-4 py-2.5">
              <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] text-slate-500 uppercase">
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500" />
                {study.label} study
              </div>
              <span className="text-right">
                <span className="block font-mono text-[10px] tracking-wider text-slate-400">
                  {zoomPercent}% · {Math.round(MIN_ZOOM * 100)}–{Math.round(MAX_ZOOM * 100)}
                </span>
                <span className="mt-0.5 block text-[9px] tracking-[0.14em] text-slate-600 uppercase">
                  Pinch twice to zoom · pinch and swipe up or down to change
                </span>
              </span>
            </div>

            <div
              ref={stageRef}
              className="scan-grid relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_50%_45%,rgba(14,165,233,0.10),transparent_60%)]"
            >
              <span className="pointer-events-none absolute top-4 left-4 h-6 w-6 border-t-2 border-l-2 border-sky-500/30" />
              <span className="pointer-events-none absolute top-4 right-4 h-6 w-6 border-t-2 border-r-2 border-sky-500/30" />
              <span className="pointer-events-none absolute bottom-4 left-4 h-6 w-6 border-b-2 border-l-2 border-sky-500/30" />
              <span className="pointer-events-none absolute right-4 bottom-4 h-6 w-6 border-b-2 border-r-2 border-sky-500/30" />

              <img
                ref={imageRef}
                src={study.src}
                alt={`${study.label} X-ray`}
                draggable={false}
                className="max-h-full max-w-full origin-center object-contain select-none"
                style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})` }}
              />
            </div>
          </section>
        </main>

        <aside className="flex min-h-0 w-full shrink-0 flex-col border-t border-slate-800/80 bg-[#070b14] md:w-[360px] md:border-t-0 md:border-l">
          <div className="border-b border-slate-800/80 px-4 py-3">
            <div className="flex items-center justify-between">
              <h2 className="text-[10px] font-semibold tracking-[0.22em] text-slate-400 uppercase">
                Surgical log
              </h2>
              <span className="font-mono text-[10px] text-slate-600">
                {totalCalls} {totalCalls === 1 ? "call" : "calls"}
              </span>
            </div>
            <p className="mt-1 text-[10px] tracking-wider text-slate-600 uppercase">
              Gemini function calls, newest first
            </p>
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
            {calls.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-800 px-3 py-6 text-center">
                <p className="text-xs text-slate-500">No tool calls yet</p>
                <p className="mt-1 text-[10px] leading-relaxed text-slate-600">
                  set_pointer and set_pinch calls appear here as the Gemini bridge emits them.
                </p>
              </div>
            ) : (
              <ul className="space-y-2">
                {calls.map((call) => (
                  <li
                    key={call.callId}
                    className="rounded-lg border border-slate-800/80 bg-slate-900/40 px-3 py-2"
                  >
                    <div className="flex items-baseline justify-between gap-2">
                      <span className="font-mono text-xs text-sky-300">{call.name}</span>
                      <time className="font-mono text-[10px] text-slate-600">
                        {formatClock(call.at)}
                      </time>
                    </div>
                    <p className="mt-1 break-all font-mono text-[11px] leading-snug text-slate-500">
                      {JSON.stringify(call.args)}
                    </p>
                    <p className="mt-1 text-[9px] tracking-[0.16em] text-slate-600 uppercase">
                      {call.origin} · {call.callId.slice(0, 8)}
                    </p>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="border-t border-slate-800/80 px-4 py-3">
            <h2 className="text-[10px] font-semibold tracking-[0.22em] text-slate-400 uppercase">
              Tool count
            </h2>
            <ul className="mt-2 space-y-1.5">
              {toolNames.map((name) => {
                const count = counts[name] ?? 0;
                return (
                  <li key={name} className="flex items-center justify-between gap-3">
                    <span className="font-mono text-[11px] text-slate-400">{name}</span>
                    <span className="flex items-center gap-2">
                      <span className="h-1 w-24 overflow-hidden rounded-full bg-slate-800">
                        <span
                          className="block h-full rounded-full bg-sky-500/70 transition-all duration-300"
                          style={{ width: `${Math.round((count / maxCount) * 100)}%` }}
                        />
                      </span>
                      <span className="w-6 text-right font-mono text-[11px] text-slate-300">
                        {count}
                      </span>
                    </span>
                  </li>
                );
              })}
            </ul>
            <p className="mt-3 flex items-center gap-2 text-[9px] tracking-[0.18em] text-slate-600 uppercase">
              <span
                className={`h-1.5 w-1.5 rounded-full ${status === "online" ? "bg-sky-500" : "bg-amber-500"}`}
              />
              {BRIDGE_URL}
            </p>
          </div>
        </aside>
      </div>

      {handDetected && telemetry && (
        <PointerCursor x={telemetry.x} y={telemetry.y} pinched={telemetry.pinch_active} />
      )}
      {handDetected && secondHand && telemetry && (
        <PointerCursor
          x={telemetry.x2 ?? 0}
          y={telemetry.y2 ?? 0}
          pinched={secondPinch}
          tone="amber"
        />
      )}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-0 z-40 transition-[opacity,box-shadow] duration-500 ease-out"
        style={{
          opacity: handDetected ? 1 : 0,
          background: `radial-gradient(ellipse at center, transparent 46%, ${frameFade} 100%)`,
          boxShadow: `inset 0 0 110px 24px ${frameEdge}`,
        }}
      />
    </div>
  );
}
