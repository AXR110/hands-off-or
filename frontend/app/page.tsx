"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { emptySpeechLog, applySpeechLog, speechLogStatusLabel, SpeechLogView, recordingFrameClass, armManualRecording } from "./speech-log-state";

const BRIDGE_URL = process.env.NEXT_PUBLIC_BRIDGE_URL ?? "ws://localhost:8000/ws";
const LOG_LIMIT = 50;
const HAND_DETECTION_TIMEOUT_MS = 500;
const VOICE_STATUS_TIMEOUT_MS = 3000;
// Hue eases across this band instead of flipping at a hard edge.
const EDGE_BLEND_START = 0.14;
const EDGE_BLEND_END = 0.025;
const FRAME_BLUE = { h: 198.44, s: 93.2, l: 59.61, a: 0.3 };
const FRAME_RED = { h: 360, s: 90.6, l: 70.78, a: 0.42 };

const STUDIES = [
  { id: "chest", label: "Chest", detail: "PA chest", src: "/studies/chest.png" },
  { id: "hand", label: "Hand", detail: "Hand", src: "/studies/hand.png" },
  { id: "teeth", label: "Teeth", detail: "Dental", src: "/studies/teeth.png" },
] as const;

type StudyId = string;
type Study = {
  id: StudyId;
  label: string;
  detail: string;
  src: string;
};

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
const SUPPORTED_IMAGE_TYPES = new Set([
  "image/png",
  "image/jpeg",
  "image/webp",
  "image/gif",
  "image/bmp",
]);
const SUPPORTED_IMAGE_EXTENSIONS = new Set(["png", "jpg", "jpeg", "webp", "gif", "bmp"]);
const IMAGE_FILE_ACCEPT =
  ".png,.jpg,.jpeg,.webp,.gif,.bmp,image/png,image/jpeg,image/webp,image/gif,image/bmp";

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

type PickedImageFile = { name: string; type: string; data_base64: string };
type PickedImagesFrame = {
  cancelled?: boolean;
  error?: string;
  files?: PickedImageFile[];
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

function smoothstep(value: number) {
  const t = clamp(value, 0, 1);
  return t * t * (3 - 2 * t);
}

function edgeProximity(x: number, y: number) {
  const distance = Math.min(x, 1 - x, y, 1 - y);
  return (EDGE_BLEND_START - distance) / (EDGE_BLEND_START - EDGE_BLEND_END);
}

function frameTint(amount: number) {
  const t = smoothstep(amount);
  const h = FRAME_BLUE.h + (FRAME_RED.h - FRAME_BLUE.h) * t;
  const s = FRAME_BLUE.s + (FRAME_RED.s - FRAME_BLUE.s) * t;
  const l = FRAME_BLUE.l + (FRAME_RED.l - FRAME_BLUE.l) * t;
  const a = FRAME_BLUE.a + (FRAME_RED.a - FRAME_BLUE.a) * t;
  return `hsla(${h.toFixed(2)}, ${s.toFixed(1)}%, ${l.toFixed(1)}%, ${a.toFixed(3)})`;
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
  return id ?? null;
}

function fileAddAtPoint(x: number, y: number) {
  return buttonAtPoint("[data-file-add]", x, y) != null;
}

function recordAtPoint(x: number, y: number) {
  return buttonAtPoint("[data-start-recording]", x, y) != null;
}

// A pinch is a click when the pointer is on the control, with a little slack for aim.
function buttonAtPoint(selector: string, x: number, y: number): HTMLButtonElement | null {
  const px = x * window.innerWidth;
  const py = y * window.innerHeight;
  const direct = document.elementFromPoint(px, py)?.closest(selector);
  if (direct instanceof HTMLButtonElement) return direct;
  const pad = 16;
  for (const node of document.querySelectorAll(selector)) {
    if (!(node instanceof HTMLButtonElement)) continue;
    const rect = node.getBoundingClientRect();
    if (
      px >= rect.left - pad &&
      px <= rect.right + pad &&
      py >= rect.top - pad &&
      py <= rect.bottom + pad
    ) {
      return node;
    }
  }
  return null;
}

function isSupportedImage(file: File) {
  const extension = file.name.split(".").pop()?.toLowerCase();
  return (
    SUPPORTED_IMAGE_TYPES.has(file.type) ||
    (extension != null && SUPPORTED_IMAGE_EXTENSIONS.has(extension))
  );
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

type FunctionResultFrame = {
  type: "function_result";
  call_id: string;
  name?: string;
  ok: boolean;
  result?: Record<string, unknown> | null;
  server_ts?: number;
};

type ToolCall = {
  callId: string;
  name: string;
  args: Record<string, unknown>;
  origin: string;
  at: number;
  result?: Record<string, unknown> | null;
  ok?: boolean;
};

function useBridge() {
  const [status, setStatus] = useState<ConnectionStatus>("connecting");
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [handDetected, setHandDetected] = useState(false);
  const [calls, setCalls] = useState<ToolCall[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const [speechLog, setSpeechLog] = useState<SpeechLogView>(emptySpeechLog());
  const [voiceLive, setVoiceLive] = useState(false);
  const seenCalls = useRef(new Set<string>());
  const socketRef = useRef<WebSocket | null>(null);
  const pickedImagesHandler = useRef<(frame: PickedImagesFrame) => void>(() => {});

  useEffect(() => {
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let handDetectionTimer: ReturnType<typeof setTimeout> | null = null;
    let voiceStatusTimer: ReturnType<typeof setTimeout> | null = null;
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
      socketRef.current = new WebSocket(url.toString());

      socketRef.current.onopen = () => {
        attempts = 0;
        setStatus("online");
      };

      socketRef.current.onmessage = (event) => {
        if (typeof event.data !== "string") return;
        let frame: Telemetry | FunctionCallFrame | StateSnapshotFrame;
        try {
          frame = JSON.parse(event.data);
        } catch {
          return;
        }
        if ((frame as any).type === "speech_log") {
          setSpeechLog((prev) => applySpeechLog(prev, frame as any));
        }
        if ((frame as { type?: string }).type === "picked_images") {
          pickedImagesHandler.current(frame as PickedImagesFrame);
        }
        if ((frame as { type?: string }).type === "voice_status") {
          const active = (frame as { active?: boolean }).active !== false;
          if (voiceStatusTimer) clearTimeout(voiceStatusTimer);
          voiceStatusTimer = null;
          if (!active) {
            setVoiceLive(false);
          } else {
            setVoiceLive(true);
            voiceStatusTimer = setTimeout(() => {
              setVoiceLive(false);
              voiceStatusTimer = null;
            }, VOICE_STATUS_TIMEOUT_MS);
          }
        }
        if (frame.type === "telemetry") {
          updateTelemetry(frame);
        } else if (frame.type === "state_snapshot") {
          if (frame.telemetry) updateTelemetry(frame.telemetry);
        } else if (frame.type === "function_call") {
          recordCall(frame);
        } else if ((frame as { type?: string }).type === "function_result") {
          const resultFrame = frame as FunctionResultFrame;
          setCalls((previous) =>
            previous.map((call) =>
              call.callId === resultFrame.call_id
                ? { ...call, result: resultFrame.result ?? null, ok: resultFrame.ok }
                : call,
            ),
          );
        }
      };

      socketRef.current.onclose = () => {
        if (disposed) return;
        setStatus("reconnecting");
        const delay = Math.min(5000, 500 * 2 ** attempts);
        attempts += 1;
        retryTimer = setTimeout(connect, delay);
      };

      socketRef.current.onerror = () => socketRef.current?.close();
    };

    connect();

    return () => {
      disposed = true;
      if (retryTimer) clearTimeout(retryTimer);
      if (handDetectionTimer) clearTimeout(handDetectionTimer);
      if (voiceStatusTimer) clearTimeout(voiceStatusTimer);
      socketRef.current?.close();
    };
  }, []);

  const startRecording = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return;
    }
    setSpeechLog((prev) => armManualRecording(prev));
    socketRef.current.send(JSON.stringify({ type: "start_recording" }));
  };

  const requestImagePick = () => {
    if (!socketRef.current || socketRef.current.readyState !== WebSocket.OPEN) {
      return false;
    }
    socketRef.current.send(JSON.stringify({ type: "pick_images" }));
    return true;
  };

  return {
    status,
    telemetry,
    handDetected,
    calls,
    counts,
    speechLog,
    voiceLive,
    startRecording,
    requestImagePick,
    pickedImagesHandler,
  };
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
  const {
    status,
    telemetry,
    handDetected,
    speechLog,
    voiceLive,
    startRecording,
    requestImagePick,
    pickedImagesHandler,
  } = useBridge();
  const [selected, setSelected] = useState<StudyId>("chest");
  const [uploadedStudies, setUploadedStudies] = useState<Study[]>([]);
  const [fileNotice, setFileNotice] = useState("");
  const [isDraggingFiles, setIsDraggingFiles] = useState(false);
  const [scale, setScale] = useState(MIN_ZOOM);
  const [pan, setPan] = useState<Point>({ x: 0, y: 0 });
  const [hovered, setHovered] = useState<StudyId | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadedUrls = useRef(new Set<string>());
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
  const studies: Study[] = [...STUDIES, ...uploadedStudies];
  const studiesRef = useRef<Study[]>([...STUDIES]);

  const pinchActive = telemetry?.pinch_active ?? false;
  const secondHand =
    telemetry != null &&
    typeof telemetry.x2 === "number" &&
    typeof telemetry.y2 === "number" &&
    typeof telemetry.pinch2_active === "boolean";
  const secondPinch = Boolean(secondHand && telemetry?.pinch2_active);
  const study = studies.find((item) => item.id === selected) ?? studies[0];
  const zoomPercent = Math.round(scale * 100);
  const pointerPositions: Point[] = telemetry
    ? [
        { x: telemetry.x, y: telemetry.y },
        ...(typeof telemetry.x2 === "number" && typeof telemetry.y2 === "number"
          ? [{ x: telemetry.x2, y: telemetry.y2 }]
          : []),
      ]
    : [];
  const edgeAmount = pointerPositions.reduce(
    (nearest, point) => Math.max(nearest, edgeProximity(point.x, point.y)),
    0,
  );
  const frameColor = frameTint(edgeAmount);
  const frameMask = [
    "linear-gradient(to right, #000 0%, transparent 72px)",
    "linear-gradient(to left, #000 0%, transparent 72px)",
    "linear-gradient(to bottom, #000 0%, transparent 72px)",
    "linear-gradient(to top, #000 0%, transparent 72px)",
  ].join(", ");

  const focusStudy = (id: StudyId | null) => {
    if (!id) return;
    selectedRef.current = id;
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
    const currentStudies = studiesRef.current;
    const index = currentStudies.findIndex((item) => item.id === current);
    if (index < 0) return;
    const next =
      currentStudies[(index + direction + currentStudies.length) % currentStudies.length];
    if (!next || next.id === current) return;
    lastPrimaryPinch.current = null;
    lastSecondaryPinch.current = null;
    focusStudy(next.id);
  };

  const imagePickPending = useRef(false);

  const openFilePicker = useCallback(() => {
    setFileNotice("Choose image files to add them to the image list.");
    fileInputRef.current?.click();
  }, []);

  const openImagesFromPinch = useCallback(() => {
    if (imagePickPending.current) return;
    if (!requestImagePick()) {
      setFileNotice("Pinch reached Add, but the bridge is not connected.");
      return;
    }
    imagePickPending.current = true;
    setFileNotice("Choose the images to add.");
  }, [requestImagePick]);

  const addFiles = (files: FileList | File[]) => {
    const selectedFiles = Array.from(files);
    if (selectedFiles.length === 0) return;

    const imageFiles = selectedFiles.filter(isSupportedImage);
    const skippedCount = selectedFiles.length - imageFiles.length;
    const additions = imageFiles.map((file) => {
      const src = URL.createObjectURL(file);
      uploadedUrls.current.add(src);
      return {
        id: `upload-${crypto.randomUUID()}`,
        label: file.name.replace(/\.[^.]+$/, "") || file.name,
        detail: "Imported image",
        src,
      };
    });

    if (additions.length > 0) {
      studiesRef.current = [...studiesRef.current, ...additions];
      setUploadedStudies((current) => [...current, ...additions]);
      focusStudy(additions[0].id);
    }

    setFileNotice(
      additions.length === 0
        ? "No supported images added. Choose PNG, JPEG, WebP, GIF, or BMP files."
        : skippedCount > 0
          ? `Added ${additions.length} image${additions.length === 1 ? "" : "s"}; skipped ${skippedCount} unsupported file${skippedCount === 1 ? "" : "s"}.`
          : `Added ${additions.length} image${additions.length === 1 ? "" : "s"}. Pinch or click a thumbnail to view it.`,
    );
  };

  pickedImagesHandler.current = (frame) => {
    imagePickPending.current = false;
    if (frame.cancelled) {
      setFileNotice("No images added.");
      return;
    }
    if (frame.error) {
      setFileNotice(frame.error);
      return;
    }
    const files = (frame.files ?? []).map((item) => {
      const binary = atob(item.data_base64);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {
        bytes[index] = binary.charCodeAt(index);
      }
      return new File([bytes], item.name, { type: item.type || "application/octet-stream" });
    });
    addFiles(files);
  };

  useEffect(() => {
    const handleShortcut = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "o") {
        const target = event.target;
        if (
          target instanceof HTMLElement &&
          (target.isContentEditable || ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName))
        ) {
          return;
        }
        event.preventDefault();
        openFilePicker();
      }
    };
    window.addEventListener("keydown", handleShortcut);
    return () => window.removeEventListener("keydown", handleShortcut);
  }, [openFilePicker]);

  useEffect(
    () => () => {
      for (const url of uploadedUrls.current) URL.revokeObjectURL(url);
      uploadedUrls.current.clear();
    },
    [],
  );

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
    if (buttonAtPoint("[data-file-add]", x, y)) {
      openImagesFromPinch();
      return;
    }
    const recordButton = buttonAtPoint("[data-start-recording]", x, y);
    if (recordButton) {
      recordButton.click();
      return;
    }
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

    const controlHover = (x: number, y: number) =>
      fileAddAtPoint(x, y) ? "add-files" : recordAtPoint(x, y) ? "record" : null;
    const nextHover =
      controlHover(telemetry.x, telemetry.y) ??
      studyAtPoint(telemetry.x, telemetry.y) ??
      (second ? (controlHover(x2, y2) ?? studyAtPoint(x2, y2)) : null);
    setHovered((current) => (current === nextHover ? current : nextHover));
  }, [telemetry]);

  return (
    <div className="relative flex h-dvh flex-col overflow-hidden bg-[#05070d] text-slate-200">
      {speechLog.recording && (
        <div data-recording-frame aria-hidden className={recordingFrameClass(true) ?? undefined} />
      )}
      <header className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-800/80 bg-[#070b14]/95 px-4 py-3 sm:px-6">
        <div className="flex items-center gap-3">
          <img
            src="/logo.jpg"
            alt=""
            className="h-12 w-12 shrink-0 object-contain mix-blend-screen"
          />
          <h1 className="text-2xl font-semibold tracking-[0.14em] text-slate-100 uppercase sm:text-3xl">
            Hands-Off OR
          </h1>
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

      <input
        ref={fileInputRef}
        type="file"
        accept={IMAGE_FILE_ACCEPT}
        multiple
        className="sr-only"
        tabIndex={-1}
        aria-label="Choose image files to add to the image viewer"
        onChange={(event) => {
          if (event.currentTarget.files) addFiles(event.currentTarget.files);
          event.currentTarget.value = "";
        }}
      />

      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <main className="flex min-h-0 min-w-0 flex-1 flex-col md:flex-row">
          <nav
            aria-label="Images"
            className="flex shrink-0 gap-2 overflow-x-auto border-b border-slate-800/80 bg-[#070b14] p-3 md:w-44 md:flex-col md:overflow-y-auto md:border-r md:border-b-0"
          >
            <div className="w-full shrink-0 md:space-y-2">
              <h2 className="hidden text-[10px] font-semibold tracking-[0.22em] text-slate-400 uppercase md:block">
                Images
              </h2>
              <button
                type="button"
                data-file-add
                onClick={() => openFilePicker()}
                className={`min-h-12 min-w-36 rounded-lg border px-4 py-3 text-xs font-semibold transition-colors focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-sky-400 md:w-full ${
                  hovered === "add-files"
                    ? "border-sky-200 bg-sky-400 text-slate-950 shadow-[0_0_24px_rgba(56,189,248,0.6)] ring-2 ring-sky-300/60"
                    : "border-sky-500/50 bg-sky-500/10 text-sky-200 hover:border-sky-400 hover:bg-sky-500/20"
                }`}
                aria-label="Add image files"
                title="Add images (Ctrl+O or Cmd+O)"
                aria-keyshortcuts="Control+O Meta+O"
              >
                + Add images
              </button>
              <p className="mt-1 hidden text-[10px] leading-relaxed tracking-wider text-slate-600 md:block">
                Pinch or click a film to view it
              </p>
            </div>
            {studies.map((item) => {
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
                {study.label} image
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
              onDragOver={(event) => {
                if (!Array.from(event.dataTransfer.types).includes("Files")) return;
                event.preventDefault();
                event.dataTransfer.dropEffect = "copy";
                setIsDraggingFiles(true);
              }}
              onDragLeave={(event) => {
                if (
                  !(event.relatedTarget instanceof Node) ||
                  !event.currentTarget.contains(event.relatedTarget)
                ) {
                  setIsDraggingFiles(false);
                }
              }}
              onDrop={(event) => {
                event.preventDefault();
                setIsDraggingFiles(false);
                addFiles(event.dataTransfer.files);
              }}
              className="scan-grid relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_50%_45%,rgba(14,165,233,0.10),transparent_60%)]"
            >
              <span className="pointer-events-none absolute top-4 left-4 h-6 w-6 border-t-2 border-l-2 border-sky-500/30" />
              <span className="pointer-events-none absolute top-4 right-4 h-6 w-6 border-t-2 border-r-2 border-sky-500/30" />
              <span className="pointer-events-none absolute bottom-4 left-4 h-6 w-6 border-b-2 border-l-2 border-sky-500/30" />
              <span className="pointer-events-none absolute right-4 bottom-4 h-6 w-6 border-b-2 border-r-2 border-sky-500/30" />

              <img
                ref={imageRef}
                src={study.src}
                alt={`${study.label} image`}
                draggable={false}
                className="max-h-full max-w-full origin-center object-contain select-none"
                style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${scale})` }}
              />
              {isDraggingFiles && (
                <div className="pointer-events-none absolute inset-3 z-10 grid place-items-center rounded-xl border-2 border-dashed border-sky-400 bg-sky-950/80 text-center">
                  <div>
                    <p className="text-sm font-medium text-sky-200">Drop images to add them</p>
                    <p className="mt-1 text-xs text-sky-100/70">
                      PNG, JPG, WebP, GIF, or BMP
                    </p>
                  </div>
                </div>
              )}
            </div>
            <div className="border-t border-slate-800/70 px-4 py-3">
              <p className="text-[10px] text-slate-500" aria-live="polite">
                {fileNotice ||
                  "Drop image files here, use Add to browse, or pinch an image thumbnail."}
              </p>
            </div>
          </section>
        </main>

        <aside className="flex min-h-0 w-full shrink-0 flex-col border-t border-slate-800/80 bg-[#070b14] md:w-[360px] md:border-t-0 md:border-l">
          <div className="border-b border-slate-800/80 px-4 py-4">
            <h2 className="text-2xl font-semibold tracking-[0.14em] text-slate-100 uppercase sm:text-3xl">
              Surgical log
            </h2>
          </div>

           <div className="min-h-0 flex-1 overflow-y-auto px-4 py-3">
            <div
              data-speech-status={speechLog.recording ? "recording" : "idle"}
              className="mb-3 rounded-lg border border-slate-800/80 bg-slate-900/40 px-3 py-2"
            >
              <div className="flex items-center gap-2 text-[10px] tracking-[0.18em] uppercase text-slate-400">
                <span
                  data-voice-dot={
                    speechLog.recording ? "log" : voiceLive ? "on" : "off"
                  }
                  className={`h-1.5 w-1.5 rounded-full ${
                    speechLog.recording
                      ? "voice-live-dot"
                      : voiceLive
                        ? "bg-sky-400"
                        : "bg-slate-600"
                  }`}
                />
                {speechLog.recording
                  ? speechLogStatusLabel(speechLog)
                  : voiceLive
                    ? "Listening for log"
                    : "Voice off"}
              </div>
              {speechLog.recording && (
                <p data-speech-live className="mt-1 font-mono text-xs text-sky-300">
                  {speechLog.liveText || "…"}
                </p>
              )}
               {speechLog.entries.length > 0 && (
                 <ul className="mt-2 space-y-2">
                   {speechLog.entries.map((entry, index) => (
                     <li
                       key={index}
                       data-speech-entry
                       className="rounded-lg border border-slate-800/80 bg-slate-900/40 px-3 py-2"
                     >
                       <div className="flex items-baseline justify-between gap-2">
                         <span className="font-mono text-xs text-sky-300 break-all">
                           {entry.text}
                         </span>
                         <time className="font-mono text-[10px] text-slate-600">
                           {formatClock(entry.at)}
                         </time>
                       </div>
                     </li>
                   ))}
                 </ul>
               )}
               <button
                 type="button"
                 data-start-recording
                 disabled={speechLog.recording}
                 onClick={startRecording}
                 className={`mt-3 flex min-h-14 w-full items-center justify-center rounded-xl border px-4 py-3 text-base font-semibold tracking-[0.22em] uppercase transition active:scale-[0.98] disabled:cursor-default disabled:opacity-40 ${
                   hovered === "record" && !speechLog.recording
                     ? "border-emerald-200 bg-emerald-400 text-slate-950 shadow-[0_0_24px_rgba(16,185,129,0.55)]"
                     : "border-emerald-400/50 bg-emerald-500/15 text-emerald-100 shadow-[0_0_28px_rgba(16,185,129,0.18)] hover:bg-emerald-500/25"
                 }`}
               >
                 Record
               </button>
             </div>
          </div>

          <div className="border-t border-slate-800/80 px-4 py-3">
            <p className="flex items-center gap-2 font-mono text-[10px] tracking-wide text-slate-500">
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
        data-voice-live={voiceLive ? "on" : "off"}
        data-log-hue={speechLog.recording ? "on" : "off"}
        className={`pointer-events-none fixed inset-0 z-40 ${
          speechLog.recording ? "voice-live-hue" : ""
        }`}
        style={{
          opacity: speechLog.recording || handDetected ? 1 : 0,
          background: speechLog.recording ? undefined : frameColor,
          transition: "opacity 500ms ease-out",
          maskImage: frameMask,
          WebkitMaskImage: frameMask,
          maskComposite: "add",
          WebkitMaskComposite: "source-over",
        }}
      />
    </div>
  );
}
