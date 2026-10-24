"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const BRIDGE_URL = process.env.NEXT_PUBLIC_BRIDGE_URL ?? "ws://localhost:8000/ws";
const LOG_LIMIT = 50;
const KNOWN_TOOLS = ["set_pointer", "set_pinch"];

const STUDIES = [
  { id: "chest", label: "Chest", detail: "PA chest", src: "/studies/chest.png" },
  { id: "hand", label: "Hand", detail: "Hand", src: "/studies/hand.png" },
  { id: "teeth", label: "Teeth", detail: "Dental", src: "/studies/teeth.png" },
] as const;

type StudyId = (typeof STUDIES)[number]["id"];

const MIN_ZOOM = 1;
const MAX_ZOOM = 4;
const ZOOM_START_PX = 48;

function clampZoom(scale: number) {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, scale));
}

function studyAtPoint(x: number, y: number): StudyId | null {
  const element = document.elementFromPoint(x * window.innerWidth, y * window.innerHeight);
  const id = element?.closest("[data-study]")?.getAttribute("data-study");
  if (id === "chest" || id === "hand" || id === "teeth") return id;
  return null;
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
  const [calls, setCalls] = useState<ToolCall[]>([]);
  const [counts, setCounts] = useState<Record<string, number>>({});
  const seenCalls = useRef(new Set<string>());

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retryTimer: ReturnType<typeof setTimeout> | null = null;
    let attempts = 0;
    let disposed = false;

    const url = new URL(BRIDGE_URL);
    url.searchParams.set("role", "viewer");
    url.searchParams.set("client_id", `web-${Math.random().toString(36).slice(2, 8)}`);

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
          setTelemetry(frame);
        } else if (frame.type === "state_snapshot") {
          if (frame.telemetry) setTelemetry(frame.telemetry);
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
      socket?.close();
    };
  }, []);

  return { status, telemetry, calls, counts };
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
  const { status, telemetry, calls, counts } = useBridge();
  const [selected, setSelected] = useState<StudyId>("chest");
  const [scale, setScale] = useState(MIN_ZOOM);
  const [hovered, setHovered] = useState<StudyId | null>(null);
  const scaleRef = useRef(MIN_ZOOM);
  const zoomGesture = useRef<{ baseline: number; scale: number } | null>(null);
  const pinchEdge = useRef({ primary: false, secondary: false });

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

  const focusStudy = (id: StudyId | null) => {
    if (!id) return;
    setSelected(id);
    scaleRef.current = MIN_ZOOM;
    setScale(MIN_ZOOM);
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
      if (!zoomGesture.current) {
        if (distance >= ZOOM_START_PX) {
          zoomGesture.current = { baseline: distance, scale: scaleRef.current };
        }
      } else {
        const next = clampZoom(
          zoomGesture.current.scale * (distance / zoomGesture.current.baseline),
        );
        scaleRef.current = next;
        setScale(next);
      }
    } else {
      zoomGesture.current = null;
      if (telemetry.pinch_active && !pinchEdge.current.primary) {
        focusStudy(studyAtPoint(telemetry.x, telemetry.y));
      }
      if (second && pinch2 && !pinchEdge.current.secondary) {
        focusStudy(studyAtPoint(x2, y2));
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
            <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-2.5">
              <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] text-slate-500 uppercase">
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500" />
                {study.label} study
              </div>
              <span className="font-mono text-[10px] tracking-wider text-slate-400">
                {zoomPercent}% · {Math.round(MIN_ZOOM * 100)}–{Math.round(MAX_ZOOM * 100)}
              </span>
            </div>

            <div className="scan-grid relative flex min-h-0 flex-1 items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_50%_45%,rgba(14,165,233,0.10),transparent_60%)]">
              <span className="pointer-events-none absolute top-4 left-4 h-6 w-6 border-t-2 border-l-2 border-sky-500/30" />
              <span className="pointer-events-none absolute top-4 right-4 h-6 w-6 border-t-2 border-r-2 border-sky-500/30" />
              <span className="pointer-events-none absolute bottom-4 left-4 h-6 w-6 border-b-2 border-l-2 border-sky-500/30" />
              <span className="pointer-events-none absolute right-4 bottom-4 h-6 w-6 border-b-2 border-r-2 border-sky-500/30" />

              <img
                src={study.src}
                alt={`${study.label} X-ray`}
                draggable={false}
                className="max-h-full max-w-full object-contain select-none"
                style={{ transform: `scale(${scale})` }}
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

      {telemetry && (
        <PointerCursor x={telemetry.x} y={telemetry.y} pinched={telemetry.pinch_active} />
      )}
      {secondHand && telemetry && (
        <PointerCursor
          x={telemetry.x2 ?? 0}
          y={telemetry.y2 ?? 0}
          pinched={secondPinch}
          tone="amber"
        />
      )}
    </div>
  );
}
