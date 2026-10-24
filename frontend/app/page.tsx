"use client";

import { useEffect, useMemo, useRef, useState } from "react";

const BRIDGE_URL = process.env.NEXT_PUBLIC_BRIDGE_URL ?? "ws://localhost:8000/ws";
const LOG_LIMIT = 50;
const KNOWN_TOOLS = ["set_pointer", "set_pinch"];

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

function PointerCursor({ telemetry }: { telemetry: Telemetry | null }) {
  if (!telemetry) return null;
  const pinched = telemetry.pinch_active;

  return (
    <div
      aria-hidden
      className="pointer-events-none fixed z-50 transition-[left,top] duration-100 ease-linear"
      style={{ left: `${telemetry.x * 100}%`, top: `${telemetry.y * 100}%` }}
    >
      <div
        className={`-translate-x-1/2 -translate-y-1/2 rounded-full transition-all duration-150 ease-out ${
          pinched
            ? "h-3 w-3 border border-sky-200 bg-sky-300 shadow-[0_0_18px_4px_rgba(125,211,252,0.85)]"
            : "h-9 w-9 border-2 border-sky-400/90 bg-sky-400/10 shadow-[0_0_28px_rgba(56,189,248,0.55)]"
        }`}
      />
    </div>
  );
}

export default function SurgicalDashboard() {
  const { status, telemetry, calls, counts } = useBridge();

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
        <main className="min-h-0 flex-1 p-3 sm:p-5">
          <section className="flex h-full min-h-[320px] flex-col overflow-hidden rounded-xl border border-slate-800/80 bg-[#060a12] shadow-[inset_0_0_120px_rgba(14,165,233,0.06)]">
            <div className="flex items-center justify-between border-b border-slate-800/70 px-4 py-2.5">
              <div className="flex items-center gap-2 text-[10px] tracking-[0.22em] text-slate-500 uppercase">
                <span className="h-1.5 w-1.5 rounded-full bg-sky-500" />
                Study viewport
              </div>
              <span className="font-mono text-[10px] tracking-wider text-slate-600">
                DICOM · 512 × 512
              </span>
            </div>

            <div className="scan-grid relative flex flex-1 items-center justify-center overflow-hidden bg-[radial-gradient(circle_at_50%_45%,rgba(14,165,233,0.10),transparent_60%)]">
              <span className="absolute top-4 left-4 h-6 w-6 border-t-2 border-l-2 border-sky-500/30" />
              <span className="absolute top-4 right-4 h-6 w-6 border-t-2 border-r-2 border-sky-500/30" />
              <span className="absolute bottom-4 left-4 h-6 w-6 border-b-2 border-l-2 border-sky-500/30" />
              <span className="absolute right-4 bottom-4 h-6 w-6 border-b-2 border-r-2 border-sky-500/30" />

              <div className="flex flex-col items-center gap-3 px-6 text-center">
                <div className="relative grid h-28 w-28 place-items-center rounded-full border border-sky-500/25">
                  <div className="absolute inset-4 rounded-full border border-sky-500/15" />
                  <div className="absolute h-px w-full bg-sky-500/20" />
                  <div className="absolute h-full w-px bg-sky-500/20" />
                  <div className="h-2 w-2 rounded-full bg-sky-400/80 shadow-[0_0_16px_rgba(56,189,248,0.8)]" />
                </div>
                <p className="text-sm text-slate-400">No study loaded</p>
                <p className="max-w-xs text-[11px] leading-relaxed text-slate-600">
                  The DICOM viewer mounts here. Pointer telemetry renders above every panel.
                </p>
              </div>
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

      <PointerCursor telemetry={telemetry} />
    </div>
  );
}
