export const SPEECH_LOG_LIMIT = 50;

export type SpeechLogEntry = { text: string; at: number };
export type SpeechLogView = { recording: boolean; liveText: string; entries: SpeechLogEntry[] };

export function emptySpeechLog(): SpeechLogView {
  return { recording: false, liveText: "", entries: [] };
}

export function speechLogStatusLabel(view: SpeechLogView): string {
  return view.recording ? "Recording" : "Listening for log";
}

/** Oldest saved log is 1. The list itself stays newest-first. */
export function speechLogChronologicalNumber(entryCount: number, newestFirstIndex: number): number {
  return entryCount - newestFirstIndex;
}

export function removeSpeechLogEntry(view: SpeechLogView, index: number): SpeechLogView {
  if (!Number.isInteger(index) || index < 0 || index >= view.entries.length) return view;
  return {
    recording: view.recording,
    liveText: view.liveText,
    entries: view.entries.filter((_, entryIndex) => entryIndex !== index),
  };
}

export function applySpeechLog(
  view: SpeechLogView,
  frame: any,
  now: number = Date.now(),
): SpeechLogView {
  if (frame?.type !== "speech_log") {
    return view;
  }
  const phase = frame.phase;
  if (phase !== "recording" && phase !== "final" && phase !== "idle") {
    return view;
  }
  if (phase === "recording") {
    return {
      recording: true,
      liveText: frame.text ?? "",
      entries: view.entries,
    };
  }
  if (phase === "final") {
    const trimmed = (frame.text ?? "").trim();
    if (trimmed.length === 0) {
      return {
        recording: false,
        liveText: "",
        entries: view.entries,
      };
    }
    const at = frame.server_ts ?? now;
    const entry: SpeechLogEntry = { text: trimmed, at };
    const nextEntries = [entry, ...view.entries].slice(0, SPEECH_LOG_LIMIT);
    return {
      recording: false,
      liveText: "",
      entries: nextEntries,
    };
  }
  if (phase === "idle") {
    return {
      recording: false,
      liveText: "",
      entries: view.entries,
    };
  }
  return view;
}

export function recordingFrameClass(recording: boolean): string | null {
  return recording ? "recording-frame" : null;
}

export function armManualRecording(view: SpeechLogView): SpeechLogView {
  if (view.recording) return view;
  return applySpeechLog(view, { type: "speech_log", phase: "recording", text: "" });
}

export function stopManualRecording(view: SpeechLogView): SpeechLogView {
  if (!view.recording) return view;
  return { ...view, recording: false, liveText: "" };
}
