import test from "node:test";
import assert from "node:assert/strict";
import {
  emptySpeechLog,
  applySpeechLog,
  speechLogStatusLabel,
  SPEECH_LOG_LIMIT,
} from "../app/speech-log-state.ts";

test("emptySpeechLog returns initial view", () => {
  const view = emptySpeechLog();
  assert.deepStrictEqual(view, { recording: false, liveText: "", entries: [] });
  assert.strictEqual(speechLogStatusLabel(view), "Listening for log");
});

test("apply recording empty text", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "recording", text: "" });
  assert.strictEqual(view.recording, true);
  assert.strictEqual(view.liveText, "");
  assert.strictEqual(view.entries.length, 0);
  assert.strictEqual(speechLogStatusLabel(view), "Recording");
});

test("apply recording with text", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "recording", text: "" });
  view = applySpeechLog(view, { type: "speech_log", phase: "recording", text: "add sponge" });
  assert.strictEqual(view.recording, true);
  assert.strictEqual(view.liveText, "add sponge");
  assert.strictEqual(view.entries.length, 0);
});

test("apply final with server_ts", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "recording", text: "" });
  view = applySpeechLog(view, { type: "speech_log", phase: "recording", text: "add sponge" });
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "add sponge", server_ts: 1700000000000 });
  assert.strictEqual(view.recording, false);
  assert.strictEqual(view.liveText, "");
  assert.strictEqual(view.entries.length, 1);
  assert.deepStrictEqual(view.entries[0], { text: "add sponge", at: 1700000000000 });
  assert.strictEqual(speechLogStatusLabel(view), "Listening for log");
});

test("idle keeps entry", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "add sponge", server_ts: 1700000000000 });
  view = applySpeechLog(view, { type: "speech_log", phase: "idle", text: "" });
  assert.strictEqual(view.recording, false);
  assert.strictEqual(view.liveText, "");
  assert.strictEqual(view.entries.length, 1);
});

test("final empty or whitespace does not add entry", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "   " });
  assert.strictEqual(view.entries.length, 0);
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "" });
  assert.strictEqual(view.entries.length, 0);
  assert.strictEqual(view.recording, false);
  assert.strictEqual(view.liveText, "");
});

test("newest-first order", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "add sponge", server_ts: 1 });
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "two clamps", server_ts: 2 });
  assert.strictEqual(view.entries.length, 2);
  assert.strictEqual(view.entries[0].text, "two clamps");
  assert.strictEqual(view.entries[1].text, "add sponge");
});

test("SPEECH_LOG_LIMIT caps at 50, newest first", () => {
  let view = emptySpeechLog();
  for (let i = 0; i < 51; i++) {
    view = applySpeechLog(view, { type: "speech_log", phase: "final", text: `item${i}`, server_ts: i });
  }
  assert.strictEqual(view.entries.length, 50);
  assert.strictEqual(SPEECH_LOG_LIMIT, 50);
  assert.strictEqual(view.entries[0].text, "item50");
  assert.strictEqual(view.entries[49].text, "item1");
});

test("unknown frame types/phases return same view", () => {
  const initial = emptySpeechLog();
  let view = applySpeechLog(initial, { type: "telemetry" } as any);
  assert.strictEqual(view.recording, initial.recording);
  assert.strictEqual(view.liveText, initial.liveText);
  assert.deepStrictEqual(view.entries, initial.entries);
  view = applySpeechLog(initial, { type: "speech_log", phase: "nope" } as any);
  assert.strictEqual(view.recording, initial.recording);
  assert.strictEqual(view.liveText, initial.liveText);
  assert.deepStrictEqual(view.entries, initial.entries);
});

test("final without server_ts uses now arg", () => {
  let view = emptySpeechLog();
  view = applySpeechLog(view, { type: "speech_log", phase: "final", text: "foo" }, 42);
  assert.strictEqual(view.entries.length, 1);
  assert.strictEqual(view.entries[0].at, 42);
});
