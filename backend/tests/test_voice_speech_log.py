"""Microphone transcript, end-word buffering, and speech-log frame tests."""

import asyncio
import json
import sys
import types
import unittest


def _install_stubs():
    sounddevice = types.ModuleType("sounddevice")
    sounddevice.PortAudioError = type("PortAudioError", (Exception,), {})
    sys.modules["sounddevice"] = sounddevice

    # speech_recognition stub
    sr = types.ModuleType("speech_recognition")
    sr.WaitTimeoutError = type("WaitTimeoutError", (Exception,), {})
    sr.UnknownValueError = type("UnknownValueError", (Exception,), {})
    sr.RequestError = type("RequestError", (Exception,), {})
    sr.Recognizer = type("Recognizer", (), {})
    sr.AudioSource = type("AudioSource", (), {})
    sr.Microphone = type("Microphone", (), {})
    sr.AudioData = type("AudioData", (), {})
    sys.modules["speech_recognition"] = sr

    # websockets stub
    ws = types.ModuleType("websockets")
    ws_exc = types.ModuleType("websockets.exceptions")
    ws_exc.ConnectionClosed = type("ConnectionClosed", (Exception,), {})
    ws.exceptions = ws_exc
    ws_asyncio = types.ModuleType("websockets.asyncio")
    ws_client = types.ModuleType("websockets.asyncio.client")
    ws_client.connect = lambda *a, **k: None
    ws_asyncio.client = ws_client
    ws.asyncio = ws_asyncio
    sys.modules["websockets"] = ws
    sys.modules["websockets.exceptions"] = ws_exc
    sys.modules["websockets.asyncio"] = ws_asyncio
    sys.modules["websockets.asyncio.client"] = ws_client


_install_stubs()

# Now safe to import from backend
import voice  # noqa: E402


class FakeMicrophone:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass


class FakeAudioData:
    def __init__(self, transcript):
        self._transcript = transcript

    def get_raw_data(self, convert_rate=None, convert_width=None):
        return b"\x00\x01"


class TestVoiceSpeechLog(unittest.TestCase):
    def setUp(self):
        self.sr = sys.modules["speech_recognition"]
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    def _make_listen(self, script):
        """listen pops from script; returns FakeAudioData or raises."""
        calls = {"i": 0}

        def listen(source, timeout=None, phrase_time_limit=None):
            i = calls["i"]
            calls["i"] += 1
            if i >= len(script):
                raise RuntimeError("done")
            item = script[i]
            if item == "timeout":
                raise self.sr.WaitTimeoutError()
            # item is the transcript string
            return FakeAudioData(item)

        return listen

    def test_voice_speech_log_interface_exists(self):
        self.assertTrue(hasattr(voice, "VoiceSpeechLog"))
        self.assertTrue(hasattr(voice, "LISTEN_TIMEOUT_S"))
        self.assertTrue(hasattr(voice, "WakeListener"))
        self.assertTrue(hasattr(voice, "send_speech_logs"))

    def test_scripted_listener_emits_speech_log_frames(self):
        recognizer = self.sr.Recognizer()
        def recognize_google(audio, language=None):
            return audio._transcript
        recognizer.recognize_google = recognize_google
        microphone = FakeMicrophone()
        log_queue: asyncio.Queue = asyncio.Queue()
        audio_queue: asyncio.Queue = asyncio.Queue()

        wl = voice.WakeListener(
            recognizer,
            microphone,
            wake_word="log",
            language="en-US",
            phrase_time_limit=5,
            loop=self.loop,
            queue=audio_queue,
            log_queue=log_queue,
        )
        wl._command_buffer.end_word_pause_ms = 0

        # Patch the internal loop object (not wl.loop)
        class FakeLoop:
            def call_soon_threadsafe(self, fn, arg):
                fn(arg)

        wl._loop = FakeLoop()

        # script 1: timeout, catalog, log
        recognizer.listen = self._make_listen(["timeout", "catalog", "log"])
        try:
            wl._loop_listen()
        except RuntimeError:
            pass
        self.assertEqual(log_queue.qsize(), 1)
        self.assertEqual(log_queue.get_nowait(), {"type": "speech_log", "phase": "recording", "text": ""})
        self.assertTrue(audio_queue.empty())

        # script 2: timeout after bare log leaves queue empty (no idle emitted)
        recognizer.listen = self._make_listen(["timeout"])
        try:
            wl._loop_listen()
        except RuntimeError:
            pass
        self.assertEqual(log_queue.qsize(), 0)
        self.assertTrue(audio_queue.empty())

        # The transcript and audio stay buffered until the end word arrives.
        recognizer.listen = self._make_listen(["Log, add sponge."])
        try:
            wl._loop_listen()
        except RuntimeError:
            pass
        self.assertEqual(log_queue.qsize(), 1)
        self.assertEqual(log_queue.get_nowait(), {"type": "speech_log", "phase": "recording", "text": "add sponge"})
        self.assertTrue(audio_queue.empty())

        # Additional stable phrases accumulate in the same open command.
        recognizer.listen = self._make_listen(["and two clamps"])
        try:
            wl._loop_listen()
        except RuntimeError:
            pass
        self.assertEqual(log_queue.qsize(), 1)
        self.assertEqual(log_queue.get_nowait(), {"type": "speech_log", "phase": "recording", "text": "add sponge and two clamps"})
        self.assertTrue(audio_queue.empty())

        # The terminal word is removed from the visible transcript, and the
        # combined audio is submitted only after the short stability pause.
        recognizer.listen = self._make_listen(["over"])
        try:
            wl._loop_listen()
        except RuntimeError:
            pass
        self.assertEqual(log_queue.qsize(), 1)
        self.assertEqual(
            log_queue.get_nowait(),
            {"type": "speech_log", "phase": "recording", "text": "add sponge and two clamps"},
        )
        self.assertTrue(audio_queue.empty())

        wl._poll_command()
        self.assertEqual(log_queue.qsize(), 2)
        self.assertEqual(log_queue.get_nowait(), {"type": "speech_log", "phase": "final", "text": "add sponge and two clamps"})
        self.assertEqual(log_queue.get_nowait(), {"type": "speech_log", "phase": "idle", "text": ""})
        self.assertEqual(audio_queue.qsize(), 1)
        utterance = audio_queue.get_nowait()
        self.assertIsInstance(utterance, voice.CapturedUtterance)
        self.assertEqual(utterance.transcript, "add sponge and two clamps")
        self.assertEqual(utterance.pcm, b"\x00\x01" * 3)

    def test_record_button_arms_the_next_phrase(self):
        recognizer = self.sr.Recognizer()

        def recognize_google(audio, language=None):
            return audio._transcript

        recognizer.recognize_google = recognize_google
        log_queue: asyncio.Queue = asyncio.Queue()
        audio_queue: asyncio.Queue = asyncio.Queue()
        wl = voice.WakeListener(
            recognizer,
            FakeMicrophone(),
            wake_word="log",
            language="en-US",
            phrase_time_limit=5,
            loop=self.loop,
            queue=audio_queue,
            log_queue=log_queue,
        )
        wl._command_buffer.end_word_pause_ms = 0

        class FakeLoop:
            def call_soon_threadsafe(self, fn, arg):
                fn(arg)

        wl._loop = FakeLoop()
        wl.arm_from_button()
        self.assertEqual(
            log_queue.get_nowait(),
            {"type": "speech_log", "phase": "recording", "text": ""},
        )
        self.assertTrue(wl._command_buffer.active)

        recognizer.listen = self._make_listen(["add sponge over"])
        try:
            wl._loop_listen()
        except RuntimeError:
            pass
        self.assertEqual(
            log_queue.get_nowait(),
            {"type": "speech_log", "phase": "recording", "text": "add sponge"},
        )
        self.assertTrue(audio_queue.empty())
        wl._poll_command()
        self.assertEqual(
            log_queue.get_nowait(),
            {"type": "speech_log", "phase": "final", "text": "add sponge"},
        )
        self.assertEqual(log_queue.get_nowait(), {"type": "speech_log", "phase": "idle", "text": ""})
        self.assertEqual(audio_queue.qsize(), 1)
        self.assertEqual(audio_queue.get_nowait().transcript, "add sponge")

    def test_send_speech_logs(self):
        q: asyncio.Queue = asyncio.Queue()
        event = {"type": "speech_log", "phase": "recording", "text": "foo"}
        q.put_nowait(event)

        sent = []
        send_event = asyncio.Event()

        class FakeWS:
            async def send(self, data):
                sent.append(data)
                send_event.set()

        async def run():
            task = asyncio.create_task(voice.send_speech_logs(FakeWS(), q))
            await asyncio.wait_for(send_event.wait(), timeout=1)
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            return sent

        sent_payloads = self.loop.run_until_complete(run())
        self.assertEqual(len(sent_payloads), 1)
        payload = json.loads(sent_payloads[0])
        self.assertEqual(payload, event)


if __name__ == "__main__":
    unittest.main()
