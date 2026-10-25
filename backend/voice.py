"""Microphone wake-word listener that feeds the bridge audio buffer.

A daemon thread continuously reads the microphone with SpeechRecognition.
When a phrase contains the wake word "log", the captured utterance is
forwarded to the bridge as one binary WebSocket frame: byte 0 is ``0x02``
(audio) and the rest is 16 kHz mono PCM16. ``server.py`` appends that payload
to the audio ring buffer a Gemini Live media pump drains with
``POST /api/media/audio/drain``.

A phrase that is only the wake word arms a short window; the next phrase is
captured and forwarded as the command. The transcript is used only to spot
the wake word (Google Web Speech, via SpeechRecognition). The audio itself
is what Gemini hears.

Function calls for ``log_instrument`` and ``verify_tray_count`` that the
bridge relays back are executed with ``tools.dispatch`` and answered with a
``function_result``. Viewers already receive the ``function_call`` broadcast,
which is what the surgical log renders.

Run with ``python voice.py`` from this directory (the server defaults to
port 8000). Microphone capture uses sounddevice, which opens the default
input as 16 kHz mono PCM16 without PyAudio.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import threading
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


def _ensure_pyaudio() -> None:
    """Use PyAudio when installed; otherwise PyAudioWPatch (has a Python 3.14 Windows wheel)."""
    if "pyaudio" in sys.modules:
        return
    try:
        import pyaudio  # noqa: F401
    except ImportError:
        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            return
        sys.modules["pyaudio"] = pyaudio


_ensure_pyaudio()

import sounddevice as sd
import speech_recognition as sr
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

# Runnable as `python voice.py` from any working directory.
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from tools import TOOL_NAMES, ToolError, dispatch
from speech_log import SpeechLogSession

LOG = logging.getLogger("voice")

# Binary frame layout shared with server.py: frame[0] is the channel tag.
AUDIO_CHANNEL_TAG = 0x02
SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # PCM16, little-endian
COMMAND_WINDOW_S = 6.0
RECONNECT_DELAY_S = 2.0
LISTEN_TIMEOUT_S = 1.0
QUEUE_SIZE = 8
VOICE_STATUS_INTERVAL_S = 1.0

_TRAILING_PUNCTUATION = " \t,.!?;:-\"'"


@dataclass(frozen=True, slots=True)
class CapturedUtterance:
    pcm: bytes
    transcript: str


def wake_remainder(transcript: str, wake_word: str) -> str | None:
    """Text after the wake word.

    ``""`` means the phrase was only the wake word. ``None`` means the wake
    word was not spoken as its own word (so "catalog" does not match "log").
    """
    match = re.search(rf"\b{re.escape(wake_word.strip())}\b", transcript, flags=re.IGNORECASE)
    if match is None:
        return None
    return transcript[match.end() :].strip(_TRAILING_PUNCTUATION)


class VoiceSpeechLog:
    """Adapter that turns wake-listener transcripts/timeouts into SpeechLogSession events."""

    def __init__(self, wake_word: str = "log", silence_s: float = 1.5) -> None:
        self._session = SpeechLogSession(wake_word=wake_word, silence_s=silence_s)
        self._lock = threading.Lock()

    def on_transcript(self, transcript: str) -> list[dict[str, str]]:
        with self._lock:
            return self._session.hear(transcript)

    def on_listen_timeout(self, seconds: float = LISTEN_TIMEOUT_S) -> list[dict[str, str]]:
        with self._lock:
            return self._session.gap(seconds)

    def on_manual_start(self) -> list[dict[str, str]]:
        with self._lock:
            return self._session.arm()


def _client_id(server_url: str) -> str:
    query = urllib.parse.parse_qs(urllib.parse.urlparse(server_url).query)
    values = query.get("client_id")
    return values[0] if values else "voice"


def _is_disconnect(exc: BaseException) -> bool:
    """True when a session died because the socket closed, not because a task failed."""
    if isinstance(exc, (OSError, ConnectionClosed)):
        return True
    if isinstance(exc, BaseExceptionGroup):
        # TaskGroup cancels the sibling task when one side hits a dead socket.
        relevant = [inner for inner in exc.exceptions if not isinstance(inner, asyncio.CancelledError)]
        return bool(relevant) and all(_is_disconnect(inner) for inner in relevant)
    return False


class _SoundDeviceStream:
    """Bytes reader matching ``speech_recognition``'s ``stream.read(chunk)``."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream

    def read(self, size: int) -> bytes:
        data, _overflowed = self._stream.read(size)
        return bytes(data)

    def close(self) -> None:
        self._stream.stop()
        self._stream.close()


class SoundDeviceMicrophone(sr.AudioSource):
    """16 kHz mono PCM16 input through sounddevice.

    ``speech_recognition.Microphone`` needs PyAudio, which is not installed
    here. sounddevice already opens the MacBook microphone, and
    ``Recognizer.listen`` only needs ``stream.read``, a sample rate, and a
    16-bit sample width.
    """

    def __init__(
        self,
        device_index: int | None = None,
        sample_rate: int = SAMPLE_RATE,
        chunk_size: int = 1024,
    ) -> None:
        if device_index is not None and device_index < 0:
            raise ValueError("device index must be >= 0")
        self.device_index = device_index
        self.SAMPLE_RATE = sample_rate
        self.SAMPLE_WIDTH = SAMPLE_WIDTH
        self.CHUNK = chunk_size
        self.stream: _SoundDeviceStream | None = None

    def __enter__(self) -> SoundDeviceMicrophone:
        if self.stream is not None:
            raise RuntimeError("microphone is already open")
        raw = sd.RawInputStream(
            samplerate=self.SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=self.CHUNK,
            device=self.device_index,
        )
        raw.start()
        self.stream = _SoundDeviceStream(raw)
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        stream = self.stream
        self.stream = None
        if stream is not None:
            stream.close()


def list_input_devices() -> list[tuple[int, str]]:
    """Input devices as ``(index, name)``, using sounddevice's numbering."""
    devices = sd.query_devices()
    found: list[tuple[int, str]] = []
    for index, device in enumerate(devices):
        if int(device["max_input_channels"]) > 0:
            found.append((index, str(device["name"])))
    return found


def _prepare_microphone(recognizer: sr.Recognizer, microphone: sr.AudioSource) -> float:
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
    return recognizer.energy_threshold


class WakeListener:
    """Background thread: ``Recognizer.listen`` until the wake word is heard."""

    def __init__(
        self,
        recognizer: sr.Recognizer,
        microphone: sr.AudioSource,
        wake_word: str,
        language: str,
        phrase_time_limit: float,
        loop: asyncio.AbstractEventLoop,
        queue: asyncio.Queue[CapturedUtterance],
        log_queue: asyncio.Queue[dict[str, str]] | None = None,
    ) -> None:
        self._recognizer = recognizer
        self._microphone = microphone
        self._wake_word = wake_word.strip()
        self._language = language
        self._phrase_time_limit = phrase_time_limit
        self._loop = loop
        self._queue = queue
        self._log_queue: asyncio.Queue[dict[str, str]] = log_queue or asyncio.Queue()
        self._speech_log = VoiceSpeechLog(wake_word)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._command_deadline: float | None = None
        self.mic_open = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop_listen, name="voice-listener", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._phrase_time_limit + LISTEN_TIMEOUT_S + 1.0)

    def arm_from_button(self) -> None:
        """Same capture window as a bare "log": record the next phrase."""
        self._arm()
        self._enqueue_logs(self._speech_log.on_manual_start())
        LOG.info("record button pressed; capturing the next phrase")

    def _loop_listen(self) -> None:
        try:
            with self._microphone as source:
                self.mic_open = True
                while not self._stop.is_set():
                    try:
                        audio = self._recognizer.listen(
                            source,
                            timeout=LISTEN_TIMEOUT_S,
                            phrase_time_limit=self._phrase_time_limit,
                        )
                    except sr.WaitTimeoutError:
                        self._enqueue_logs(self._speech_log.on_listen_timeout())
                        continue
                    try:
                        self._handle(audio)
                    except Exception:
                        LOG.exception("wake listener failed on a phrase")
        except Exception:
            LOG.exception("microphone listener stopped")
        finally:
            self.mic_open = False

    def _handle(self, audio: sr.AudioData) -> None:
        now = time.monotonic()
        armed = self._command_deadline is not None and now <= self._command_deadline
        if self._command_deadline is not None and not armed:
            LOG.info("command window elapsed; waiting for the wake word again")
            self._command_deadline = None

        try:
            transcript = self._recognizer.recognize_google(audio, language=self._language)
        except sr.UnknownValueError:
            if armed:
                LOG.info("wake word heard, but the command was not understood; still listening")
            else:
                LOG.debug("unrecognized speech")
            return
        except sr.RequestError as exc:
            LOG.warning("wake-word recognizer unavailable: %s", exc)
            return

        transcript = transcript.strip()
        LOG.debug("heard %r", transcript)
        if not transcript:
            return

        self._enqueue_logs(self._speech_log.on_transcript(transcript))

        if armed:
            self._command_deadline = None
            followup = wake_remainder(transcript, self._wake_word)
            if followup == "":
                self._arm()
                LOG.info("wake word repeated; still waiting for the command")
                return
            self._forward(audio, transcript)
            return

        remainder = wake_remainder(transcript, self._wake_word)
        if remainder is None:
            return
        if remainder:
            self._forward(audio, transcript)
            return
        self._arm()
        LOG.info("wake word %r heard; capturing the next phrase", self._wake_word)

    def _arm(self) -> None:
        self._command_deadline = time.monotonic() + COMMAND_WINDOW_S

    def _forward(self, audio: sr.AudioData, transcript: str) -> None:
        pcm = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=SAMPLE_WIDTH)
        if not pcm:
            LOG.warning("captured audio was empty")
            return
        utterance = CapturedUtterance(pcm=pcm, transcript=transcript)
        LOG.info("wake word matched; forwarding %r (%d bytes pcm16)", transcript, len(pcm))
        try:
            self._loop.call_soon_threadsafe(self._enqueue, utterance)
        except RuntimeError:
            LOG.warning("dropped audio; the event loop is closed")

    def _enqueue(self, utterance: CapturedUtterance) -> None:
        try:
            self._queue.put_nowait(utterance)
        except asyncio.QueueFull:
            LOG.warning("dropped audio; the sender queue is full")

    def _enqueue_logs(self, events: list[dict[str, str]]) -> None:
        if not events:
            return
        try:
            self._loop.call_soon_threadsafe(self._put_logs, events)
        except RuntimeError:
            LOG.warning("dropped speech log events; the event loop is closed")

    def _put_logs(self, events: list[dict[str, str]]) -> None:
        for event in events:
            try:
                self._log_queue.put_nowait(event)
            except asyncio.QueueFull:
                LOG.warning("dropped speech log event; the sender queue is full")


def _result_frame(
    call_id: str,
    name: str,
    *,
    ok: bool,
    result: dict[str, Any] | None = None,
    error: dict[str, str] | None = None,
) -> dict[str, Any]:
    frame: dict[str, Any] = {
        "type": "function_result",
        "call_id": call_id,
        "name": name,
        "ok": ok,
    }
    if ok:
        frame["result"] = result if result is not None else {}
    else:
        frame["error"] = error
    return frame


async def _handle_function_call(websocket: Any, payload: dict[str, Any], client_id: str) -> None:
    name = payload.get("name")
    if name not in TOOL_NAMES:
        return
    target = payload.get("to")
    if target not in (None, client_id):
        return
    call_id = payload.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        LOG.warning("function_call %s is missing call_id", name)
        return

    args = payload.get("args")
    if args is None:
        args = {}
    try:
        result = dispatch(name, args)
    except ToolError as exc:
        LOG.info("tool %s rejected: %s", name, exc.message)
        frame = _result_frame(
            call_id,
            name,
            ok=False,
            error={"code": exc.code, "message": exc.message},
        )
    except Exception as exc:
        LOG.exception("tool %s failed", name)
        frame = _result_frame(
            call_id,
            name,
            ok=False,
            error={"code": "internal", "message": str(exc) or "tool failed"},
        )
    else:
        LOG.info("tool %s ok", name)
        frame = _result_frame(call_id, name, ok=True, result=result)
    await websocket.send(json.dumps(frame))


async def receive_frames(websocket: Any, client_id: str, on_start_recording=None) -> None:
    """Execute surgical tool calls the bridge relays from Gemini."""
    async for message in websocket:
        if not isinstance(message, str):
            continue
        try:
            payload = json.loads(message)
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        kind = payload.get("type")
        if kind == "start_recording":
            if on_start_recording is not None:
                on_start_recording()
        elif kind == "error":
            LOG.warning(
                "server rejected a frame: %s (%s)", payload.get("code"), payload.get("message")
            )
        elif kind == "function_call":
            await _handle_function_call(websocket, payload, client_id)
        else:
            LOG.debug("server -> %s", kind)


async def send_audio(websocket: Any, queue: asyncio.Queue[CapturedUtterance]) -> None:
    """Push each captured utterance into the bridge audio buffer."""
    while True:
        utterance = await queue.get()
        await websocket.send(bytes((AUDIO_CHANNEL_TAG,)) + utterance.pcm)
        LOG.info(
            "sent %d bytes of 16 kHz mono pcm16 for %r",
            len(utterance.pcm),
            utterance.transcript,
        )


async def send_voice_status(websocket: Any, listener: WakeListener) -> None:
    """Tell viewers the microphone is open. Silence means voice is off."""
    while True:
        if listener.mic_open:
            await websocket.send(json.dumps({"type": "voice_status", "active": True}))
        await asyncio.sleep(VOICE_STATUS_INTERVAL_S)


async def send_speech_logs(websocket: Any, queue: asyncio.Queue[dict[str, str]]) -> None:
    """Send each speech_log event as one JSON text frame."""
    while True:
        event = await queue.get()
        await websocket.send(json.dumps(event))


async def session(
    websocket: Any,
    queue: asyncio.Queue[CapturedUtterance],
    client_id: str,
    log_queue: asyncio.Queue[dict[str, str]],
    listener: WakeListener,
    on_start_recording=None,
) -> None:
    async with asyncio.TaskGroup() as tasks:
        tasks.create_task(receive_frames(websocket, client_id, on_start_recording), name="voice-receive")
        tasks.create_task(send_audio(websocket, queue), name="voice-send")
        tasks.create_task(send_speech_logs(websocket, log_queue), name="voice-speech-log")
        tasks.create_task(send_voice_status(websocket, listener), name="voice-status")


async def run(args: argparse.Namespace) -> None:
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    try:
        microphone = SoundDeviceMicrophone(device_index=args.device)
    except (OSError, ValueError, sd.PortAudioError) as exc:
        raise SystemExit(f"Could not open a microphone. {exc}") from exc

    LOG.info("calibrating microphone for ambient noise")
    try:
        threshold = await asyncio.to_thread(_prepare_microphone, recognizer, microphone)
    except (OSError, ValueError, sd.PortAudioError) as exc:
        raise SystemExit(f"Could not open the microphone: {exc}") from exc
    LOG.info(
        "listening for wake word %r (energy threshold %.0f)",
        args.wake_word,
        threshold,
    )

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[CapturedUtterance] = asyncio.Queue(maxsize=QUEUE_SIZE)
    log_queue: asyncio.Queue[dict[str, str]] = asyncio.Queue(maxsize=QUEUE_SIZE)
    listener = WakeListener(
        recognizer,
        microphone,
        args.wake_word,
        args.language,
        args.phrase_time_limit,
        loop,
        queue,
        log_queue=log_queue,
    )
    listener.start()
    client_id = _client_id(args.server)
    try:
        while True:
            try:
                async with connect(args.server, max_size=None, ping_interval=20) as websocket:
                    LOG.info("connected to %s", args.server)
                    try:
                        await session(
                            websocket,
                            queue,
                            client_id,
                            log_queue,
                            listener,
                            listener.arm_from_button,
                        )
                    except BaseExceptionGroup as exc:
                        if not _is_disconnect(exc):
                            raise
                        LOG.warning(
                            "connection lost (%s); retrying in %.0fs",
                            exc.exceptions[0],
                            RECONNECT_DELAY_S,
                        )
            except (OSError, ConnectionClosed) as exc:
                LOG.warning("connection lost (%s); retrying in %.0fs", exc, RECONNECT_DELAY_S)
            await asyncio.sleep(RECONNECT_DELAY_S)
    finally:
        listener.stop()


def _list_devices() -> None:
    try:
        devices = list_input_devices()
    except (OSError, sd.PortAudioError) as exc:
        raise SystemExit(f"Could not list microphones. {exc}") from exc
    if not devices:
        print("no microphones found")
        return
    for index, name in devices:
        print(f"{index}\t{name}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Listen for the wake word 'log' and forward the utterance to the bridge."
    )
    parser.add_argument(
        "--server",
        default="ws://127.0.0.1:8000/ws?role=input&client_id=voice",
        help="WebSocket URL of the bridge, including the role and client_id query.",
    )
    parser.add_argument(
        "--wake-word",
        default="log",
        help="Whole word that arms audio capture. Default: log.",
    )
    parser.add_argument(
        "--language",
        default="en-US",
        help="SpeechRecognition language used only to spot the wake word.",
    )
    parser.add_argument(
        "--phrase-time-limit",
        type=float,
        default=8.0,
        help="Maximum seconds of speech captured in one phrase.",
    )
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Microphone index from --list-devices. Defaults to the system microphone.",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="Print microphone indexes and exit.",
    )
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    args.wake_word = args.wake_word.strip()
    if not args.wake_word:
        parser.error("--wake-word must not be empty")
    if args.phrase_time_limit <= 0:
        parser.error("--phrase-time-limit must be positive")
    if args.device is not None and args.device < 0:
        parser.error("--device must be >= 0")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.list_devices:
        _list_devices()
        return
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        LOG.info("stopped")


if __name__ == "__main__":
    main()
