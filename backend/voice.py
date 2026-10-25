"""Microphone wake-word listener that feeds the bridge audio buffer.

A daemon thread continuously reads the microphone with SpeechRecognition.
After the wake word "log" or the Record button opens a command buffer,
transcribed phrases and their audio accumulate until the command ends with
"over". The stable transcript is shown in the viewer, and the captured audio
is forwarded as one binary WebSocket frame only after the delimiter survives a
short pause. A silence timeout discards stale commands.

The audio frame uses byte 0 as the ``0x02`` channel tag and the rest as 16 kHz
mono PCM16. ``server.py`` appends it to the audio ring buffer that the Gemini
Live media pump drains with ``POST /api/media/audio/drain``.

Function calls for ``log_instrument`` and ``verify_tray_count`` that the
bridge relays back are executed with ``tools.dispatch`` and answered with a
``function_result``. Viewers already receive the ``function_call`` broadcast,
which is what the surgical log renders.

Run with ``python voice.py`` from this directory (the server defaults to
port 8000). Capture opens the built-in MacBook microphone through sounddevice
at that device's native 48000 Hz or 44100 Hz, falling back to 16 kHz. The
socket payload stays 16 kHz mono PCM16, which is what Gemini Live consumes.
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
from end_word import END_WORD, END_WORD_PAUSE_MS, EndWordBuffer
from speech_log import SpeechLogSession

LOG = logging.getLogger("voice")

# Binary frame layout shared with server.py: frame[0] is the channel tag.
AUDIO_CHANNEL_TAG = 0x02
# Wire rate for the bridge and Gemini Live. The capture stream uses the
# Mac's native rate instead; opening PyAudio at 16 kHz crashes the built-in mic.
SAMPLE_RATE = 16_000
MAC_NATIVE_SAMPLE_RATES = (48_000, 44_100)
SAMPLE_WIDTH = 2  # PCM16, little-endian
# Fixed gate. A loud floor plus dynamic adjustment climbs above a speaking voice.
ENERGY_THRESHOLD = 300
AMBIENT_CALIBRATION_S = 1.5
RECONNECT_DELAY_S = 2.0
LISTEN_TIMEOUT_S = max(0.05, min(1.0, END_WORD_PAUSE_MS / 1000))
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


def _builtin_microphone_rank(name: str) -> int | None:
    """Lower rank is a better match. ``None`` means this is not the built-in array."""
    lowered = name.casefold()
    if any(skip in lowered for skip in ("speaker", "iphone", "airpod", "bluetooth")):
        return None
    if "macbook" in lowered and "microphone" in lowered:
        return 0
    if lowered in {"built-in microphone", "built in microphone", "builtin microphone"} or (
        "built-in" in lowered and "microphone" in lowered
    ):
        return 1
    return None


def _device_names() -> list[str]:
    """sounddevice names. The index matches ``SoundDeviceMicrophone(device_index=...)``."""
    return [str(device["name"]) for device in sd.query_devices()]


def select_builtin_microphone_index(names: Sequence[str | None] | None = None) -> int:
    """Index of the MacBook built-in microphone.

    The default input on a laptop with Continuity can be a phone mic, which
    drops the room audio. Names come from sounddevice so the index can be
    opened without PyAudio.
    """
    if names is None:
        names = _device_names()
    ranked: list[tuple[int, int]] = []
    for index, name in enumerate(names):
        if not name:
            continue
        rank = _builtin_microphone_rank(name)
        if rank is not None:
            ranked.append((rank, index))
    if not ranked:
        listed = ", ".join(f"{index}: {name}" for index, name in enumerate(names)) or "none"
        raise ValueError(f"built-in MacBook microphone not found among: {listed}")
    ranked.sort()
    return ranked[0][1]


def match_native_sample_rate(reported_rate: float) -> int:
    """Snap a device rate to 48000 or 44100.

    Those are the rates the Mac built-in array actually runs. Asking PyAudio
    for 16000 raises an invalid-sample-rate error and the stream never opens.
    """
    rate = int(round(float(reported_rate)))
    if rate in MAC_NATIVE_SAMPLE_RATES:
        return rate
    return MAC_NATIVE_SAMPLE_RATES[0]


def _candidate_sample_rates(reported_rate: float) -> tuple[int, int]:
    primary = match_native_sample_rate(reported_rate)
    secondary = 44_100 if primary == 48_000 else 48_000
    return primary, secondary


def _capture_rates(reported_rate: float) -> tuple[int, ...]:
    """Native Mac rate first, then 16 kHz, which sounddevice can resample."""
    rates: list[int] = []
    for rate in (*_candidate_sample_rates(reported_rate), SAMPLE_RATE):
        if rate not in rates:
            rates.append(rate)
    return tuple(rates)


def open_microphone(device_index: int | None = None) -> SoundDeviceMicrophone:
    """Open the built-in MacBook mic, or ``device_index`` when one was passed.

    The app environment does not install PyAudio. sounddevice opens the same
    Core Audio input and was the path that actually delivered speech.
    """
    devices = list(sd.query_devices())
    index = device_index if device_index is not None else select_builtin_microphone_index()
    if index < 0 or index >= len(devices):
        raise ValueError(f"device index {index} is out of range ({len(devices)} devices)")
    info = devices[index]
    name = str(info["name"])
    if int(info["max_input_channels"]) < 1:
        raise ValueError(f"device {index} ({name}) is not an input")
    reported = float(info["default_samplerate"])
    last_error: Exception | None = None
    for rate in _capture_rates(reported):
        microphone = SoundDeviceMicrophone(device_index=index, sample_rate=rate)
        try:
            with microphone as source:
                if source.stream is None:
                    raise OSError(f"microphone stream did not open at {rate} Hz")
        except (OSError, ValueError, sd.PortAudioError) as exc:
            last_error = exc
            LOG.warning("microphone %d %r rejected %d Hz: %s", index, name, rate, exc)
            continue
        LOG.info("selected microphone %d %r at %d Hz", index, name, rate)
        return SoundDeviceMicrophone(device_index=index, sample_rate=rate)
    raise OSError(f"could not open microphone {index} ({name})") from last_error


def _prepare_microphone(recognizer: sr.Recognizer, microphone: sr.AudioSource) -> float:
    recognizer.dynamic_energy_threshold = False
    with microphone as source:
        if isinstance(source, (sr.Microphone, SoundDeviceMicrophone)) and source.stream is None:
            raise OSError(
                f"microphone stream did not open at {microphone.SAMPLE_RATE} Hz"
            )
        recognizer.adjust_for_ambient_noise(source, duration=AMBIENT_CALIBRATION_S)
    # A loud floor pushes this above a speaking voice. Cap it, and never raise
    # a quiet-room calibration up to the cap or soft speech is ignored.
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = min(recognizer.energy_threshold, ENERGY_THRESHOLD)
    return recognizer.energy_threshold


class WakeListener:
    """Listen continuously; buffer command text and audio until its end word."""

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
        self._command_buffer = EndWordBuffer()
        self._command_audio: list[bytes] = []
        self._command_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
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
        """Open a command buffer when the Record button is pressed."""
        with self._command_lock:
            self._command_audio.clear()
            events = self._command_buffer.start(time.monotonic() * 1000)
        self._enqueue_logs(events)
        LOG.info("record button pressed; listening for a command ending in %r", END_WORD)

    def _loop_listen(self) -> None:
        try:
            with self._microphone as source:
                if isinstance(source, (sr.Microphone, SoundDeviceMicrophone)) and source.stream is None:
                    raise OSError("microphone stream did not open")
                self.mic_open = True
                while not self._stop.is_set():
                    try:
                        audio = self._recognizer.listen(
                            source,
                            timeout=LISTEN_TIMEOUT_S,
                            phrase_time_limit=self._phrase_time_limit,
                        )
                    except sr.WaitTimeoutError:
                        self._poll_command()
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
        try:
            transcript = self._recognizer.recognize_google(audio, language=self._language)
        except sr.UnknownValueError:
            LOG.debug("unrecognized speech")
            return
        except sr.RequestError as exc:
            LOG.warning("wake-word recognizer unavailable: %s", exc)
            return

        transcript = transcript.strip()
        LOG.debug("heard %r", transcript)
        if not transcript:
            return

        with self._command_lock:
            armed = self._command_buffer.active
        remainder = wake_remainder(transcript, self._wake_word)
        if not armed:
            if remainder is None:
                return
            with self._command_lock:
                self._command_audio.clear()
                events = self._command_buffer.start(time.monotonic() * 1000)
            self._enqueue_logs(events)
            LOG.info("wake word %r heard; listening for a command ending in %r", self._wake_word, END_WORD)
        phrase = remainder if remainder else transcript
        if remainder == "":
            return
        if remainder is not None:
            phrase = remainder

        pcm = audio.get_raw_data(convert_rate=SAMPLE_RATE, convert_width=SAMPLE_WIDTH)
        with self._command_lock:
            if not self._command_buffer.active:
                return
            self._command_audio.append(pcm)
            events = self._command_buffer.append(phrase, time.monotonic() * 1000)
        self._enqueue_logs(events)
        self._send_completed_if_ready(events)

    def _poll_command(self) -> None:
        with self._command_lock:
            events = self._command_buffer.poll(time.monotonic() * 1000)
        if events and events[-1]["phase"] == "idle":
            self._enqueue_logs(events)
            self._send_completed_if_ready(events)

    def _send_completed_if_ready(self, events: list[dict[str, str]]) -> None:
        final = next((event for event in events if event["phase"] == "final"), None)
        if final is None:
            if events and events[-1]["phase"] == "idle":
                with self._command_lock:
                    self._command_audio.clear()
            return
        with self._command_lock:
            pcm = b"".join(self._command_audio)
            self._command_audio.clear()
        if not pcm:
            LOG.warning("completed voice command had no audio")
            return
        utterance = CapturedUtterance(pcm=pcm, transcript=final["text"])
        LOG.info("end word matched; forwarding %r (%d bytes pcm16)", final["text"], len(pcm))
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
    recognizer.dynamic_energy_threshold = False
    recognizer.energy_threshold = ENERGY_THRESHOLD
    try:
        microphone = open_microphone(args.device)
    except (OSError, ValueError, AssertionError, sd.PortAudioError) as exc:
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
        devices = sd.query_devices()
    except (OSError, sd.PortAudioError) as exc:
        raise SystemExit(f"Could not list microphones. {exc}") from exc
    inputs = [
        (index, str(device["name"]), int(device["default_samplerate"]))
        for index, device in enumerate(devices)
        if int(device["max_input_channels"]) > 0
    ]
    if not inputs:
        print("no microphones found")
        return
    for index, name, rate in inputs:
        print(f"{index}\t{name}\t{rate} Hz")


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
        help="Input device index from --list-devices. Defaults to the built-in MacBook microphone.",
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
