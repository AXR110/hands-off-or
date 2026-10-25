"""Drain the bridge audio buffer into a Gemini Live session.

Voice captures 16 kHz mono PCM16 and the bridge stores it. This task forwards
each utterance, then ``audioStreamEnd``, so the model commits the turn and can
request ``log_instrument`` or ``verify_tray_count``. The server executes those
calls and sends the tool response back on this socket.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
from collections.abc import Awaitable, Callable
from typing import Any

from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

from tools import live_function_declarations

LOG = logging.getLogger("gemini-live")

LIVE_URL = (
    "wss://generativelanguage.googleapis.com/ws/"
    "google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent"
)
DEFAULT_MODEL = "gemini-3.8-live"
PCM_MIME = "audio/pcm;rate=16000"
# ~100 ms at 16 kHz mono PCM16. Live input is streamed, then closed per utterance.
CHUNK_BYTES = 3200
RECONNECT_DELAY_S = 2.0
IDLE_POLL_S = 0.05

SYSTEM_INSTRUCTION = (
    "You are the instrument logger for a sterile operating room. "
    "The surgeon speaks a command after the wake word. "
    "When they add, remove, or use an instrument, call log_instrument. "
    "added and removed change the running tray count; used records use and does not. "
    "Quantity defaults to 1. "
    "When they ask to verify, confirm, or reconcile a count, call verify_tray_count. "
    "Do not call a tool for unrelated speech."
)

ToolHandler = Callable[[list[dict[str, Any]]], Awaitable[list[dict[str, Any]]]]
AudioDrain = Callable[[], Awaitable[list[bytes]]]

_active = False


def is_active() -> bool:
    """True while a Live session has finished setup and is accepting audio."""
    return _active


def api_key() -> str | None:
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
        value = os.getenv(name, "").strip()
        if value:
            return value
    return None


def _model_name() -> str:
    model = os.getenv("GEMINI_LIVE_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    if model.startswith("models/"):
        return model
    return f"models/{model}"


def _setup_message(system_instruction: str | None = None) -> dict[str, Any]:
    provided = system_instruction.strip() if system_instruction else ""
    instruction = provided or SYSTEM_INSTRUCTION
    return {
        "setup": {
            "model": _model_name(),
            "generationConfig": {"responseModalities": ["AUDIO"]},
            "systemInstruction": {"parts": [{"text": instruction}]},
            "tools": [{"functionDeclarations": live_function_declarations()}],
        }
    }


def _audio_message(pcm: bytes) -> str:
    return json.dumps(
        {
            "realtimeInput": {
                "audio": {
                    "data": base64.b64encode(pcm).decode("ascii"),
                    "mimeType": PCM_MIME,
                }
            }
        }
    )


def _redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


async def _send_utterance(websocket: Any, pcm: bytes) -> None:
    for start in range(0, len(pcm), CHUNK_BYTES):
        piece = pcm[start : start + CHUNK_BYTES]
        if piece:
            await websocket.send(_audio_message(piece))
    await websocket.send(json.dumps({"realtimeInput": {"audioStreamEnd": True}}))


async def _pump_audio(websocket: Any, drain: AudioDrain, ready: asyncio.Event) -> None:
    await ready.wait()
    while True:
        chunks = await drain()
        if not chunks:
            await asyncio.sleep(IDLE_POLL_S)
            continue
        for pcm in chunks:
            if pcm:
                LOG.info("forwarding %d bytes of pcm16 to Gemini Live", len(pcm))
                await _send_utterance(websocket, pcm)


async def _receive(websocket: Any, on_tool_call: ToolHandler, ready: asyncio.Event) -> None:
    global _active
    async for raw in websocket:
        if not isinstance(raw, str):
            continue
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            LOG.warning("gemini live sent a non-JSON frame")
            continue
        if not isinstance(message, dict):
            continue
        if message.get("error"):
            LOG.warning("gemini live error: %s", message["error"])
            continue
        if "setupComplete" in message:
            _active = True
            ready.set()
            LOG.info("gemini live session ready")
            continue
        tool_call = message.get("toolCall")
        if not isinstance(tool_call, dict):
            continue
        calls = tool_call.get("functionCalls")
        if not isinstance(calls, list) or not calls:
            continue
        responses = await on_tool_call([call for call in calls if isinstance(call, dict)])
        if responses:
            await websocket.send(json.dumps({"toolResponse": {"functionResponses": responses}}))


async def session(
    drain: AudioDrain,
    on_tool_call: ToolHandler,
    key: str,
    system_instruction: str | None = None,
) -> None:
    """One Live socket: setup, then audio in and tool calls out."""
    global _active
    ready = asyncio.Event()
    async with connect(f"{LIVE_URL}?key={key}", max_size=None, ping_interval=20) as websocket:
        await websocket.send(json.dumps(_setup_message(system_instruction)))
        try:
            async with asyncio.TaskGroup() as tasks:
                tasks.create_task(_pump_audio(websocket, drain, ready), name="gemini-audio")
                tasks.create_task(_receive(websocket, on_tool_call, ready), name="gemini-receive")
        finally:
            _active = False


async def supervise(
    drain: AudioDrain,
    on_tool_call: ToolHandler,
    system_instruction: str | None = None,
) -> None:
    """Reconnect the Live session until cancelled. Idle when no API key is set."""
    key = api_key()
    if not key:
        LOG.warning(
            "GEMINI_API_KEY is not set; spoken instrument commands are logged from the "
            "transcript instead of Gemini Live"
        )
        return
    while True:
        try:
            await session(drain, on_tool_call, key, system_instruction)
            LOG.warning("gemini live session closed; reconnecting in %.0fs", RECONNECT_DELAY_S)
        except asyncio.CancelledError:
            raise
        except ConnectionClosed as exc:
            LOG.warning("gemini live connection closed (%s); reconnecting", _redact(str(exc), key))
        except Exception as exc:
            LOG.warning("gemini live session failed (%s); reconnecting", _redact(str(exc), key))
        await asyncio.sleep(RECONNECT_DELAY_S)
