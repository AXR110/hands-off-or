"""Async WebSocket bridge for the pointer-telemetry protocol.

Responsibilities
----------------
* Accept WebSocket clients in one of three roles: ``input`` (emits pointer
  telemetry and media), ``viewer`` (renders the shared pointer) and ``gemini``
  (relays Gemini Live function calls).
* Fan out ``telemetry`` samples from the input device to every other connected
  client, stamping them with a server sequence and timestamp, and keep the most
  recent sample to hand to late joiners as a ``state_snapshot``.
* Relay ``function_call`` / ``function_result`` frames between the Gemini
  bridge and the clients that act on them.
* Append video (binary tag-1 frames or base64 ``video_frame`` JSON) and binary
  audio frames to bounded in-memory ring buffers that a Gemini Live media pump
  can drain over HTTP.

The wire contract lives in ``shared_schema.json`` next to this file; the Gemini
tool declarations are served straight out of it, so there is one source of
truth for both ends.

Design notes
------------
* Telemetry is fire-and-forget: a slow or dead client is dropped from the
  broadcast rather than blocking the input stream.
* Media buffers are bounded in both bytes and chunk count and evict oldest
  first, so an unread buffer cannot grow without limit.
* Protocol errors are reported with an ``error`` frame and keep the socket
  open, so a live input stream can recover without reconnecting.

Run with ``python server.py`` or ``uvicorn server:app --reload``.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import contextlib
import itertools
import json
import logging
import math
import os
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import gemini_live
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from image_picker import encode_images, pick_image_paths
from tools import ToolError, dispatch, interpret_log_command
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    field_validator,
    model_validator,
)

LOG = logging.getLogger("bridge")


def _now_ms() -> int:
    return int(time.time() * 1000)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        LOG.warning("ignoring non-numeric %s=%r", name, raw)
        return default


# --- shared contract -------------------------------------------------------

SCHEMA_PATH = Path(__file__).with_name("shared_schema.json")
SCHEMA: dict[str, Any] = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
GEMINI_TOOLS: list[dict[str, Any]] = SCHEMA["x-gemini"]["tools"]
TOOL_NAMES = frozenset(tool["name"] for tool in GEMINI_TOOLS)
TRAY_TOOLS = frozenset({"log_instrument", "verify_tray_count"})

# Injected into the Gemini Live setup as systemInstruction. The session hears
# raw audio from a loud room, so ambiguous words should resolve to instruments,
# sutures, procedures, and spoken counts.
GEMINI_LIVE_SYSTEM_PROMPT = (
    "You are the instrument logger for a sterile operating room. "
    "The surgeon speaks a command after the wake word, often on a loud floor where words are clipped. "
    "Heavily bias your hearing and parsing toward surgical terminology and numerical quantities. "
    "When audio is ambiguous, choose the surgical instrument, suture, procedure, or spoken count "
    "instead of an everyday look-alike. Keep terms such as scalpel, Vicryl, laparotomy, forceps, "
    "hemostat, retractor, needle driver, sponge, suture, trocar, and clamp, and keep quantities "
    "such as one, two, three, four, five, and ten. "
    "When they add, remove, or use an instrument, call log_instrument. "
    "added and removed change the running tray count; used records use and does not. "
    "Quantity defaults to 1. Copy any spoken quantity exactly and do not drop it. "
    "When they ask to verify, confirm, or reconcile a count, call verify_tray_count. "
    "Do not call a tool for unrelated speech."
)

ROLES = ("input", "viewer", "gemini")
CLIENT_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,64}$")
ClientId = Annotated[str, StringConstraints(pattern=CLIENT_ID_RE.pattern, min_length=1, max_length=64)]

# Binary frame layout: frame[0] is the channel tag, frame[1:] is the payload.
MEDIA_TAGS = {0x01: "video", 0x02: "audio"}

VIDEO_BUFFER_BYTES = _env_int("BRIDGE_VIDEO_BUFFER_BYTES", 16 * 1024 * 1024)
AUDIO_BUFFER_BYTES = _env_int("BRIDGE_AUDIO_BUFFER_BYTES", 8 * 1024 * 1024)
MAX_CHUNKS_PER_CHANNEL = _env_int("BRIDGE_MAX_CHUNKS", 4096)
MAX_DRAIN_BYTES = 64 * 1024 * 1024
MAX_VIDEO_FRAME_BYTES = _env_int("BRIDGE_MAX_VIDEO_FRAME_BYTES", 4 * 1024 * 1024)
MAX_VIDEO_FRAME_B64 = (MAX_VIDEO_FRAME_BYTES + 2) // 3 * 4


# --- inbound frames --------------------------------------------------------


class TelemetryFrame(BaseModel):
    """``input`` -> server: one pointer sample."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["telemetry"] = "telemetry"
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    pinch_active: bool
    x2: float | None = Field(default=None, ge=0.0, le=1.0)
    y2: float | None = Field(default=None, ge=0.0, le=1.0)
    pinch2_active: bool | None = None
    ts: int | None = Field(default=None, ge=0)
    seq: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _second_hand(self) -> TelemetryFrame:
        fields = (self.x2, self.y2, self.pinch2_active)
        if any(value is not None for value in fields) and any(value is None for value in fields):
            raise ValueError("x2, y2, and pinch2_active must be sent together")
        return self


class FunctionCallFrame(BaseModel):
    """A call to one of the tools declared in ``x-gemini.tools``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function_call"] = "function_call"
    call_id: str = Field(min_length=1, max_length=128)
    name: str = Field(min_length=1, max_length=64)
    args: dict[str, Any] = Field(default_factory=dict)
    to: ClientId | None = None

    @field_validator("name")
    @classmethod
    def _declared_tool(cls, value: str) -> str:
        if value not in TOOL_NAMES:
            raise ValueError(f"unknown tool {value!r}; expected one of {sorted(TOOL_NAMES)}")
        return value


class VideoFrame(BaseModel):
    """``input`` -> server: one base64-encoded image for the video buffer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["video_frame"] = "video_frame"
    encoding: Literal["jpeg", "png", "webp"] = "jpeg"
    data_base64: str = Field(min_length=4, max_length=MAX_VIDEO_FRAME_B64)
    width: int | None = Field(default=None, ge=1, le=8192)
    height: int | None = Field(default=None, ge=1, le=8192)
    ts: int | None = Field(default=None, ge=0)
    seq: int | None = Field(default=None, ge=0)


class FunctionError(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1)


class FunctionResultFrame(BaseModel):
    """Outcome of a :class:`FunctionCallFrame`, correlated by ``call_id``."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["function_result"] = "function_result"
    call_id: str = Field(min_length=1, max_length=128)
    name: str | None = Field(default=None, min_length=1, max_length=64)
    ok: bool
    result: dict[str, Any] | None = None
    error: FunctionError | None = None
    to: ClientId | None = None

    @model_validator(mode="after")
    def _consistent(self) -> FunctionResultFrame:
        if self.ok and self.error is not None:
            raise ValueError("ok=true must not carry an error")
        if not self.ok and self.error is None:
            raise ValueError("ok=false requires error{code,message}")
        return self


class VoiceStatusFrame(BaseModel):
    """input -> server: microphone is open and the wake listener is running."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["voice_status"] = "voice_status"
    active: bool = True


class SpeechLogFrame(BaseModel):
    """input -> server: spoken text after the wake word (for the surgical log sidebar)."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["speech_log"] = "speech_log"
    phase: Literal["recording", "final", "idle"]
    text: str = Field(min_length=0, max_length=500)

    @model_validator(mode="after")
    def _final_requires_text(self) -> SpeechLogFrame:
        if self.phase == "final" and not self.text.strip():
            raise ValueError("final speech_log requires text")
        return self


# --- connections -----------------------------------------------------------


@dataclass(slots=True)
class Client:
    id: str
    role: str
    websocket: WebSocket
    connected_at_ms: int
    _send_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    async def send_json(self, payload: dict[str, Any]) -> None:
        # One writer at a time: broadcasts and direct replies share the socket.
        async with self._send_lock:
            await self.websocket.send_text(json.dumps(payload, separators=(",", ":")))


class Bridge:
    """Registry of live clients plus fan-out helpers."""

    def __init__(self) -> None:
        self._clients: dict[str, Client] = {}
        self._lock = asyncio.Lock()
        self._seq = itertools.count(1)
        self.last_telemetry: dict[str, Any] | None = None

    @property
    def clients(self) -> list[Client]:
        return list(self._clients.values())

    def get(self, client_id: str) -> Client | None:
        return self._clients.get(client_id)

    def roster(self) -> list[dict[str, Any]]:
        return [
            {"id": client.id, "role": client.role, "connected_at_ms": client.connected_at_ms}
            for client in self._clients.values()
        ]

    async def add(self, client: Client) -> bool:
        """Register ``client``; returns False if its id is already taken."""
        async with self._lock:
            if client.id in self._clients:
                return False
            self._clients[client.id] = client
            return True

    async def remove(self, client_id: str) -> None:
        async with self._lock:
            self._clients.pop(client_id, None)

    def next_seq(self) -> int:
        return next(self._seq)

    async def send(self, client: Client, payload: dict[str, Any]) -> bool:
        try:
            await client.send_json(payload)
            return True
        except Exception as exc:  # socket already gone
            LOG.info("dropping client %s: %s", client.id, exc)
            await self.remove(client.id)
            return False

    async def broadcast(
        self,
        payload: dict[str, Any],
        *,
        exclude: str | None = None,
        roles: Iterable[str] | None = None,
    ) -> int:
        """Send ``payload`` concurrently to all matching clients.

        Returns the number of deliveries; send failures evict the client
        instead of aborting the broadcast.
        """
        allowed = None if roles is None else set(roles)
        targets = [
            client
            for client in self.clients
            if client.id != exclude and (allowed is None or client.role in allowed)
        ]
        if not targets:
            return 0

        results = await asyncio.gather(
            *(client.send_json(payload) for client in targets), return_exceptions=True
        )
        delivered = 0
        for client, result in zip(targets, results):
            if isinstance(result, BaseException):
                LOG.info("dropping client %s: %s", client.id, result)
                await self.remove(client.id)
            else:
                delivered += 1
        return delivered


# --- media buffering -------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MediaChunk:
    seq: int
    received_at_ms: int
    data: bytes

    def as_dict(self, *, include_data: bool) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "seq": self.seq,
            "received_at_ms": self.received_at_ms,
            "bytes": len(self.data),
        }
        if include_data:
            chunk["data_base64"] = base64.b64encode(self.data).decode("ascii")
        return chunk


class ChannelBuffer:
    """Bounded FIFO of media chunks: oldest chunks are evicted under pressure."""

    def __init__(self, name: str, max_bytes: int, max_chunks: int) -> None:
        self.name = name
        self.max_bytes = max_bytes
        self.max_chunks = max_chunks
        self._chunks: deque[MediaChunk] = deque()
        self._bytes = 0
        self._total_chunks = 0
        self._total_bytes = 0
        self._dropped_chunks = 0
        self._dropped_bytes = 0
        self._seq = itertools.count(1)
        self._lock = asyncio.Lock()

    async def append(self, data: bytes, received_at_ms: int | None = None) -> MediaChunk:
        if not data:
            raise ValueError("empty media chunk")
        chunk = MediaChunk(next(self._seq), received_at_ms or _now_ms(), bytes(data))
        async with self._lock:
            self._chunks.append(chunk)
            self._bytes += len(chunk.data)
            self._total_chunks += 1
            self._total_bytes += len(chunk.data)
            while self._chunks and (
                self._bytes > self.max_bytes or len(self._chunks) > self.max_chunks
            ):
                evicted = self._chunks.popleft()
                self._bytes -= len(evicted.data)
                self._dropped_chunks += 1
                self._dropped_bytes += len(evicted.data)
        return chunk

    async def drain(self, max_bytes: int | None = None) -> list[MediaChunk]:
        """Remove and return buffered chunks, oldest first, up to ``max_bytes``.

        Always returns at least one chunk when the buffer is non-empty so a
        single oversized chunk cannot stall the consumer.
        """
        taken: list[MediaChunk] = []
        size = 0
        async with self._lock:
            while self._chunks and (max_bytes is None or size < max_bytes):
                chunk = self._chunks.popleft()
                taken.append(chunk)
                size += len(chunk.data)
                self._bytes -= len(chunk.data)
        return taken

    async def stats(self) -> dict[str, Any]:
        async with self._lock:
            return {
                "name": self.name,
                "chunks": len(self._chunks),
                "bytes": self._bytes,
                "total_chunks": self._total_chunks,
                "total_bytes": self._total_bytes,
                "dropped_chunks": self._dropped_chunks,
                "dropped_bytes": self._dropped_bytes,
                "max_chunks": self.max_chunks,
                "max_bytes": self.max_bytes,
                "oldest_received_at_ms": self._chunks[0].received_at_ms if self._chunks else None,
                "newest_received_at_ms": self._chunks[-1].received_at_ms if self._chunks else None,
            }


class MediaStore:
    def __init__(self) -> None:
        self._buffers = {
            "video": ChannelBuffer("video", VIDEO_BUFFER_BYTES, MAX_CHUNKS_PER_CHANNEL),
            "audio": ChannelBuffer("audio", AUDIO_BUFFER_BYTES, MAX_CHUNKS_PER_CHANNEL),
        }

    def get(self, channel: str) -> ChannelBuffer:
        try:
            return self._buffers[channel]
        except KeyError:
            raise KeyError(channel) from None

    async def add(self, channel: str, data: bytes) -> MediaChunk:
        return await self.get(channel).append(data)

    async def stats(self) -> dict[str, Any]:
        return {name: await buffer.stats() for name, buffer in self._buffers.items()}


bridge = Bridge()
media_store = MediaStore()


# --- frame handlers --------------------------------------------------------


async def send_protocol_error(
    client: Client,
    code: str,
    message: str,
    detail: dict[str, Any] | None = None,
) -> None:
    frame: dict[str, Any] = {
        "type": "error",
        "code": code,
        "message": message,
        "server_ts": _now_ms(),
    }
    if detail is not None:
        frame["detail"] = detail
    await bridge.send(client, frame)


def _validation_detail(exc: ValidationError) -> dict[str, Any]:
    return {
        "errors": [
            {"path": ".".join(str(part) for part in error["loc"]), "message": error["msg"]}
            for error in exc.errors()
        ]
    }


def _origin(role: str) -> str:
    """Map a connection role onto the schema's function-call origin enum."""
    return "gemini" if role == "gemini" else "client"


async def handle_telemetry(client: Client, frame: TelemetryFrame) -> None:
    event = {
        "type": "telemetry",
        "x": frame.x,
        "y": frame.y,
        "pinch_active": frame.pinch_active,
        "x2": frame.x2,
        "y2": frame.y2,
        "pinch2_active": frame.pinch2_active,
        "source": client.id,
        "seq": bridge.next_seq(),
        "client_ts": frame.ts,
        "client_seq": frame.seq,
        "server_ts": _now_ms(),
    }
    bridge.last_telemetry = event
    await bridge.broadcast(event, exclude=client.id)


async def execute_tray_tool(
    name: str,
    args: dict[str, Any],
    call_id: str,
    *,
    origin: str,
) -> dict[str, Any]:
    """Run a tray tool and fan the call plus its result out to viewers."""
    error: dict[str, str] | None = None
    result: dict[str, Any] | None = None
    try:
        result = dispatch(name, args)
        ok = True
        LOG.info("tool %s ok", name)
    except ToolError as exc:
        ok = False
        error = {"code": exc.code, "message": exc.message}
        LOG.info("tool %s rejected: %s", name, exc.message)
    except Exception as exc:
        ok = False
        error = {"code": "internal", "message": str(exc) or "tool failed"}
        LOG.exception("tool %s failed", name)
    call = {
        "type": "function_call",
        "call_id": call_id,
        "name": name,
        "args": args,
        "origin": origin,
        "server_ts": _now_ms(),
    }
    result_frame: dict[str, Any] = {
        "type": "function_result",
        "call_id": call_id,
        "name": name,
        "ok": ok,
        "result": result,
        "error": error,
        "origin": "server",
        "server_ts": _now_ms(),
    }
    await bridge.broadcast(call, roles={"viewer"})
    await bridge.broadcast(result_frame, roles={"viewer"})
    return result_frame


async def log_spoken_command(text: str) -> None:
    """Store a transcript as tray tool calls when Gemini Live is not in the session."""
    for name, args in interpret_log_command(text):
        await execute_tray_tool(name, args, uuid.uuid4().hex, origin="server")


async def handle_function_call(client: Client, frame: FunctionCallFrame) -> None:
    if frame.name in TRAY_TOOLS:
        await execute_tray_tool(frame.name, frame.args, frame.call_id, origin=_origin(client.role))
        return
    call = {
        "type": "function_call",
        "call_id": frame.call_id,
        "name": frame.name,
        "args": frame.args,
        "to": frame.to,
        "origin": _origin(client.role),
        "server_ts": _now_ms(),
    }
    if frame.to is not None:
        target = bridge.get(frame.to)
        if target is None:
            await send_protocol_error(client, "unknown_target", f"no connected client {frame.to!r}")
            return
        await bridge.send(target, call)
    else:
        await bridge.broadcast(call, exclude=client.id)


async def handle_function_result(client: Client, frame: FunctionResultFrame) -> None:
    result = {
        "type": "function_result",
        "call_id": frame.call_id,
        "name": frame.name,
        "ok": frame.ok,
        "result": frame.result,
        "error": frame.error.model_dump() if frame.error else None,
        "to": frame.to,
        "origin": _origin(client.role),
        "server_ts": _now_ms(),
    }
    if frame.to is not None:
        target = bridge.get(frame.to)
        if target is None:
            await send_protocol_error(client, "unknown_target", f"no connected client {frame.to!r}")
            return
        await bridge.send(target, result)
    else:
        await bridge.broadcast(result, exclude=client.id)


async def handle_speech_log(client: Client, frame: SpeechLogFrame) -> None:
    if client.role != "input":
        await send_protocol_error(
            client, "forbidden_role", "speech_log is only accepted from an input client"
        )
        return
    event = {
        "type": "speech_log",
        "phase": frame.phase,
        "text": frame.text,
        "source": client.id,
        "server_ts": _now_ms(),
    }
    await bridge.broadcast(event, exclude=client.id)
    if frame.phase == "final" and not gemini_live.is_active():
        await log_spoken_command(frame.text)


async def handle_voice_status(client: Client, frame: VoiceStatusFrame) -> None:
    if client.role != "input":
        await send_protocol_error(
            client, "forbidden_role", "voice_status is only accepted from an input client"
        )
        return
    await bridge.broadcast(
        {
            "type": "voice_status",
            "active": frame.active,
            "source": client.id,
            "server_ts": _now_ms(),
        },
        exclude=client.id,
        roles={"viewer"},
    )


_image_pick_lock = asyncio.Lock()


async def handle_pick_images(client: Client) -> None:
    """Open the system file window for a viewer pinch on Add images."""
    if client.role != "viewer":
        await send_protocol_error(
            client, "forbidden_role", "pick_images is only accepted from a viewer client"
        )
        return
    if _image_pick_lock.locked():
        await client.send_json(
            {
                "type": "picked_images",
                "files": [],
                "error": "A file window is already open.",
            }
        )
        return
    async with _image_pick_lock:
        try:
            chosen = await asyncio.to_thread(pick_image_paths)
        except Exception as exc:
            LOG.warning("image picker failed: %s", exc)
            await client.send_json(
                {
                    "type": "picked_images",
                    "files": [],
                    "error": "Could not open the file window.",
                }
            )
            return
        if chosen is None:
            await client.send_json({"type": "picked_images", "files": [], "cancelled": True})
            return
        files = await asyncio.to_thread(encode_images, chosen)
        await client.send_json({"type": "picked_images", "files": files})


async def handle_text(client: Client, raw: str) -> None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        await send_protocol_error(client, "invalid_json", f"frame is not valid JSON: {exc.msg}")
        return

    if not isinstance(payload, dict):
        await send_protocol_error(client, "invalid_payload", "expected a JSON object")
        return

    kind = payload.get("type")
    try:
        if kind == "telemetry":
            await handle_telemetry(client, TelemetryFrame.model_validate(payload))
        elif kind == "function_call":
            await handle_function_call(client, FunctionCallFrame.model_validate(payload))
        elif kind == "function_result":
            await handle_function_result(client, FunctionResultFrame.model_validate(payload))
        elif kind == "video_frame":
            await handle_video_frame(client, VideoFrame.model_validate(payload))
        elif kind == "speech_log":
            await handle_speech_log(client, SpeechLogFrame.model_validate(payload))
        elif kind == "voice_status":
            await handle_voice_status(client, VoiceStatusFrame.model_validate(payload))
        elif kind == "start_recording":
            if client.role == "viewer":
                await bridge.broadcast({"type": "start_recording"}, exclude=client.id, roles={"input"})
                return
            else:
                await send_protocol_error(client, "forbidden_role", "start_recording is only accepted from a viewer client")
                return
        elif kind == "pick_images":
            await handle_pick_images(client)
        else:
            await send_protocol_error(client, "unknown_type", f"unsupported frame type {kind!r}")
    except ValidationError as exc:
        await send_protocol_error(
            client, "invalid_payload", "frame failed validation", _validation_detail(exc)
        )


async def handle_video_frame(client: Client, frame: VideoFrame) -> None:
    try:
        data = base64.b64decode(frame.data_base64, validate=True)
    except (binascii.Error, ValueError):
        await send_protocol_error(
            client, "invalid_payload", "data_base64 is not valid standard base64"
        )
        return
    if not data:
        await send_protocol_error(client, "empty_frame", "video_frame carries no data")
        return
    if len(data) > MAX_VIDEO_FRAME_BYTES:
        await send_protocol_error(
            client,
            "payload_too_large",
            f"decoded frame is {len(data)} bytes; limit is {MAX_VIDEO_FRAME_BYTES}",
        )
        return

    chunk = await media_store.add("video", data)
    LOG.debug(
        "client %s buffered video frame %d (%d bytes, %s)",
        client.id,
        chunk.seq,
        len(chunk.data),
        frame.encoding,
    )


async def handle_binary(client: Client, frame: bytes) -> None:
    if len(frame) < 2:
        await send_protocol_error(
            client, "empty_frame", "binary frame needs a channel byte plus a payload"
        )
        return

    channel = MEDIA_TAGS.get(frame[0])
    if channel is None:
        await send_protocol_error(
            client,
            "unsupported_channel",
            f"unknown channel tag {frame[0]}; expected 1 (video) or 2 (audio)",
        )
        return

    chunk = await media_store.add(channel, frame[1:])
    LOG.debug(
        "client %s buffered %s chunk %d (%d bytes)",
        client.id,
        channel,
        chunk.seq,
        len(chunk.data),
    )


async def snapshot_payload() -> dict[str, Any]:
    return {
        "type": "state_snapshot",
        "telemetry": bridge.last_telemetry,
        "clients": bridge.roster(),
        "buffers": await media_store.stats(),
        "server_ts": _now_ms(),
    }


# --- application -----------------------------------------------------------


async def _drain_audio() -> list[bytes]:
    chunks = await media_store.get("audio").drain()
    return [chunk.data for chunk in chunks]


async def _on_gemini_tool_call(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    responses: list[dict[str, Any]] = []
    for call in calls:
        name = call.get("name")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = uuid.uuid4().hex
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        if not isinstance(name, str) or name not in TRAY_TOOLS:
            responses.append(
                {
                    "id": call_id,
                    "name": name if isinstance(name, str) else "unknown",
                    "response": {"error": f"unsupported tool {name!r}"},
                }
            )
            continue
        frame = await execute_tray_tool(name, args, call_id, origin="gemini")
        if frame["ok"]:
            response: dict[str, Any] = {"result": frame["result"]}
        else:
            response = {"error": (frame.get("error") or {}).get("message", "tool failed")}
        responses.append({"id": call_id, "name": name, "response": response})
    return responses


@asynccontextmanager
async def lifespan(_: FastAPI):
    LOG.info(
        "bridge ready: tools=%s video_buffer=%d audio_buffer=%d max_chunks=%d",
        sorted(TOOL_NAMES),
        VIDEO_BUFFER_BYTES,
        AUDIO_BUFFER_BYTES,
        MAX_CHUNKS_PER_CHANNEL,
    )
    live = asyncio.create_task(
        gemini_live.supervise(
            _drain_audio,
            _on_gemini_tool_call,
            system_instruction=GEMINI_LIVE_SYSTEM_PROMPT,
        ),
        name="gemini-live",
    )
    try:
        yield
    finally:
        live.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await live


app = FastAPI(
    title="Pointer telemetry bridge",
    description="Async WebSocket fan-out for pointer telemetry plus video/audio buffering.",
    version="0.1.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["GET"],
    allow_headers=["*"],
)

def _load_dotenv_file(path: Path) -> None:
    """Fill unset environment variables from a local env file. Never overrides or logs values."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        if stripped.startswith("export "):
            stripped = stripped[len("export ") :].strip()
        name, value = stripped.split("=", 1)
        name = name.strip()
        if not name or name in os.environ:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[name] = value


_load_dotenv_file(Path(__file__).with_name(".env"))

NEARBY_MEDICAL_LIMIT = 5
NEARBY_MEDICAL_ZOOM = "14z"


def _local_ll(lat: float, lng: float) -> str:
    """SerpApi GPS form: @latitude,longitude,zoom."""
    return f"@{lat:.6f},{lng:.6f},{NEARBY_MEDICAL_ZOOM}"


def _coordinate_uule(lat: float, lng: float) -> str:
    """Google coordinate UULE. google_local ignores ll and otherwise uses the proxy IP."""
    payload = (
        "role:1\n"
        "producer:12\n"
        "provenance:0\n"
        f"timestamp:{int(time.time() * 1_000_000)}\n"
        "latlng{\n"
        f"latitude_e7:{int(round(lat * 10_000_000))}\n"
        f"longitude_e7:{int(round(lng * 10_000_000))}\n"
        "}\n"
        "radius:-1\n"
    )
    return "a+" + base64.b64encode(payload.encode()).decode()


def _haversine_miles(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    radius_miles = 3958.8
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lng2 - lng1)
    chord = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * radius_miles * math.asin(math.sqrt(chord))


def _format_distance_miles(miles: float) -> str:
    if miles < 0.1:
        return f"{round(miles * 5280)} ft"
    return f"{miles:.1f} mi"


def _distance_label(item: dict[str, Any], lat: float, lng: float) -> str:
    provided = item.get("distance")
    if isinstance(provided, str) and provided.strip():
        return provided.strip()
    coords = item.get("gps_coordinates")
    if not isinstance(coords, dict):
        return ""
    try:
        place_lat = float(coords["latitude"])
        place_lng = float(coords["longitude"])
    except (KeyError, TypeError, ValueError):
        return ""
    return _format_distance_miles(_haversine_miles(lat, lng, place_lat, place_lng))


def _google_local_search(params: dict[str, Any]) -> dict[str, Any]:
    try:
        from serpapi import GoogleSearch
    except ImportError as exc:
        raise RuntimeError("google-search-results is not installed") from exc
    result = GoogleSearch(params).get_dict()
    if not isinstance(result, dict):
        raise RuntimeError("unexpected nearby medical response")
    return result


_FACILITY_TOKEN_STOP = {
    "medical",
    "center",
    "hospital",
    "clinic",
    "health",
    "healthcare",
    "texas",
    "austin",
    "saint",
    "downtown",
    "online",
    "group",
    "care",
    "the",
    "and",
}
# More specific services win. A bullet is kept only when the phrase appears in
# a public listing, so the panel never invents a specialty.
_SERVICE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("Level I trauma center", r"level\s+i\s+trauma"),
    ("Level II trauma center", r"level\s+ii\s+trauma"),
    ("Level III trauma center", r"level\s+iii\s+trauma"),
    ("Burn center", r"burn\s+(center|unit)"),
    ("Transplant program", r"\btransplant\b"),
    ("Stroke care", r"\bstroke\b"),
    ("Heart and arrhythmia care", r"arrhythmia|cardiology|\bcardiac\b"),
    ("Telemedicine", r"telemedicine|virtual visit|video visit|virtual consultation"),
    ("Neonatal intensive care", r"\bnicu\b|neonatal"),
    ("Labor and delivery", r"labor and delivery|maternity|\bobstetric"),
    ("Cancer care", r"\bcancer\b|oncolog"),
    ("Dialysis", r"\bdialysis\b"),
    ("Pediatric care", r"pediatric"),
    ("Preventive care", r"preventive care"),
    ("Intensive care", r"\bicu\b|intensive care"),
    ("Surgical services", r"surgical services|operating rooms?"),
    ("Orthopedic care", r"orthop"),
    ("24/7 emergency care", r"24/7 emergency"),
    ("Emergency care", r"emergency department|emergency room|emergency care"),
    ("Urgent care", r"urgent care"),
    ("Primary care", r"primary care"),
    ("Behavioral health", r"behavioral health|psychiatr"),
    ("Rehabilitation", r"\brehab"),
    ("Women's health", r"women'?s health"),
)
_TYPE_SERVICE_NOTES = {
    "emergency room": "Emergency room",
    "urgent care center": "Urgent care",
}
_NEARBY_CACHE: dict[tuple[float, float], tuple[float, list[dict[str, Any]]]] = {}
_NEARBY_CACHE_TTL_S = 600


def _facility_tokens(name: str) -> list[str]:
    words = re.findall(r"[A-Za-z][A-Za-z']+", name)
    tokens = [word.lower() for word in words if word.lower() not in _FACILITY_TOKEN_STOP and len(word) >= 4]
    return tokens or [name.lower()]


def _service_notes_from_text(name: str, place_type: str, hours: str, texts: list[str]) -> list[str]:
    """Up to two services named in public text for this facility."""
    blob = "\n".join(texts).lower()
    notes: list[str] = []
    for label, pattern in _SERVICE_PATTERNS:
        if label == "Emergency care" and "24/7 emergency care" in notes:
            continue
        if re.search(pattern, blob) and label not in notes:
            notes.append(label)
        if len(notes) == 2:
            return notes
    type_note = _TYPE_SERVICE_NOTES.get(place_type.strip().lower())
    emergency_named = any("emergency" in note.lower() for note in notes)
    if type_note and type_note not in notes and not (type_note == "Emergency room" and emergency_named):
        notes.append(type_note)
    if (
        len(notes) < 2
        and "24" in hours.lower()
        and "open" in hours.lower()
        and "24/7 emergency care" not in notes
        and "Open 24 hours" not in notes
    ):
        notes.append("Open 24 hours")
    return notes[:2]


def _service_page_texts(name: str, api_key: str) -> list[str]:
    payload = _google_local_search(
        {
            "engine": "google",
            "q": f'"{name}" services',
            "gl": "us",
            "hl": "en",
            "num": 5,
            "api_key": api_key,
        }
    )
    if payload.get("error"):
        return []
    texts: list[str] = []
    description = (payload.get("knowledge_graph") or {}).get("description")
    if isinstance(description, str) and description.strip():
        texts.append(description)
    tokens = _facility_tokens(name)
    for item in payload.get("organic_results") or []:
        if not isinstance(item, dict):
            continue
        snippet = item.get("snippet")
        if not isinstance(snippet, str) or not snippet.strip():
            continue
        haystack = f"{item.get('title') or ''} {snippet}".lower()
        if any(token in haystack for token in tokens):
            texts.append(snippet)
    return texts


def _attach_service_notes(facilities: list[dict[str, str]], api_key: str) -> list[dict[str, Any]]:
    def enrich(facility: dict[str, str]) -> dict[str, Any]:
        try:
            texts = _service_page_texts(facility["name"], api_key)
        except Exception:
            LOG.warning("service note lookup failed for %s", facility["name"])
            texts = []
        return {
            "name": facility["name"],
            "distance": facility["distance"],
            "address": facility["address"],
            "services": _service_notes_from_text(
                facility["name"],
                facility.get("place_type") or "",
                facility.get("hours") or "",
                texts,
            ),
        }

    if not facilities:
        return []
    with ThreadPoolExecutor(max_workers=min(NEARBY_MEDICAL_LIMIT, len(facilities))) as pool:
        return list(pool.map(enrich, facilities))


def _top_local_results(payload: dict[str, Any], lat: float, lng: float) -> list[dict[str, str]]:
    raw = payload.get("local_results")
    if isinstance(raw, dict):
        raw = raw.get("places") or []
    if not isinstance(raw, list):
        return []
    facilities: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("title") or item.get("name") or "").strip()
        if not name:
            continue
        facilities.append(
            {
                "name": name,
                "distance": _distance_label(item, lat, lng),
                "address": str(item.get("address") or "").strip(),
                "place_type": str(item.get("type") or "").strip(),
                "hours": str(item.get("hours") or "").strip(),
            }
        )
        if len(facilities) >= NEARBY_MEDICAL_LIMIT:
            break
    return facilities


@app.websocket("/ws")
async def telemetry_socket(
    websocket: WebSocket,
    role: str = "viewer",
    client_id: str | None = None,
) -> None:
    """Single multiplexed socket for telemetry, function calls and media."""
    await websocket.accept()

    if role not in ROLES:
        await websocket.close(code=1008, reason=f"role must be one of {ROLES}")
        return
    if client_id is not None and not CLIENT_ID_RE.fullmatch(client_id):
        await websocket.close(code=1008, reason="invalid client_id")
        return

    client = Client(
        id=client_id or f"{role}-{uuid.uuid4().hex[:8]}",
        role=role,
        websocket=websocket,
        connected_at_ms=_now_ms(),
    )
    if not await bridge.add(client):
        await websocket.close(code=1008, reason=f"client_id {client.id!r} already connected")
        return

    LOG.info("client %s connected as %s (%d live)", client.id, client.role, len(bridge.clients))
    try:
        await client.send_json(await snapshot_payload())
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            data = message.get("bytes")
            if data is not None:
                await handle_binary(client, data)
                continue
            text = message.get("text")
            if text is not None:
                await handle_text(client, text)
    except WebSocketDisconnect:
        pass
    except Exception:
        LOG.exception("client %s failed", client.id)
    finally:
        await bridge.remove(client.id)
        LOG.info("client %s disconnected (%d live)", client.id, len(bridge.clients))


@app.get("/api/nearby-medical")
async def nearby_medical(
    lat: float = Query(..., ge=-90, le=90),
    lng: float = Query(..., ge=-180, le=180),
) -> dict[str, Any]:
    """Nearest hospitals and clinics for a GPS fix, via SerpApi Google Local."""
    api_key = os.getenv("SERPAPI_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=503, detail="SERPAPI_API_KEY is not configured")
    cache_key = (round(lat, 3), round(lng, 3))
    cached = _NEARBY_CACHE.get(cache_key)
    if cached and time.time() - cached[0] < _NEARBY_CACHE_TTL_S:
        return {"local_results": cached[1]}
    params = {
        "engine": "google_local",
        "q": "hospital clinic",
        "ll": _local_ll(lat, lng),
        "uule": _coordinate_uule(lat, lng),
        "gl": "us",
        "api_key": api_key,
    }
    try:
        payload = await asyncio.to_thread(_google_local_search, params)
    except Exception:
        LOG.exception("nearby medical search failed")
        raise HTTPException(status_code=502, detail="nearby medical search failed") from None
    error = payload.get("error")
    if error:
        if isinstance(error, str) and api_key not in error:
            LOG.warning("nearby medical search rejected: %s", error)
        else:
            LOG.warning("nearby medical search rejected")
        raise HTTPException(status_code=502, detail="nearby medical search failed")
    facilities = await asyncio.to_thread(
        _attach_service_notes, _top_local_results(payload, lat, lng), api_key
    )
    _NEARBY_CACHE[cache_key] = (time.time(), facilities)
    return {"local_results": facilities}


@app.get("/healthz")
async def healthz() -> dict[str, Any]:
    roster = bridge.roster()
    by_role = {role: 0 for role in ROLES}
    for entry in roster:
        by_role[entry["role"]] += 1
    return {
        "status": "ok",
        "clients": {"total": len(roster), "by_role": by_role},
        "buffers": await media_store.stats(),
    }


@app.get("/api/state")
async def state() -> dict[str, Any]:
    return {
        "telemetry": bridge.last_telemetry,
        "clients": bridge.roster(),
        "buffers": await media_store.stats(),
        "server_ts": _now_ms(),
    }


@app.get("/api/gemini/tools")
async def gemini_tools() -> dict[str, Any]:
    """Tool declarations ready to drop into a Gemini Live session."""
    return {"tools": GEMINI_TOOLS}


@app.get("/api/media/stats")
async def media_stats() -> dict[str, Any]:
    return await media_store.stats()


@app.post("/api/media/{channel}/drain")
async def drain_media(
    channel: str,
    max_bytes: int = Query(1024 * 1024, gt=0, le=MAX_DRAIN_BYTES),
    include_data: bool = True,
) -> dict[str, Any]:
    """Take buffered chunks out of a channel, oldest first.

    This is the read side for a Gemini Live media pump: each drain returns the
    chunks accumulated since the previous call, base64-encoded for JSON.
    """
    try:
        buffer = media_store.get(channel)
    except KeyError:
        raise HTTPException(
            status_code=404, detail=f"unknown channel {channel!r}; expected 'video' or 'audio'"
        ) from None

    chunks = await buffer.drain(max_bytes=max_bytes)
    return {
        "channel": channel,
        "count": len(chunks),
        "bytes": sum(len(chunk.data) for chunk in chunks),
        "chunks": [chunk.as_dict(include_data=include_data) for chunk in chunks],
        "server_ts": _now_ms(),
    }


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=os.getenv("BRIDGE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    uvicorn.run(
        app,
        host=os.getenv("BRIDGE_HOST", "127.0.0.1"),
        port=int(os.getenv("BRIDGE_PORT", "8000")),
    )
