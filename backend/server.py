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
import os
import re
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal

import gemini_live
from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
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
        gemini_live.supervise(_drain_audio, _on_gemini_tool_call),
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
