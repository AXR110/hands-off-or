"""Webcam hand tracking: pointer telemetry + base64 video frames over WebSocket.

Per captured frame:

    cv2.VideoCapture -> MediaPipe HandLandmarker -> PointerSample
        x, y         = index fingertip (landmark 8), normalized to the frame
        pinch_active = thumb tip (4) to index tip (8) distance, normalized by
                       the wrist (0) -> middle MCP (9) hand length, with
                       hysteresis

The pinch distance is measured in pixels, so it is not distorted by the frame
aspect ratio, and divided by the pixel length of wrist -> middle MCP, so the
threshold does not depend on how far the hand is from the camera. Hysteresis
keeps the boolean from chattering while the fingers hover at the threshold;
calibrate with --pinch-engage / --pinch-release and watch the ratio in the logs.

Outputs, on one socket to the bridge:
    * ``telemetry`` JSON at --telemetry-fps (default 30) for viewer fan-out
    * ``video_frame`` JSON carrying a base64 JPEG at --video-fps (default 1),
      which the server decodes into its video ring buffer

Run with ``python vision.py`` (the server defaults to port 8000). The model is
downloaded on first use if it is missing.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import logging
import math
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from websockets.asyncio.client import connect
from websockets.exceptions import ConnectionClosed

LOG = logging.getLogger("vision")

DEFAULT_MODEL = Path(__file__).with_name("models") / "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)

# Hand landmark indices (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_FINGER_MCP = 9

RECONNECT_DELAY_S = 2.0
MAX_READ_FAILURES = 60  # consecutive dropped reads before giving up (~2 s at camera rate)


# --- hand tracking ---------------------------------------------------------


class HandTracker:
    """MediaPipe Hand Landmarker in VIDEO mode; returns the first hand's landmarks."""

    def __init__(
        self,
        model_path: Path,
        *,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> Sequence[Any] | None:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return result.hand_landmarks[0] if result.hand_landmarks else None

    def close(self) -> None:
        self._landmarker.close()


@dataclass(frozen=True)
class PointerSample:
    x: float
    y: float
    pinch_active: bool
    pinch_ratio: float
    pinch_distance_px: float


class PinchDetector:
    """Hysteresis on the normalized thumb-index distance."""

    def __init__(self, engage: float = 0.25, release: float = 0.40) -> None:
        if not 0.0 < engage < release:
            raise ValueError("pinch thresholds require 0 < engage < release")
        self.engage = engage
        self.release = release
        self.active = False

    def update(self, ratio: float) -> bool:
        if self.active:
            if ratio > self.release:
                self.active = False
        elif ratio < self.engage:
            self.active = True
        return self.active

    def reset(self) -> None:
        self.active = False


def _distance_px(a: Any, b: Any, width: int, height: int) -> float:
    """Euclidean distance between two normalized landmarks, in pixels."""
    return math.hypot((a.x - b.x) * width, (a.y - b.y) * height)


def pointer_sample(
    landmarks: Sequence[Any] | None,
    frame_shape: Sequence[int],
    detector: PinchDetector,
) -> PointerSample | None:
    """Turn one hand's landmarks into a pointer sample, or None without a hand."""
    if not landmarks:
        return None

    height, width = frame_shape[:2]
    thumb, index = landmarks[THUMB_TIP], landmarks[INDEX_TIP]
    distance = _distance_px(thumb, index, width, height)
    hand_length = _distance_px(landmarks[WRIST], landmarks[MIDDLE_FINGER_MCP], width, height)
    ratio = distance / max(hand_length, 1e-6)

    return PointerSample(
        x=min(max(float(index.x), 0.0), 1.0),
        y=min(max(float(index.y), 0.0), 1.0),
        pinch_active=detector.update(ratio),
        pinch_ratio=ratio,
        pinch_distance_px=distance,
    )


# --- capture + transport ---------------------------------------------------


class Camera:
    """Thin wrapper over an OpenCV capture device."""

    def __init__(self, index: int, width: int, height: int) -> None:
        self._capture = cv2.VideoCapture(index)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"cannot open camera {index}; check it is connected and that this "
                "process is allowed to use the camera"
            )
        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always hand out the newest frame

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        return frame if ok else None

    def release(self) -> None:
        self._capture.release()


@dataclass(frozen=True)
class StreamSettings:
    telemetry_fps: float = 30.0
    video_fps: float = 1.0
    jpeg_quality: int = 80

    @property
    def telemetry_interval(self) -> float:
        return 1.0 / self.telemetry_fps

    @property
    def video_interval(self) -> float:
        return 1.0 / self.video_fps


def encode_jpeg_base64(frame_bgr: np.ndarray, quality: int) -> str:
    ok, buffer = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(buffer.tobytes()).decode("ascii")


def _now_ms() -> int:
    return int(time.time() * 1000)


def telemetry_frame(sample: PointerSample, seq: int) -> dict[str, Any]:
    return {
        "type": "telemetry",
        "x": round(sample.x, 5),
        "y": round(sample.y, 5),
        "pinch_active": sample.pinch_active,
        "ts": _now_ms(),
        "seq": seq,
    }


def video_frame(frame_bgr: np.ndarray, settings: StreamSettings, seq: int) -> dict[str, Any]:
    height, width = frame_bgr.shape[:2]
    return {
        "type": "video_frame",
        "encoding": "jpeg",
        "width": int(width),
        "height": int(height),
        "data_base64": encode_jpeg_base64(frame_bgr, settings.jpeg_quality),
        "ts": _now_ms(),
        "seq": seq,
    }


async def stream(
    websocket: Any,
    camera: Any,
    tracker: Any,
    detector: PinchDetector,
    settings: StreamSettings,
) -> None:
    """Push telemetry and video frames until the socket or camera fails."""
    telemetry_seq = 0
    video_seq = 0
    telemetry_sent = 0
    video_sent = 0
    last_telemetry_at = 0.0
    last_video_at = 0.0
    last_report_at = time.monotonic()
    last_sample: PointerSample | None = None
    read_failures = 0

    while True:
        frame = await asyncio.to_thread(camera.read)
        if frame is None:
            read_failures += 1
            if read_failures >= MAX_READ_FAILURES:
                raise RuntimeError("camera stopped delivering frames")
            await asyncio.sleep(0.01)
            continue
        read_failures = 0

        now = time.monotonic()
        sent_telemetry = False
        sent_video = False

        if now - last_telemetry_at >= settings.telemetry_interval:
            landmarks = await asyncio.to_thread(tracker.detect, frame)
            sample = pointer_sample(landmarks, frame.shape, detector)
            last_telemetry_at = now

            if sample is not None:
                last_sample = sample
                telemetry_seq += 1
                telemetry_sent += 1
                await websocket.send(json.dumps(telemetry_frame(sample, telemetry_seq)))
                sent_telemetry = True
                LOG.debug(
                    "hand x=%.3f y=%.3f pinch=%s ratio=%.3f (%.1fpx)",
                    sample.x,
                    sample.y,
                    sample.pinch_active,
                    sample.pinch_ratio,
                    sample.pinch_distance_px,
                )
            elif detector.active and last_sample is not None:
                # The hand left while pinching: release at the last known point so
                # viewers do not keep a stuck pinch.
                detector.reset()
                released = PointerSample(
                    x=last_sample.x,
                    y=last_sample.y,
                    pinch_active=False,
                    pinch_ratio=last_sample.pinch_ratio,
                    pinch_distance_px=last_sample.pinch_distance_px,
                )
                telemetry_seq += 1
                telemetry_sent += 1
                await websocket.send(json.dumps(telemetry_frame(released, telemetry_seq)))
                sent_telemetry = True
                LOG.debug("hand lost; released pinch at %.3f,%.3f", released.x, released.y)

        now = time.monotonic()
        if now - last_video_at >= settings.video_interval:
            video_seq += 1
            video_sent += 1
            await websocket.send(json.dumps(video_frame(frame, settings, video_seq)))
            last_video_at = now
            sent_video = True

        if now - last_report_at >= 5.0:
            LOG.info(
                "sent %d telemetry and %d video frames so far (targets %g / %g fps), pinch=%s",
                telemetry_sent,
                video_sent,
                settings.telemetry_fps,
                settings.video_fps,
                detector.active,
            )
            last_report_at = now

        if not sent_telemetry and not sent_video:
            # Nothing was due (a fast source can outpace the send schedule).
            await asyncio.sleep(0.002)


async def log_server_frames(websocket: Any) -> None:
    """Surface protocol rejections from the bridge instead of ignoring them."""
    try:
        async for message in websocket:
            if not isinstance(message, str):
                continue
            try:
                payload = json.loads(message)
            except json.JSONDecodeError:
                continue
            if payload.get("type") == "error":
                LOG.warning(
                    "server rejected a frame: %s (%s)", payload.get("code"), payload.get("message")
                )
            else:
                LOG.debug("server -> %s", payload.get("type"))
    except ConnectionClosed:
        return


def ensure_model(path: Path) -> Path:
    """Return the landmarker model, downloading it once if it is missing."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("downloading hand landmarker model to %s", path)
    temporary = path.with_name(path.name + ".part")
    with urllib.request.urlopen(MODEL_URL, timeout=120) as response, temporary.open("wb") as out:
        shutil.copyfileobj(response, out)
    temporary.replace(path)
    return path


async def run(args: argparse.Namespace) -> None:
    model_path = ensure_model(Path(args.model))
    camera = Camera(args.camera, args.width, args.height)
    tracker = HandTracker(
        model_path,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    detector = PinchDetector(args.pinch_engage, args.pinch_release)
    settings = StreamSettings(args.telemetry_fps, args.video_fps, args.jpeg_quality)
    LOG.info(
        "camera %d at %dx%d, telemetry %g fps, video %g fps, pinch %.2f..%.2f",
        args.camera,
        args.width,
        args.height,
        settings.telemetry_fps,
        settings.video_fps,
        detector.engage,
        detector.release,
    )

    try:
        while True:
            try:
                async with connect(args.server, max_size=None, ping_interval=20) as websocket:
                    LOG.info("connected to %s", args.server)
                    reader = asyncio.create_task(log_server_frames(websocket))
                    try:
                        await stream(websocket, camera, tracker, detector, settings)
                    finally:
                        reader.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await reader
            except (OSError, ConnectionClosed) as exc:
                LOG.warning("connection lost (%s); retrying in %.0fs", exc, RECONNECT_DELAY_S)
                await asyncio.sleep(RECONNECT_DELAY_S)
    finally:
        camera.release()
        tracker.close()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream hand-tracking pointer telemetry and JPEG frames to the bridge."
    )
    parser.add_argument(
        "--server",
        default="ws://127.0.0.1:8000/ws?role=input&client_id=vision",
        help="WebSocket URL, including the role/client_id query.",
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="Hand landmarker .task file.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--telemetry-fps", type=float, default=30.0, help="Pointer samples per second."
    )
    parser.add_argument("--video-fps", type=float, default=1.0, help="JPEG frames per second.")
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument(
        "--pinch-engage",
        type=float,
        default=0.25,
        help="Normalized thumb-index distance below which a pinch starts.",
    )
    parser.add_argument(
        "--pinch-release",
        type=float,
        default=0.40,
        help="Normalized thumb-index distance above which a pinch ends.",
    )
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)

    if not 0.0 < args.telemetry_fps <= 240.0:
        parser.error("--telemetry-fps must be in (0, 240]")
    if not 0.0 < args.video_fps <= args.telemetry_fps:
        parser.error("--video-fps must be in (0, telemetry fps]")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if not 0.0 < args.pinch_engage < args.pinch_release:
        parser.error("--pinch-engage must be positive and below --pinch-release")
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    return args


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    try:
        asyncio.run(run(args))
    except KeyboardInterrupt:
        LOG.info("stopped")


if __name__ == "__main__":
    main()
