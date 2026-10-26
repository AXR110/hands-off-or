"""Webcam hand tracking: pointer telemetry + base64 video frames over WebSocket.

Per captured frame:

    cv2.VideoCapture -> horizontal mirror -> MediaPipe HandLandmarker (2 hands)
        -> PointerSample
        x, y, pinch_active    = first hand's index fingertip (landmark 8)
        x2, y2, pinch2_active = second hand, omitted when only one hand is visible
        pinch             = thumb tip (4) to index tip (8) distance, normalized by
                            the wrist (0) -> middle MCP (9) hand length, with
                            hysteresis, tracked independently per hand

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
import threading
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

from hand_lock import (
    ARM_REACH_RATIO,
    DEBUG_HANDS,
    LOST_GRACE_FRAMES,
    POSE_FPS,
    POSE_INPUT_WIDTH,
    POSE_MAX_AGE_MS,
    SIZE_MAX_RATIO,
    SIZE_MIN_RATIO,
    SWITCH_COOLDOWN_S,
    SWITCH_FRAMES,
    WRIST_MATCH_DISTANCE,
    SWITCH_HOLD_S,
    HandCandidate,
    HandProfile,
    PoseProfile,
    choose_main_hand,
    decide_hand,
    distance_px,
    lenient_hand_check,
    link_operator_pose,
    match_operator_pose,
)

LOG = logging.getLogger("vision")

DEFAULT_MODEL = Path(__file__).with_name("models") / "hand_landmarker.task"
POSE_MODEL = Path(__file__).with_name("models") / "pose_landmarker_lite.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
)

# Hand landmark indices (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
WRIST = 0
THUMB_TIP = 4
INDEX_TIP = 8
MIDDLE_FINGER_MCP = 9
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_WRIST = 15
RIGHT_WRIST = 16
POSE_CONFIDENCE = 0.5
WAKE_HOLD_SECONDS = 1.5
WAKE_MISSED_FRAME_TOLERANCE = 5
NO_HAND_SLEEP_SECONDS = 10.5

RECONNECT_DELAY_S = 2.0
MAX_READ_FAILURES = 60  # consecutive dropped reads before giving up (~2 s at camera rate)


# --- hand tracking ---------------------------------------------------------

# OpenCV hue is 0-179. Blue latex sits in this band; bare skin is near 0-25.
_BLUE_HUE_LOW = 85
_BLUE_HUE_HIGH = 135
_BLUE_MIN_SATURATION = 40
_SKIN_HUE = np.uint8(12)


def _frame_for_landmarker(frame_bgr: np.ndarray) -> np.ndarray:
    """RGB frame for MediaPipe, with blue latex shifted toward skin tone.

    A full red/blue swap makes a glove look warm, but it also turns a bare
    hand blue, so the landmarker drops it. Only saturated blue pixels move.
    """
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    blue = (
        (hue >= _BLUE_HUE_LOW)
        & (hue <= _BLUE_HUE_HIGH)
        & (hsv[:, :, 1] >= _BLUE_MIN_SATURATION)
    )
    if np.any(blue):
        hsv[:, :, 0] = np.where(blue, _SKIN_HUE, hue)
        frame_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


class HandTracker:
    """MediaPipe Hand Landmarker in VIDEO mode; returns up to two hands."""

    def __init__(
        self,
        model_path: Path,
        *,
        min_detection_confidence: float = 0.4,
        min_tracking_confidence: float = 0.4,
    ) -> None:
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=3,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = 0

    def detect(self, frame_bgr: np.ndarray) -> list[Sequence[Any]]:
        rgb = _frame_for_landmarker(frame_bgr)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = max(int(time.monotonic() * 1000), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        return list(result.hand_landmarks)

    def close(self) -> None:
        self._landmarker.close()


@dataclass(frozen=True)
class PoseFrame:
    poses: tuple[PoseProfile, ...]
    captured_at: float
    frame_shape: tuple[int, int]


def _pose_profiles(result: Any, frame_shape: Sequence[int]) -> tuple[PoseProfile, ...]:
    width = frame_shape[1]
    profiles: list[PoseProfile] = []
    for landmarks in result.pose_landmarks:
        selected = [
            landmarks[index]
            for index in (LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_WRIST, RIGHT_WRIST)
        ]
        confidence = min(
            min(float(getattr(point, "visibility", 1.0)), float(getattr(point, "presence", 1.0)))
            for point in selected
        )
        if confidence < POSE_CONFIDENCE:
            continue
        left_shoulder, right_shoulder, left_wrist, right_wrist = selected
        shoulder_width = abs(float(left_shoulder.x) - float(right_shoulder.x)) * width
        if shoulder_width <= 1e-6:
            continue
        profiles.append(
            PoseProfile(
                left_wrist=(float(left_wrist.x), float(left_wrist.y)),
                right_wrist=(float(right_wrist.x), float(right_wrist.y)),
                shoulder_width=shoulder_width,
            )
        )
    return tuple(profiles)


class PoseWorker:
    """Runs the latest offered frame through Pose Landmarker on one worker thread."""

    def __init__(self, model_path: Path) -> None:
        self._model_path = model_path
        self._lock = threading.Lock()
        self._latest_frame: tuple[np.ndarray, tuple[int, int]] | None = None
        self._latest_result: PoseFrame | None = None
        self._pending = threading.Event()
        self._stopping = threading.Event()
        self._thread = threading.Thread(target=self._run, name="pose-landmarker", daemon=True)
        self._thread.start()

    def offer(self, frame_bgr: np.ndarray, *, skip: bool = False) -> None:
        if skip:
            with self._lock:
                self._latest_frame = None
            self._pending.clear()
            return
        height, width = frame_bgr.shape[:2]
        with self._lock:
            self._latest_frame = (frame_bgr.copy(), (height, width))
        self._pending.set()

    def latest(self) -> PoseFrame | None:
        with self._lock:
            return self._latest_result

    def close(self) -> None:
        self._stopping.set()
        self._pending.set()
        self._thread.join()

    def _run(self) -> None:
        landmarker = None
        try:
            options = mp_vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=str(self._model_path)),
                running_mode=mp_vision.RunningMode.VIDEO,
                num_poses=2,
            )
            landmarker = mp_vision.PoseLandmarker.create_from_options(options)
            last_timestamp_ms = 0
            next_inference_at = 0.0
            while not self._stopping.is_set():
                self._pending.wait()
                if self._stopping.is_set():
                    break
                remaining = next_inference_at - time.monotonic()
                if remaining > 0 and self._stopping.wait(remaining):
                    break
                self._pending.clear()
                with self._lock:
                    pending = self._latest_frame
                    self._latest_frame = None
                if pending is None:
                    continue
                frame_bgr, original_shape = pending
                height, width = frame_bgr.shape[:2]
                scaled_height = max(1, round(height * POSE_INPUT_WIDTH / width))
                small = cv2.resize(
                    frame_bgr,
                    (POSE_INPUT_WIDTH, scaled_height),
                    interpolation=cv2.INTER_AREA,
                )
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=np.ascontiguousarray(rgb),
                )
                timestamp_ms = max(int(time.monotonic() * 1000), last_timestamp_ms + 1)
                last_timestamp_ms = timestamp_ms
                result = landmarker.detect_for_video(image, timestamp_ms)
                pose_frame = PoseFrame(
                    _pose_profiles(result, original_shape),
                    time.monotonic(),
                    original_shape,
                )
                with self._lock:
                    self._latest_result = pose_frame
                next_inference_at = time.monotonic() + 1.0 / POSE_FPS
        except Exception:
            LOG.exception("pose landmarker failed; continuing with hand-size and reach matching")
        finally:
            if landmarker is not None:
                landmarker.close()


@dataclass(frozen=True)
class PointerSample:
    x: float
    y: float
    pinch_active: bool
    pinch_ratio: float
    pinch_distance_px: float
    x2: float | None = None
    y2: float | None = None
    pinch2_active: bool | None = None


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


class OneEuroFilter:
    def __init__(
        self,
        *,
        frequency: float = 30.0,
        min_cutoff: float = 1.0,
        beta: float = 0.007,
        derivative_cutoff: float = 1.0,
    ) -> None:
        self.frequency = frequency
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.derivative_cutoff = derivative_cutoff
        self._value: float | None = None
        self._derivative = 0.0
        self._timestamp: float | None = None

    @staticmethod
    def _alpha(cutoff: float, frequency: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        period = 1.0 / frequency
        return 1.0 / (1.0 + tau / period)

    def update(self, value: float, timestamp: float) -> float:
        if self._value is None or self._timestamp is None:
            self._value = value
            self._timestamp = timestamp
            return value
        self.frequency = 1.0 / max(timestamp - self._timestamp, 1e-6)
        derivative = (value - self._value) * self.frequency
        derivative_alpha = self._alpha(self.derivative_cutoff, self.frequency)
        self._derivative += derivative_alpha * (derivative - self._derivative)
        cutoff = self.min_cutoff + self.beta * abs(self._derivative)
        alpha = self._alpha(cutoff, self.frequency)
        self._value += alpha * (value - self._value)
        self._timestamp = timestamp
        return self._value

    def reset(self) -> None:
        self._value = None
        self._derivative = 0.0
        self._timestamp = None


class HandRig:
    """Lock onto the wake hand, then publish it as the primary pointer."""

    def __init__(self, engage: float, release: float) -> None:
        self._detectors = [PinchDetector(engage, release), PinchDetector(engage, release)]
        self._filters = [OneEuroFilter(), OneEuroFilter(), OneEuroFilter(), OneEuroFilter()]
        self._profile: HandProfile | None = None
        self._operator_pose: PoseProfile | None = None
        self._last_main_wrist: tuple[float, float] | None = None
        self._last_secondary_wrist: tuple[float, float] | None = None
        self._locked = False
        self._wake_started_at: float | None = None
        self._wake_missed_frames = 0
        self._lost_frames = 0
        self._no_sample_since: float | None = None
        self._last: PointerSample | None = None
        self._last_main_pinch_at: float | None = None
        self._challenger_since: float | None = None

    @property
    def engage(self) -> float:
        return self._detectors[0].engage

    @property
    def release(self) -> float:
        return self._detectors[0].release

    @property
    def any_pinch(self) -> bool:
        return any(detector.active for detector in self._detectors)

    @property
    def lock_confident(self) -> bool:
        return self._locked and self._lost_frames == 0

    def should_skip_pose(
        self,
        hands: Sequence[Sequence[Any]] | None,
        frame_shape: Sequence[int],
    ) -> bool:
        if not self.lock_confident or self._profile is None or len(hands or []) != 1:
            return False
        candidate, _, _ = self._candidate(0, hands[0], frame_shape, self._detectors[0].engage)
        return lenient_hand_check(
            candidate.wrist, candidate.size, self._profile, frame_shape
        ).accepted

    def update(
        self,
        hands: Sequence[Sequence[Any]] | None,
        frame_shape: Sequence[int],
        pose_frame: PoseFrame | None = None,
        *,
        now: float | None = None,
    ) -> PointerSample | None:
        timestamp = time.monotonic() if now is None else now
        detections = list(hands or [])[:3]
        pose_is_fresh = (
            pose_frame is not None
            and (timestamp - pose_frame.captured_at) * 1000 < POSE_MAX_AGE_MS
        )
        poses = pose_frame.poses if pose_is_fresh and pose_frame is not None else ()
        if pose_is_fresh and pose_frame is not None and self._operator_pose is not None:
            pose_index = match_operator_pose(self._operator_pose, poses, frame_shape)
            if pose_index is not None:
                self._operator_pose = poses[pose_index]

        candidates: list[HandCandidate] = []
        raw_by_index: dict[int, tuple[Sequence[Any], float, float]] = {}
        for index, landmarks in enumerate(detections):
            candidate, ratio, pinch_distance = self._candidate(
                index, landmarks, frame_shape, self._detectors[0].engage
            )
            if self._locked and self._profile is not None:
                operator_pose, other_poses = self._current_operator_pose(poses, frame_shape)
                decision = decide_hand(
                    candidate.wrist,
                    candidate.size,
                    self._profile,
                    operator_pose,
                    other_poses,
                    frame_shape,
                    pose_fresh=pose_is_fresh,
                )
            else:
                decision = None
            if decision is None or decision.accepted:
                candidates.append(candidate)
                raw_by_index[index] = (landmarks, ratio, pinch_distance)
            if DEBUG_HANDS:
                reason = "size ok" if decision is None else decision.reason
                status = "accepted" if decision is None or decision.accepted else "rejected"
                print(f"hand {index}: {status} ({reason})")

        if candidates:
            candidates = self._carry_pinch_state(candidates, raw_by_index, frame_shape)
        hold_previous = (
            self._last_main_pinch_at is not None
            and timestamp - self._last_main_pinch_at < SWITCH_COOLDOWN_S
        )
        main = choose_main_hand(
            candidates, self._last_main_wrist, frame_shape, hold_previous=hold_previous
        )
        if main is not None and self._last_main_wrist is not None and len(candidates) > 1:
            incumbent = min(
                candidates,
                key=lambda c: distance_px(c.wrist, self._last_main_wrist, frame_shape),
            )
            if main.index != incumbent.index:
                if self._challenger_since is None:
                    self._challenger_since = timestamp
                if timestamp - self._challenger_since < SWITCH_HOLD_S:
                    main = incumbent
            else:
                self._challenger_since = None
        else:
            self._challenger_since = None
        if main is None:
            self._lost_frames += 1
            self._wake_missed_frames += 1
            was_pinching = self.any_pinch
            if self._lost_frames >= SWITCH_FRAMES:
                self._reset_motion()
            if self._lost_frames > LOST_GRACE_FRAMES:
                self._last_main_wrist = None
                self._last_secondary_wrist = None
            if self._wake_missed_frames > WAKE_MISSED_FRAME_TOLERANCE:
                self._wake_started_at = None
                self._wake_missed_frames = 0
            sample = self._release_sample(was_pinching)
            self._note_sample(sample, timestamp)
            if (
                self._locked
                and self._no_sample_since is not None
                and timestamp - self._no_sample_since >= NO_HAND_SLEEP_SECONDS
            ):
                self._release_lock()
            return sample

        others = [candidate for candidate in candidates if candidate.index != main.index]
        if self._last_secondary_wrist is not None and others:
            secondary = min(
                others,
                key=lambda candidate: distance_px(
                    candidate.wrist, self._last_secondary_wrist, frame_shape
                ),
            )
        else:
            secondary = min(others, key=lambda candidate: candidate.wrist[0]) if others else None

        previous_main = None
        if self._last_main_wrist is not None and len(candidates) > 1:
            previous_main = min(
                candidates,
                key=lambda candidate: distance_px(
                    candidate.wrist, self._last_main_wrist, frame_shape
                ),
            )
        main_changed = previous_main is not None and previous_main.index != main.index
        if self._last_main_wrist is not None and self._profile is not None:
            nearest_previous_distance = min(
                distance_px(candidate.wrist, self._last_main_wrist, frame_shape)
                for candidate in candidates
            )
            if nearest_previous_distance > 2.0 * self._profile.size:
                main_changed = True
            if self._last_secondary_wrist is not None and len(candidates) == 1:
                distance_to_previous_secondary = distance_px(
                    main.wrist, self._last_secondary_wrist, frame_shape
                )
                distance_to_previous_main = distance_px(
                    main.wrist, self._last_main_wrist, frame_shape
                )
                if distance_to_previous_secondary < distance_to_previous_main:
                    main_changed = True
        if main_changed:
            self._reset_motion()

        main_landmarks, main_ratio, main_distance = raw_by_index[main.index]
        primary_sample = pointer_sample(main_landmarks, frame_shape, self._detectors[0])
        if primary_sample is None:
            raise RuntimeError("selected hand has no pointer landmarks")
        x, y = self._filter_point(primary_sample.x, primary_sample.y, 0, 1, timestamp)
        primary_sample = PointerSample(
            x=x,
            y=y,
            pinch_active=primary_sample.pinch_active,
            pinch_ratio=main_ratio,
            pinch_distance_px=main_distance,
        )

        secondary_sample = None
        if secondary is not None:
            secondary_landmarks, secondary_ratio, secondary_distance = raw_by_index[
                secondary.index
            ]
            secondary_sample = pointer_sample(
                secondary_landmarks, frame_shape, self._detectors[1]
            )
            if secondary_sample is None:
                raise RuntimeError("selected secondary hand has no pointer landmarks")
            x2, y2 = self._filter_point(
                secondary_sample.x, secondary_sample.y, 2, 3, timestamp
            )
            secondary_sample = PointerSample(
                x=x2,
                y=y2,
                pinch_active=secondary_sample.pinch_active,
                pinch_ratio=secondary_ratio,
                pinch_distance_px=secondary_distance,
            )
        else:
            self._detectors[1].reset()
            self._filters[2].reset()
            self._filters[3].reset()

        sample = PointerSample(
            x=primary_sample.x,
            y=primary_sample.y,
            pinch_active=primary_sample.pinch_active,
            pinch_ratio=primary_sample.pinch_ratio,
            pinch_distance_px=primary_sample.pinch_distance_px,
            x2=None if secondary_sample is None else secondary_sample.x,
            y2=None if secondary_sample is None else secondary_sample.y,
            pinch2_active=None if secondary_sample is None else secondary_sample.pinch_active,
        )
        self._last = sample
        self._last_main_wrist = main.wrist
        if sample.pinch_active:
            self._last_main_pinch_at = timestamp
        self._last_secondary_wrist = None if secondary is None else secondary.wrist
        self._lost_frames = 0

        if not self._locked:
            if sample.pinch_active:
                self._wake_missed_frames = 0
                if self._wake_started_at is None:
                    self._wake_started_at = timestamp
                elif timestamp - self._wake_started_at >= WAKE_HOLD_SECONDS:
                    self._locked = True
                    self._profile = HandProfile(main.wrist, main.size)
                    pose_index = link_operator_pose(main.wrist, poses, frame_shape)
                    self._operator_pose = None if pose_index is None else poses[pose_index]
                    self._wake_started_at = None
                    self._reset_motion()
                    sample = self._sample_with_fresh_pinch(
                        main_landmarks,
                        frame_shape,
                        main_ratio,
                        main_distance,
                        secondary,
                        raw_by_index,
                        timestamp,
                    )
                    self._last = sample
            elif self._wake_started_at is not None:
                self._wake_missed_frames += 1
                if self._wake_missed_frames > WAKE_MISSED_FRAME_TOLERANCE:
                    self._wake_started_at = None
                    self._wake_missed_frames = 0
        elif self._profile is not None:
            self._profile = HandProfile(main.wrist, self._profile.size)

        self._note_sample(sample, timestamp)
        return sample

    def _candidate(
        self,
        index: int,
        landmarks: Sequence[Any],
        frame_shape: Sequence[int],
        engage: float,
    ) -> tuple[HandCandidate, float, float]:
        height, width = frame_shape[:2]
        wrist = (float(landmarks[WRIST].x), float(landmarks[WRIST].y))
        size = _distance_px(
            landmarks[WRIST], landmarks[MIDDLE_FINGER_MCP], width, height
        )
        pinch_distance = _distance_px(
            landmarks[THUMB_TIP], landmarks[INDEX_TIP], width, height
        )
        ratio = pinch_distance / max(size, 1e-6)
        candidate = HandCandidate(
            index=index,
            wrist=wrist,
            size=size,
            pinch=ratio < engage,
        )
        return candidate, ratio, pinch_distance

    def _carry_pinch_state(
        self,
        candidates: list[HandCandidate],
        raw_by_index: dict[int, tuple[Sequence[Any], float, float]],
        frame_shape: Sequence[int],
    ) -> list[HandCandidate]:
        previous_roles = (
            (self._last_main_wrist, self._detectors[0]),
            (self._last_secondary_wrist, self._detectors[1]),
        )
        carried: list[HandCandidate] = []
        for candidate in candidates:
            active_detector = None
            if any(wrist is not None for wrist, _ in previous_roles):
                active_detector = min(
                    (
                        (distance_px(candidate.wrist, wrist, frame_shape), detector)
                        for wrist, detector in previous_roles
                        if wrist is not None
                    ),
                    key=lambda item: item[0],
                )[1]
            _, ratio, _ = raw_by_index[candidate.index]
            pinching = candidate.pinch or (
                active_detector is not None
                and active_detector.active
                and ratio <= active_detector.release
            )
            carried.append(
                HandCandidate(
                    index=candidate.index,
                    wrist=candidate.wrist,
                    size=candidate.size,
                    pinch=pinching,
                )
            )
        return carried

    def _current_operator_pose(
        self,
        poses: Sequence[PoseProfile],
        frame_shape: Sequence[int],
    ) -> tuple[PoseProfile | None, tuple[PoseProfile, ...]]:
        if self._operator_pose is None or not poses:
            return None, tuple(poses)
        pose_index = match_operator_pose(self._operator_pose, poses, frame_shape)
        if pose_index is None:
            return None, tuple(poses)
        return poses[pose_index], tuple(
            pose for index, pose in enumerate(poses) if index != pose_index
        )

    def _filter_point(
        self,
        x: float,
        y: float,
        x_filter: int,
        y_filter: int,
        timestamp: float,
    ) -> tuple[float, float]:
        return (
            self._filters[x_filter].update(x, timestamp),
            self._filters[y_filter].update(y, timestamp),
        )

    def _sample_with_fresh_pinch(
        self,
        main_landmarks: Sequence[Any],
        frame_shape: Sequence[int],
        main_ratio: float,
        main_distance: float,
        secondary: HandCandidate | None,
        raw_by_index: dict[int, tuple[Sequence[Any], float, float]],
        timestamp: float,
    ) -> PointerSample:
        primary = pointer_sample(main_landmarks, frame_shape, self._detectors[0])
        if primary is None:
            raise RuntimeError("wake hand has no pointer landmarks")
        x, y = self._filter_point(primary.x, primary.y, 0, 1, timestamp)
        secondary_sample = None
        if secondary is not None:
            landmarks, ratio, pinch_distance = raw_by_index[secondary.index]
            secondary_sample = pointer_sample(landmarks, frame_shape, self._detectors[1])
            if secondary_sample is None:
                raise RuntimeError("selected secondary hand has no pointer landmarks")
            x2, y2 = self._filter_point(
                secondary_sample.x, secondary_sample.y, 2, 3, timestamp
            )
            secondary_sample = PointerSample(
                x=x2,
                y=y2,
                pinch_active=secondary_sample.pinch_active,
                pinch_ratio=ratio,
                pinch_distance_px=pinch_distance,
            )
        return PointerSample(
            x=x,
            y=y,
            pinch_active=primary.pinch_active,
            pinch_ratio=main_ratio,
            pinch_distance_px=main_distance,
            x2=None if secondary_sample is None else secondary_sample.x,
            y2=None if secondary_sample is None else secondary_sample.y,
            pinch2_active=None
            if secondary_sample is None
            else secondary_sample.pinch_active,
        )

    def _reset_motion(self) -> None:
        for detector in self._detectors:
            detector.reset()
        for point_filter in self._filters:
            point_filter.reset()

    def _release_sample(self, was_pinching: bool) -> PointerSample | None:
        self._reset_motion()
        if not was_pinching or self._last is None:
            return None
        released = PointerSample(
            x=self._last.x,
            y=self._last.y,
            pinch_active=False,
            pinch_ratio=self._last.pinch_ratio,
            pinch_distance_px=self._last.pinch_distance_px,
            x2=self._last.x2,
            y2=self._last.y2,
            pinch2_active=False if self._last.x2 is not None else None,
        )
        self._last = released
        return released

    def _note_sample(self, sample: PointerSample | None, timestamp: float) -> None:
        if sample is None:
            if self._no_sample_since is None:
                self._no_sample_since = timestamp
        else:
            self._no_sample_since = None

    def _release_lock(self) -> None:
        self._locked = False
        self._profile = None
        self._operator_pose = None
        self._last_main_wrist = None
        self._last_secondary_wrist = None
        self._lost_frames = 0
        self._wake_started_at = None
        self._wake_missed_frames = 0
        self._no_sample_since = None
        self._reset_motion()
        self._last_main_pinch_at = None
        self._challenger_since = None


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
        if not ok:
            return None
        # A front camera is mirrored relative to the user: image-left is their
        # right. Flip before landmarking so a hand moving right increases x,
        # and so the JPEG sent to the bridge matches the pointer.
        return cv2.flip(frame, 1)

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
    frame: dict[str, Any] = {
        "type": "telemetry",
        "x": round(sample.x, 5),
        "y": round(sample.y, 5),
        "pinch_active": sample.pinch_active,
        "ts": _now_ms(),
        "seq": seq,
    }
    if sample.x2 is not None and sample.y2 is not None and sample.pinch2_active is not None:
        frame["x2"] = round(sample.x2, 5)
        frame["y2"] = round(sample.y2, 5)
        frame["pinch2_active"] = sample.pinch2_active
    return frame


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
    rig: HandRig,
    settings: StreamSettings,
    pose_worker: PoseWorker | None = None,
) -> None:
    """Push telemetry and video frames until the socket or camera fails."""
    telemetry_seq = 0
    video_seq = 0
    telemetry_sent = 0
    video_sent = 0
    last_telemetry_at = 0.0
    last_video_at = 0.0
    last_report_at = time.monotonic()
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
            hands = await asyncio.to_thread(tracker.detect, frame)
            if pose_worker is not None:
                pose_worker.offer(
                    frame,
                    skip=rig.should_skip_pose(hands, frame.shape),
                )
            sample = rig.update(
                hands,
                frame.shape,
                None if pose_worker is None else pose_worker.latest(),
            )
            last_telemetry_at = now

            if sample is not None:
                telemetry_seq += 1
                telemetry_sent += 1
                await websocket.send(json.dumps(telemetry_frame(sample, telemetry_seq)))
                sent_telemetry = True
                LOG.debug(
                    "hands x=%.3f y=%.3f pinch=%s x2=%s y2=%s pinch2=%s",
                    sample.x,
                    sample.y,
                    sample.pinch_active,
                    None if sample.x2 is None else round(sample.x2, 3),
                    None if sample.y2 is None else round(sample.y2, 3),
                    sample.pinch2_active,
                )

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
                rig.any_pinch,
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


def ensure_pose_model(path: Path) -> Path:
    """Return the lite pose model, downloading it if needed."""
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("downloading pose landmarker lite model to %s", path)
    temporary = path.with_name(path.name + ".part")
    with urllib.request.urlopen(POSE_MODEL_URL, timeout=120) as response, temporary.open(
        "wb"
    ) as out:
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
    pose_worker = None
    try:
        pose_worker = PoseWorker(ensure_pose_model(POSE_MODEL))
    except (OSError, urllib.error.URLError) as exc:
        LOG.warning("pose helper unavailable (%s); using hand-size and reach matching", exc)

    rig = HandRig(args.pinch_engage, args.pinch_release)
    settings = StreamSettings(args.telemetry_fps, args.video_fps, args.jpeg_quality)
    LOG.info(
        "camera %d at %dx%d, telemetry %g fps, video %g fps, three hands, pinch %.2f..%.2f",
        args.camera,
        args.width,
        args.height,
        settings.telemetry_fps,
        settings.video_fps,
        rig.engage,
        rig.release,
    )
    LOG.info(
        "operator lock size %.1f..%.1f, reach %.1fx, pose %.0f fps / %d px, wrist match %.1fx, debug=%s",
        SIZE_MIN_RATIO,
        SIZE_MAX_RATIO,
        ARM_REACH_RATIO,
        POSE_FPS,
        POSE_INPUT_WIDTH,
        WRIST_MATCH_DISTANCE,
        DEBUG_HANDS,
    )

    try:
        while True:
            try:
                async with connect(args.server, max_size=None, ping_interval=20) as websocket:
                    LOG.info("connected to %s", args.server)
                    reader = asyncio.create_task(log_server_frames(websocket))
                    try:
                        await stream(websocket, camera, tracker, rig, settings, pose_worker)
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
        if pose_worker is not None:
            pose_worker.close()


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
    parser.add_argument("--min-detection-confidence", type=float, default=0.4)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.4)
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
