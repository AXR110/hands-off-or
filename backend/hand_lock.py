"""Pure hand-lock matching helpers shared by vision.py and its unit tests."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

SIZE_MIN_RATIO = 0.5
SIZE_MAX_RATIO = 1.6
ARM_REACH_RATIO = 6.0
SWITCH_FRAMES = 3
LOST_GRACE_FRAMES = 15
POSE_INPUT_WIDTH = 256
POSE_FPS = 5
POSE_MAX_AGE_MS = 400
WRIST_MATCH_DISTANCE = 1.0
DEBUG_HANDS = False
SWITCH_COOLDOWN_S = 1.0
SWITCH_HOLD_S = 0.4
HOLD_MAX_DISTANCE_RATIO = 2.0


@dataclass(frozen=True)
class HandProfile:
    wrist: tuple[float, float]
    size: float


@dataclass(frozen=True)
class PoseProfile:
    left_wrist: tuple[float, float]
    right_wrist: tuple[float, float]
    shoulder_width: float


@dataclass(frozen=True)
class HandCandidate:
    index: int
    wrist: tuple[float, float]
    size: float
    pinch: bool


@dataclass(frozen=True)
class HandDecision:
    accepted: bool
    reason: str


def distance_px(
    first: tuple[float, float],
    second: tuple[float, float],
    frame_shape: Sequence[int],
) -> float:
    height, width = frame_shape[:2]
    return math.hypot((first[0] - second[0]) * width, (first[1] - second[1]) * height)


def lenient_hand_check(
    wrist: tuple[float, float],
    size: float,
    profile: HandProfile,
    frame_shape: Sequence[int],
) -> HandDecision:
    ratio = size / max(profile.size, 1e-6)
    if ratio < SIZE_MIN_RATIO:
        return HandDecision(False, "too small")
    if ratio > SIZE_MAX_RATIO:
        return HandDecision(False, "too large")
    if distance_px(wrist, profile.wrist, frame_shape) > ARM_REACH_RATIO * profile.size:
        return HandDecision(False, "too far")
    return HandDecision(True, "size ok")


def near_pose_wrist(
    wrist: tuple[float, float],
    pose: PoseProfile,
    frame_shape: Sequence[int],
) -> bool:
    maximum_distance = WRIST_MATCH_DISTANCE * pose.shoulder_width
    return min(
        distance_px(wrist, pose.left_wrist, frame_shape),
        distance_px(wrist, pose.right_wrist, frame_shape),
    ) <= maximum_distance


def decide_hand(
    wrist: tuple[float, float],
    size: float,
    profile: HandProfile,
    operator_pose: PoseProfile | None,
    other_poses: Sequence[PoseProfile],
    frame_shape: Sequence[int],
    *,
    pose_fresh: bool,
) -> HandDecision:
    lenient = lenient_hand_check(wrist, size, profile, frame_shape)
    if lenient.accepted:
        return lenient
    if not pose_fresh or operator_pose is None:
        return lenient
    if near_pose_wrist(wrist, operator_pose, frame_shape):
        return HandDecision(True, "near pose wrist")

    operator_distance = min(
        distance_px(wrist, operator_pose.left_wrist, frame_shape),
        distance_px(wrist, operator_pose.right_wrist, frame_shape),
    )
    other_distance = min(
        (
            min(
                distance_px(wrist, pose.left_wrist, frame_shape),
                distance_px(wrist, pose.right_wrist, frame_shape),
            )
            for pose in other_poses
        ),
        default=math.inf,
    )
    pose_margin = operator_pose.shoulder_width * 0.25
    if other_distance + pose_margin < operator_distance:
        return HandDecision(False, "other pose")
    return lenient


def link_operator_pose(
    wrist: tuple[float, float],
    poses: Sequence[PoseProfile],
    frame_shape: Sequence[int],
) -> int | None:
    if not poses:
        return None
    return min(
        range(len(poses)),
        key=lambda index: min(
            distance_px(wrist, poses[index].left_wrist, frame_shape),
            distance_px(wrist, poses[index].right_wrist, frame_shape),
        ),
    )


def match_operator_pose(
    anchor: PoseProfile,
    poses: Sequence[PoseProfile],
    frame_shape: Sequence[int],
) -> int | None:
    if not poses:
        return None
    return min(
        range(len(poses)),
        key=lambda index: (
            distance_px(anchor.left_wrist, poses[index].left_wrist, frame_shape)
            + distance_px(anchor.right_wrist, poses[index].right_wrist, frame_shape)
        ),
    )


def choose_main_hand(
    hands: Sequence[HandCandidate],
    previous_wrist: tuple[float, float] | None,
    frame_shape: Sequence[int],
    *,
    hold_previous: bool = False,
) -> HandCandidate | None:
    if not hands:
        return None

    pinching = [hand for hand in hands if hand.pinch]
    if previous_wrist is not None:
        previous = min(
            hands,
            key=lambda hand: distance_px(hand.wrist, previous_wrist, frame_shape),
        )
        if previous.pinch:
            return previous
        still_close = (
            distance_px(previous.wrist, previous_wrist, frame_shape)
            <= HOLD_MAX_DISTANCE_RATIO * previous.size
        )
        if hold_previous and still_close:
            return previous
        if pinching:
            return min(
                pinching,
                key=lambda hand: distance_px(hand.wrist, previous_wrist, frame_shape),
            )
        return previous
    if pinching:
        return min(pinching, key=lambda hand: hand.wrist[0])
    return min(hands, key=lambda hand: hand.wrist[0])