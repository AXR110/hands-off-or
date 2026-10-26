import unittest

from hand_lock import (
    HandCandidate,
    HandProfile,
    PoseProfile,
    choose_main_hand,
    decide_hand,
)


FRAME_SHAPE = (720, 1280, 3)


class TestHandLock(unittest.TestCase):
    def test_operator_switches_to_the_pinching_hand(self):
        left = HandCandidate(0, (0.25, 0.5), 100.0, False)
        right = HandCandidate(1, (0.75, 0.5), 100.0, True)
        self.assertIs(
            choose_main_hand([left, right], left.wrist, FRAME_SHAPE),
            right,
        )

    def test_second_hand_matches_operator_pose_wrist(self):
        profile = HandProfile(wrist=(0.25, 0.5), size=100.0)
        operator_pose = PoseProfile(
            left_wrist=(0.25, 0.5),
            right_wrist=(0.75, 0.5),
            shoulder_width=400.0,
        )
        decision = decide_hand(
            wrist=(0.75, 0.5),
            size=100.0,
            profile=profile,
            operator_pose=operator_pose,
            other_poses=(),
            frame_shape=FRAME_SHAPE,
            pose_fresh=True,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "near pose wrist")

    def test_stray_smaller_hand_is_rejected(self):
        profile = HandProfile(wrist=(0.5, 0.5), size=100.0)
        decision = decide_hand(
            wrist=(0.5, 0.5),
            size=50.0,
            profile=profile,
            operator_pose=None,
            other_poses=(),
            frame_shape=FRAME_SHAPE,
            pose_fresh=False,
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.reason, "too small")

    def test_stale_pose_falls_back_to_lenient_size_and_reach(self):
        profile = HandProfile(wrist=(0.5, 0.5), size=100.0)
        stale_pose = PoseProfile(
            left_wrist=(0.99, 0.5),
            right_wrist=(0.99, 0.6),
            shoulder_width=500.0,
        )
        decision = decide_hand(
            wrist=(0.51, 0.5),
            size=120.0,
            profile=profile,
            operator_pose=stale_pose,
            other_poses=(),
            frame_shape=FRAME_SHAPE,
            pose_fresh=False,
        )
        self.assertTrue(decision.accepted)
        self.assertEqual(decision.reason, "size ok")

        pose_only_match = decide_hand(
            wrist=(0.99, 0.5),
            size=100.0,
            profile=profile,
            operator_pose=stale_pose,
            other_poses=(),
            frame_shape=FRAME_SHAPE,
            pose_fresh=False,
        )
        self.assertFalse(pose_only_match.accepted)
        self.assertEqual(pose_only_match.reason, "too far")


if __name__ == "__main__":
    unittest.main()
