import unittest
from speech_log import SpeechLogSession

class TestManualRecord(unittest.TestCase):
    def test_fresh_arm_returns_recording_empty(self):
        session = SpeechLogSession()
        events = session.arm()
        self.assertEqual(events, [{"type": "speech_log", "phase": "recording", "text": ""}])
        self.assertEqual(session.phase, "recording")
        self.assertEqual(session.text, "")

    def test_arm_after_hear_keeps_text(self):
        session = SpeechLogSession()
        session.hear("log add sponge")
        self.assertEqual(session.phase, "recording")
        self.assertEqual(session.text, "add sponge")
        events = session.arm()
        self.assertEqual(events, [{"type": "speech_log", "phase": "recording", "text": "add sponge"}])
        self.assertEqual(session.phase, "recording")
        self.assertEqual(session.text, "add sponge")

    def test_arm_does_not_emit_final(self):
        session = SpeechLogSession()
        session.hear("log add sponge")
        events = session.arm()
        self.assertTrue(all(e.get("phase") != "final" for e in events))

if __name__ == "__main__":
    unittest.main()
