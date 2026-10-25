import unittest
from speech_log import SpeechLogSession

class TestSpeechLogSession(unittest.TestCase):
    def test_hear_without_wake_word_stays_idle(self):
        session = SpeechLogSession()
        self.assertEqual(session.hear("catalog"), [])
        self.assertEqual(session.phase, "idle")
        self.assertEqual(session.hear("logging the count"), [])
        self.assertEqual(session.phase, "idle")

    def test_hear_log_starts_recording(self):
        session = SpeechLogSession()
        events = session.hear("log")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "recording", "text": ""})
        self.assertEqual(session.phase, "recording")
        self.assertEqual(session.text, "")
        events2 = session.hear("log.")
        self.assertEqual(len(events2), 1)
        self.assertEqual(events2[0], {"type": "speech_log", "phase": "recording", "text": ""})
        self.assertEqual(session.phase, "recording")
        self.assertEqual(session.text, "")

    def test_hear_log_phrase_starts_recording_with_text(self):
        session = SpeechLogSession()
        events = session.hear("Log, add sponge.")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "recording", "text": "add sponge"})
        self.assertEqual(session.phase, "recording")
        self.assertEqual(session.text, "add sponge")
        # still recording, not final
        self.assertNotEqual(session.phase, "final")

    def test_short_gap_does_not_stop(self):
        session = SpeechLogSession()
        session.hear("Log, add sponge.")
        events = session.gap(0.3)
        self.assertEqual(events, [])
        self.assertEqual(session.phase, "recording")

    def test_gap_reaches_threshold_emits_final_then_idle(self):
        session = SpeechLogSession()
        session.hear("Log, add sponge.")
        session.gap(0.3)
        events = session.gap(1.2)  # total 1.5
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "final", "text": "add sponge"})
        self.assertEqual(events[1], {"type": "speech_log", "phase": "idle", "text": ""})
        self.assertEqual(session.phase, "idle")
        self.assertEqual(session.text, "")

    def test_after_idle_hear_without_wake_stays_idle(self):
        session = SpeechLogSession()
        session.hear("Log, add sponge.")
        session.gap(1.5)
        events = session.hear("add sponge")
        self.assertEqual(events, [])
        self.assertEqual(session.phase, "idle")

    def test_accumulates_text_and_finalizes_full(self):
        session = SpeechLogSession()
        self.assertEqual(session.hear("log"), [{"type": "speech_log", "phase": "recording", "text": ""}])
        self.assertEqual(session.hear("add sponge"), [{"type": "speech_log", "phase": "recording", "text": "add sponge"}])
        self.assertEqual(session.hear("and two clamps"), [{"type": "speech_log", "phase": "recording", "text": "add sponge and two clamps"}])
        events = session.gap(1.5)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "final", "text": "add sponge and two clamps"})
        self.assertEqual(events[1], {"type": "speech_log", "phase": "idle", "text": ""})
        self.assertEqual(session.phase, "idle")
        self.assertEqual(session.text, "")

    def test_empty_text_on_gap_emits_only_idle_no_final(self):
        session = SpeechLogSession()
        session.hear("log")
        events = session.gap(1.0)
        self.assertEqual(events, [])
        self.assertEqual(session.phase, "recording")

        events = session.gap(5.0)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "idle", "text": ""})
        self.assertEqual(session.phase, "idle")
        # no final emitted

    def test_words_then_1s_gap_stays_recording(self):
        session = SpeechLogSession()
        session.hear("log add sponge")
        events = session.gap(1.0)
        self.assertEqual(events, [])
        self.assertEqual(session.phase, "recording")

    def test_words_then_short_gap_emits_final(self):
        session = SpeechLogSession()
        session.hear("log add sponge")
        events = session.gap(1.5)
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "final", "text": "add sponge"})
        self.assertEqual(events[1], {"type": "speech_log", "phase": "idle", "text": ""})
        self.assertEqual(session.phase, "idle")

    def test_gap_idle_or_zero_negative(self):
        session = SpeechLogSession()
        self.assertEqual(session.gap(1.0), [])
        session.hear("log")
        self.assertEqual(session.gap(0), [])
        self.assertEqual(session.gap(-1), [])
        self.assertEqual(session.phase, "recording")

    def test_custom_wake_word(self):
        session = SpeechLogSession("note")
        events = session.hear("note this")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], {"type": "speech_log", "phase": "recording", "text": "this"})
        default_session = SpeechLogSession()
        self.assertEqual(default_session.hear("note this"), [])

    def test_hear_resets_silence_gap(self):
        session = SpeechLogSession()
        session.hear("log add sponge")
        session.gap(0.5)
        session.hear("now")
        events = session.gap(0.5)
        self.assertEqual(events, [])  # not yet final
        events2 = session.gap(1.0)
        self.assertEqual(len(events2), 2)
        self.assertEqual(events2[0], {"type": "speech_log", "phase": "final", "text": "add sponge now"})
        self.assertEqual(events2[1], {"type": "speech_log", "phase": "idle", "text": ""})

if __name__ == "__main__":
    unittest.main()
