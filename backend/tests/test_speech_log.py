import unittest
from speech_log import speech_log_from_transcript

class TestSpeechLog(unittest.TestCase):
    def test_log_add_sponge(self):
        result = speech_log_from_transcript("log add sponge")
        self.assertEqual(result, {"type": "speech_log", "phase": "final", "text": "add sponge"})

    def test_log_comma_add_sponge(self):
        result = speech_log_from_transcript("Log, add sponge.")
        self.assertEqual(result, {"type": "speech_log", "phase": "final", "text": "add sponge"})

    def test_please_log_two_clamps(self):
        result = speech_log_from_transcript("please log two clamps")
        self.assertEqual(result, {"type": "speech_log", "phase": "final", "text": "two clamps"})

    def test_log_only(self):
        for phrase in ["log", "log."]:
            with self.subTest(phrase=phrase):
                result = speech_log_from_transcript(phrase)
                self.assertEqual(result, {"type": "speech_log", "phase": "recording", "text": ""})

    def test_no_match_cases(self):
        for phrase in ["catalog", "logging the count", "", "   "]:
            with self.subTest(phrase=phrase):
                result = speech_log_from_transcript(phrase)
                self.assertIsNone(result)

    def test_armed_add_sponge(self):
        result = speech_log_from_transcript("add sponge.", armed=True)
        self.assertEqual(result, {"type": "speech_log", "phase": "final", "text": "add sponge"})

    def test_armed_log_only(self):
        result = speech_log_from_transcript("log", armed=True)
        self.assertEqual(result, {"type": "speech_log", "phase": "recording", "text": ""})

    def test_armed_whitespace(self):
        result = speech_log_from_transcript("   ", armed=True)
        self.assertIsNone(result)

    def test_custom_wake_word(self):
        result = speech_log_from_transcript("note this", "note")
        self.assertEqual(result, {"type": "speech_log", "phase": "final", "text": "this"})

    def test_default_wake_word_not_match(self):
        result = speech_log_from_transcript("note this")
        self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
