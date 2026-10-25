import unittest

from end_word import EndWordBuffer, extract_command


class TestExtractCommand(unittest.TestCase):
    def test_removes_terminal_end_word(self):
        self.assertEqual(
            extract_command("zoom in over"),
            {"command": "zoom in", "ended": True},
        )

    def test_does_not_match_end_word_in_the_middle(self):
        self.assertEqual(
            extract_command("move it over to the left"),
            {"command": "move it over to the left", "ended": False},
        )

    def test_does_not_match_inside_another_word(self):
        self.assertEqual(
            extract_command("hover"),
            {"command": "hover", "ended": False},
        )

    def test_end_word_alone_has_no_command(self):
        self.assertEqual(extract_command("over"), {"command": "", "ended": True})

    def test_ignores_trailing_punctuation_and_case(self):
        self.assertEqual(
            extract_command("log scalpel, OVER."),
            {"command": "log scalpel", "ended": True},
        )


class TestEndWordBuffer(unittest.TestCase):
    def test_sends_only_after_pause_and_discards_on_timeout(self):
        buffer = EndWordBuffer(silence_timeout_ms=8000, end_word_pause_ms=400)
        buffer.start(0)
        buffer.append("zoom in", 100)
        buffer.append("over", 200)

        self.assertEqual(buffer.poll(599), [])
        self.assertEqual(
            buffer.poll(600),
            [
                {"type": "speech_log", "phase": "final", "text": "zoom in"},
                {"type": "speech_log", "phase": "idle", "text": ""},
            ],
        )
        buffer.start(1000)
        buffer.append("stale command", 1100)
        self.assertEqual(buffer.poll(9100), [{"type": "speech_log", "phase": "idle", "text": ""}])
        self.assertFalse(buffer.active)

    def test_empty_and_duplicate_commands_are_not_submitted(self):
        buffer = EndWordBuffer(end_word_pause_ms=0, debounce_ms=1500)
        buffer.start(0)
        buffer.append("over", 1)
        self.assertEqual(buffer.poll(1), [{"type": "speech_log", "phase": "idle", "text": ""}])

        buffer.start(100)
        buffer.append("log scalpel over", 101)
        buffer.append("over", 102)
        self.assertEqual(buffer.poll(102)[0]["phase"], "final")
        buffer.start(200)
        buffer.append("log scalpel over", 201)
        self.assertEqual(buffer.poll(201), [{"type": "speech_log", "phase": "idle", "text": ""}])


if __name__ == "__main__":
    unittest.main()
