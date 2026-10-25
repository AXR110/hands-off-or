import sys
import unittest

class TestAudioBackend(unittest.TestCase):
    def test_ensure_pyaudio_lazy_load(self):
        if "pyaudio" in sys.modules:
            del sys.modules["pyaudio"]
        import voice
        voice._ensure_pyaudio()
        import pyaudio
        self.assertTrue(hasattr(pyaudio, "PyAudio"))
