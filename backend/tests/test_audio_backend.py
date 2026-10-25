import unittest

import voice


class TestAudioBackend(unittest.TestCase):
    def test_microphone_is_sounddevice_pcm16(self):
        mic = voice.SoundDeviceMicrophone()
        self.assertIsInstance(mic, voice.sr.AudioSource)
        self.assertEqual(mic.SAMPLE_RATE, voice.SAMPLE_RATE)
        self.assertEqual(mic.SAMPLE_WIDTH, 2)
        self.assertIsNone(mic.stream)
