import unittest

import voice


class TestAudioBackend(unittest.TestCase):
    def test_microphone_is_sounddevice_pcm16(self):
        mic = voice.SoundDeviceMicrophone()
        self.assertIsInstance(mic, voice.sr.AudioSource)
        self.assertEqual(mic.SAMPLE_RATE, voice.SAMPLE_RATE)
        self.assertEqual(mic.SAMPLE_WIDTH, 2)
        self.assertIsNone(mic.stream)

    def test_selects_macbook_microphone_ahead_of_iphone(self):
        names = [
            "iPhone Microphone",
            "MacBook Pro Microphone",
            "MacBook Pro Speakers",
        ]
        self.assertEqual(voice.select_builtin_microphone_index(names), 1)

    def test_selects_macbook_air_and_builtin_names(self):
        self.assertEqual(
            voice.select_builtin_microphone_index(["MacBook Air Microphone", "External Headphones"]),
            0,
        )
        self.assertEqual(
            voice.select_builtin_microphone_index(["Built-in Microphone"]),
            0,
        )

    def test_prefers_macbook_array_over_generic_builtin(self):
        names = ["Built-in Microphone", "MacBook Pro Microphone"]
        self.assertEqual(voice.select_builtin_microphone_index(names), 1)

    def test_missing_builtin_microphone_is_an_error(self):
        with self.assertRaises(ValueError):
            voice.select_builtin_microphone_index(["iPhone Microphone", "MacBook Pro Speakers"])

    def test_native_rate_stays_on_44100_or_48000(self):
        self.assertEqual(voice.match_native_sample_rate(48000), 48_000)
        self.assertEqual(voice.match_native_sample_rate(44100.0), 44_100)
        self.assertEqual(voice.match_native_sample_rate(16_000), 48_000)
        self.assertEqual(voice._candidate_sample_rates(44_100), (44_100, 48_000))
