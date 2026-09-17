"""
tests/test_ambient_audio.py
─────────────────────────────────────────────────────────────────
Unit tests for Ambient Audio Chime Engine.
"""
import unittest
from src.ambient_audio import generate_chime_wav, get_chime_base64_data_uri, get_web_audio_inline_js


class TestAmbientAudio(unittest.TestCase):

    def test_wav_generation_all_types(self):
        chime_types = ["bollinger_breakout", "credit_divergence", "danger_zone", "fomc_shock"]
        for ctype in chime_types:
            wav_bytes = generate_chime_wav(ctype, duration_sec=0.5, sample_rate=16000)
            self.assertTrue(len(wav_bytes) > 44)
            # Verify WAV header
            self.assertEqual(wav_bytes[:4], b"RIFF")
            self.assertEqual(wav_bytes[8:12], b"WAVE")
            self.assertEqual(wav_bytes[12:16], b"fmt ")

    def test_base64_data_uri(self):
        uri = get_chime_base64_data_uri("bollinger_breakout")
        self.assertTrue(uri.startswith("data:audio/wav;base64,"))
        self.assertTrue(len(uri) > 100)

    def test_web_audio_inline_js(self):
        js = get_web_audio_inline_js("credit_divergence", volume=0.5)
        self.assertIn("AudioContext", js)
        self.assertIn("createOscillator", js)
        self.assertIn("130.81", js)


if __name__ == "__main__":
    unittest.main()
