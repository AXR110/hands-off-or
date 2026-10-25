import tempfile
import unittest
from pathlib import Path

from image_picker import CANCELLED, encode_images, paths_from_dialog_output


class TestImagePicker(unittest.TestCase):
    def test_cancel_and_blank_output(self):
        self.assertIsNone(paths_from_dialog_output(""))
        self.assertIsNone(paths_from_dialog_output(CANCELLED))
        self.assertIsNone(paths_from_dialog_output("  \n"))

    def test_paths_keep_spaces(self):
        self.assertEqual(
            paths_from_dialog_output("/tmp/Chest X Ray.png\n/tmp/hand.jpg\n"),
            [Path("/tmp/Chest X Ray.png"), Path("/tmp/hand.jpg")],
        )

    def test_encode_reads_png_and_skips_other_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "scan.png"
            image.write_bytes(b"\x89PNG\r\n")
            note = root / "notes.txt"
            note.write_text("hello", encoding="utf-8")
            encoded = encode_images([image, note, root / "missing.png"])
        self.assertEqual(len(encoded), 1)
        self.assertEqual(encoded[0]["name"], "scan.png")
        self.assertEqual(encoded[0]["type"], "image/png")
        self.assertTrue(encoded[0]["data_base64"])


if __name__ == "__main__":
    unittest.main()
