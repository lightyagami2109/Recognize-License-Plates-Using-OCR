"""Small offline checks; no model download needed."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from read_plate import read_plate


class ReadPlateTests(unittest.TestCase):
    def test_image_and_ocr_results(self):
        with TemporaryDirectory() as folder:
            image = Path(folder) / "plate.png"
            self.assertTrue(cv2.imwrite(str(image), np.full((80, 320, 3), 255, np.uint8)))
            with patch("read_plate.easyocr.Reader") as reader:
                reader.return_value.recognize.return_value = [
                    ([], "mh 12 ab 1234", 0.92), ([], " ", 0.0),
                ]
                self.assertEqual(read_plate(image), [("MH12AB1234", 0.92)])
                self.assertFalse(reader.call_args.kwargs["gpu"])
                self.assertFalse(reader.call_args.kwargs["detector"])
                self.assertEqual(reader.return_value.recognize.call_args.args[0].ndim, 2)
                reader.return_value.recognize.return_value = []
                self.assertEqual(read_plate(image), [])

    def test_invalid_images_do_not_load_models(self):
        with TemporaryDirectory() as folder, patch("read_plate.easyocr.Reader") as reader:
            image = Path(folder) / "missing.png"
            with self.assertRaisesRegex(ValueError, "not found"):
                read_plate(image)
            image.write_text("not an image")
            with self.assertRaisesRegex(ValueError, "Cannot decode"):
                read_plate(image)
            reader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
