import json
import tempfile
import unittest
from pathlib import Path

from pred import export_prediction_json, resolve_source_paths


class ResolveSourcePathsTests(unittest.TestCase):
    def test_resolve_source_paths_maps_images_to_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "images" / "test" / "sample.png"
            motion = root / "image" / "test" / "sample.png"
            source.parent.mkdir(parents=True)
            motion.parent.mkdir(parents=True)
            source.write_bytes(b"vis")
            motion.write_bytes(b"ir")

            resolved_source, resolved_motion = resolve_source_paths(str(source))

            self.assertEqual(resolved_source, source.resolve())
            self.assertEqual(resolved_motion, motion.resolve())

    def test_resolve_source_paths_requires_images_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "foo" / "sample.png"
            source.parent.mkdir(parents=True)
            source.write_bytes(b"vis")

            with self.assertRaisesRegex(ValueError, "images/"):
                resolve_source_paths(str(source))


class ExportPredictionJsonTests(unittest.TestCase):
    def test_export_prediction_json_writes_expected_structure(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "result.json"
            source = Path(tmpdir) / "images" / "test" / "sample.png"
            motion = Path(tmpdir) / "image" / "test" / "sample.png"
            payload = {
                "source_image": str(source),
                "motion_image": str(motion),
                "names": {"0": "target"},
                "appearance_detections": [
                    {
                        "class_id": 0,
                        "class_name": "target",
                        "confidence": 0.9,
                        "xyxy": [1.0, 2.0, 3.0, 4.0],
                    }
                ],
                "motion_detections": [],
            }

            export_prediction_json(output, payload)

            saved = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(saved, payload)


if __name__ == "__main__":
    unittest.main()
