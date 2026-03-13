import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np


def write_test_frame(path: Path, offset: int) -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:, 4 + offset : 8 + offset] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


class VideoOnnxCliTests(unittest.TestCase):
    def test_parse_args_uses_expected_defaults(self):
        import video_onnx

        args = video_onnx.parse_args(["--source", "frames"])

        self.assertEqual(args.weights, "checkpoints/DAUB-R.onnx")
        self.assertEqual(args.source, "frames")
        self.assertEqual(args.motion_mode, "reference")
        self.assertFalse(args.recursive)
        self.assertEqual(args.device, "")
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.project, "runs/video_onnx")
        self.assertEqual(args.name, "mi-detr")
        self.assertFalse(args.exist_ok)
        self.assertTrue(args.save_rgb)

    def test_main_rejects_non_directory_source(self):
        import video_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "clip.mp4"
            source.write_bytes(b"video")

            with self.assertRaisesRegex(ValueError, "仅支持帧目录"):
                video_onnx.main(["--source", str(source)])


class VideoPipelineTests(unittest.TestCase):
    def test_prepare_images_from_directory_preserves_tree(self):
        import video_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "frames"
            images_root = root / "run" / "input" / "images"
            write_test_frame(source_root / "seq_a" / "000001.png", 0)
            write_test_frame(source_root / "seq_a" / "000002.png", 1)
            write_test_frame(source_root / "seq_b" / "000001.png", 2)

            copied = video_pipeline.prepare_images_from_directory(source_root, images_root, recursive=True)

            self.assertEqual(len(copied), 3)
            self.assertTrue((images_root / "seq_a" / "000001.png").is_file())
            self.assertTrue((images_root / "seq_a" / "000002.png").is_file())
            self.assertTrue((images_root / "seq_b" / "000001.png").is_file())

    def test_generate_motion_maps_onnx_writes_one_file_per_frame(self):
        import video_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            images_root = root / "input" / "images"
            motion_root = root / "input" / "image"
            write_test_frame(images_root / "seq" / "000001.png", 0)
            write_test_frame(images_root / "seq" / "000002.png", 1)
            write_test_frame(images_root / "seq" / "000003.png", 2)

            written = video_pipeline.generate_motion_maps(
                images_root=images_root,
                motion_root=motion_root,
                mode="onnx",
                recursive=True,
                save_rgb=True,
            )

            self.assertEqual(written, 3)
            self.assertTrue((motion_root / "seq" / "000001.png").is_file())
            self.assertTrue((motion_root / "seq" / "000002.png").is_file())
            self.assertTrue((motion_root / "seq" / "000003.png").is_file())

    def test_build_source_list_file_writes_sorted_paths(self):
        import video_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_a = root / "images" / "000002.png"
            frame_b = root / "images" / "000001.png"
            frame_a.parent.mkdir(parents=True, exist_ok=True)
            frame_a.write_bytes(b"a")
            frame_b.write_bytes(b"b")

            output = video_pipeline.build_source_list_file([frame_a, frame_b], root / "source.txt")

            self.assertEqual(
                output.read_text(encoding="utf-8").splitlines(),
                [str(frame_b), str(frame_a)],
            )

    def test_video_onnx_main_runs_predict_with_expected_parameters(self):
        import video_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "frames"
            weights = root / "DAUB-R.onnx"
            write_test_frame(source_root / "seq" / "000001.png", 0)
            write_test_frame(source_root / "seq" / "000002.png", 1)
            weights.write_bytes(b"onnx")

            model = MagicMock()
            model.predictor.save_dir = root / "runs" / "video_onnx" / "mi-detr"
            model.predict.return_value = iter([object(), object()])

            with patch("video_onnx.RTDETR", return_value=model) as mock_rtdetr:
                save_dir = video_onnx.main(
                    [
                        "--weights",
                        str(weights),
                        "--source",
                        str(source_root),
                        "--recursive",
                        "--project",
                        str(root / "runs" / "video_onnx"),
                        "--name",
                        "mi-detr",
                        "--exist-ok",
                    ]
                )

        mock_rtdetr.assert_called_once_with(str(weights))
        self.assertEqual(model.predict.call_count, 1)
        kwargs = model.predict.call_args.kwargs
        self.assertEqual(kwargs["ch"], 6)
        self.assertTrue(kwargs["save"])
        self.assertTrue(kwargs["stream"])
        self.assertTrue(kwargs["exist_ok"])
        self.assertEqual(save_dir, Path(model.predictor.save_dir).resolve())


if __name__ == "__main__":
    unittest.main()
