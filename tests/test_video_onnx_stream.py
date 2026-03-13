import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import torch


def write_test_frame(path: Path, offset: int) -> None:
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    frame[:, 4 + offset : 8 + offset] = 255
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), frame)


class VideoOnnxStreamCliTests(unittest.TestCase):
    def test_parse_args_uses_expected_defaults(self):
        import video_onnx_stream

        args = video_onnx_stream.parse_args(["--source", "frames"])

        self.assertEqual(args.weights, "checkpoints/DAUB-R.onnx")
        self.assertEqual(args.source, "frames")
        self.assertFalse(args.recursive)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.conf, 0.25)
        self.assertEqual(args.project, "runs/video_onnx_stream")
        self.assertEqual(args.name, "mi-detr")
        self.assertFalse(args.exist_ok)

    def test_main_rejects_non_directory_source(self):
        import video_onnx_stream

        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "clip.mp4"
            source.write_bytes(b"video")

            with self.assertRaisesRegex(ValueError, "仅支持帧目录"):
                video_onnx_stream.main(["--source", str(source)])


class VideoPipelineStreamTests(unittest.TestCase):
    def test_collect_stream_groups_returns_grouped_frames(self):
        import video_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            write_test_frame(root / "seq_a" / "000001.png", 0)
            write_test_frame(root / "seq_a" / "000002.png", 1)
            write_test_frame(root / "seq_b" / "000001.png", 2)

            groups = video_pipeline.collect_stream_groups(root, recursive=True)

            self.assertEqual(len(groups), 2)
            self.assertEqual(groups[0][0].as_posix(), "seq_a")
            self.assertEqual(groups[1][0].as_posix(), "seq_b")

    def test_motion_stream_state_initializes_and_resets(self):
        import video_pipeline

        frame = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        state = video_pipeline.reset_motion_stream_state(frame)

        self.assertEqual(tuple(state.adapt_state.shape), (1, 1, 16, 16))
        self.assertTrue(torch.equal(state.adapt_state, torch.zeros_like(frame)))
        self.assertTrue(torch.equal(state.memory_state, torch.zeros_like(frame)))
        self.assertTrue(torch.equal(state.state_valid, torch.zeros((1, 1, 1, 1), dtype=frame.dtype)))

    def test_generate_stream_motion_frame_updates_state(self):
        import video_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            first = root / "000001.png"
            second = root / "000002.png"
            write_test_frame(first, 0)
            write_test_frame(second, 1)

            module = video_pipeline.create_motion_stream_module()
            _, _, first_tensor = video_pipeline.load_stream_frame(first)
            state = video_pipeline.reset_motion_stream_state(first_tensor)

            _, _, first_motion, state = video_pipeline.generate_stream_motion_frame(first, module, state)
            _, _, second_motion, state = video_pipeline.generate_stream_motion_frame(second, module, state)

            self.assertEqual(first_motion.shape, (16, 16, 3))
            self.assertEqual(second_motion.shape, (16, 16, 3))
            self.assertEqual(float(state.state_valid.item()), 1.0)
            self.assertFalse(torch.equal(state.adapt_state, torch.zeros_like(state.adapt_state)))

    def test_preprocess_stream_frame_returns_expected_tensor(self):
        import video_pipeline

        appearance = np.full((16, 16, 3), 255, dtype=np.uint8)
        motion = np.zeros((16, 16, 3), dtype=np.uint8)

        tensor = video_pipeline.preprocess_stream_frame(appearance, motion, imgsz=32)

        self.assertEqual(tensor.shape, (1, 6, 32, 32))
        self.assertGreaterEqual(float(tensor.min()), 0.0)
        self.assertLessEqual(float(tensor.max()), 1.0)

    def test_postprocess_stream_predictions_maps_boxes_to_original_image(self):
        import video_pipeline

        appearance = np.zeros((20, 40, 3), dtype=np.uint8)
        names = {0: "uav"}
        preds = [np.array([[[0.5, 0.5, 0.25, 0.5, 0.9]]], dtype=np.float32)]

        result = video_pipeline.postprocess_stream_predictions(
            preds=preds,
            appearance_image=appearance,
            frame_path=Path("frame.png"),
            names=names,
            conf=0.25,
        )

        self.assertEqual(len(result.boxes), 1)
        xyxy = result.boxes.xyxy.view(-1).tolist()
        self.assertEqual([round(v, 1) for v in xyxy], [15.0, 5.0, 25.0, 15.0])

    def test_save_stream_outputs_writes_detection_image_and_json(self):
        import video_pipeline

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_path = root / "source" / "seq" / "000001.png"
            write_test_frame(frame_path, 0)
            appearance, _, _ = video_pipeline.load_stream_frame(frame_path)
            boxes = torch.tensor([[1.0, 2.0, 8.0, 9.0, 0.9, 0.0]], dtype=torch.float32)
            result = video_pipeline.build_stream_results(appearance, frame_path, {0: "uav"}, boxes)

            image_path, json_path = video_pipeline.save_stream_outputs(
                result=result,
                output_root=root / "runs",
                source_root=frame_path.parent.parent,
                frame_path=frame_path,
                frame_index=1,
            )

            self.assertTrue(image_path.is_file())
            self.assertTrue(json_path.is_file())
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["frame_index"], 1)
            self.assertEqual(payload["detections"][0]["class_name"], "uav")


class VideoOnnxStreamIntegrationTests(unittest.TestCase):
    def test_main_runs_stream_pipeline_with_single_session(self):
        import video_onnx_stream

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "frames"
            weights = root / "DAUB-R.onnx"
            write_test_frame(source_root / "seq_a" / "000001.png", 0)
            write_test_frame(source_root / "seq_a" / "000002.png", 1)
            write_test_frame(source_root / "seq_b" / "000001.png", 2)
            weights.write_bytes(b"onnx")

            session = MagicMock()
            session.run.return_value = [np.array([[[0.5, 0.5, 0.2, 0.2, 0.9]]], dtype=np.float32)]

            with patch("video_onnx_stream.build_onnx_session", return_value=(session, "images", ["output0"])) as mock_build, patch(
                "video_onnx_stream.load_onnx_class_names", return_value={0: "uav"}
            ), patch("video_onnx_stream.reset_motion_stream_state", wraps=video_onnx_stream.reset_motion_stream_state) as mock_reset:
                save_dir = video_onnx_stream.main(
                    [
                        "--weights",
                        str(weights),
                        "--source",
                        str(source_root),
                        "--recursive",
                        "--project",
                        str(root / "runs"),
                        "--name",
                        "stream",
                        "--exist-ok",
                    ]
                )

            mock_build.assert_called_once_with(weights.resolve(), "cpu")
            self.assertEqual(session.run.call_count, 3)
            self.assertEqual(mock_reset.call_count, 2)
            self.assertTrue((save_dir / "images" / "seq_a" / "000001.png").is_file())
            self.assertTrue((save_dir / "json" / "seq_b" / "000001.json").is_file())


if __name__ == "__main__":
    unittest.main()
