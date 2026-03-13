import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch


class ExportCliTests(unittest.TestCase):
    def test_parse_args_uses_expected_defaults(self):
        import export

        args = export.parse_args(["--weights", "checkpoints/DAUB-R.pt"])

        self.assertEqual(args.weights, "checkpoints/DAUB-R.pt")
        self.assertIsNone(args.output)
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.device, "")
        self.assertEqual(args.opset, 13)
        self.assertTrue(args.simplify)
        self.assertFalse(args.dynamic)
        self.assertEqual(args.batch, 1)
        self.assertFalse(args.half)

    def test_parse_args_supports_motion_stream_export(self):
        import export

        args = export.parse_args(["--weights", "checkpoints/DAUB-R.pt", "--with-motion-stream"])

        self.assertTrue(args.with_motion_stream)

    def test_resolve_output_path_defaults_to_onnx_suffix(self):
        import export

        output = export.resolve_output_path(Path("checkpoints/DAUB-R.pt"), None, with_motion_stream=False)

        self.assertEqual(output, Path("checkpoints/DAUB-R.onnx"))

    def test_resolve_output_path_uses_stream_suffix_for_motion_stream_export(self):
        import export

        output = export.resolve_output_path(Path("checkpoints/DAUB-R.pt"), None, with_motion_stream=True)

        self.assertEqual(output, Path("checkpoints/DAUB-R-stream.onnx"))

    def test_main_calls_rtdetr_export_with_expected_arguments(self):
        import export

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "DAUB-R.pt"
            output = root / "DAUB-R.onnx"
            weights.write_bytes(b"weights")

            model = MagicMock()
            model.export.return_value = str(output)

            with patch("export.RTDETR", return_value=model) as mock_rtdetr, patch(
                "export.normalize_onnx_ir_version"
            ) as mock_normalize:
                result = export.main(
                    [
                        "--weights",
                        str(weights),
                        "--output",
                        str(output),
                    ]
                )

        mock_rtdetr.assert_called_once_with(str(weights))
        mock_normalize.assert_called_once_with(output.resolve())
        model.export.assert_called_once_with(
            format="onnx",
            imgsz=512,
            device="",
            opset=13,
            simplify=True,
            dynamic=False,
            batch=1,
            half=False,
        )
        self.assertEqual(result, output)

    def test_main_retries_with_opset_16_for_grid_sampler_export_failure(self):
        import export

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "DAUB-R.pt"
            output = root / "DAUB-R.onnx"
            weights.write_bytes(b"weights")

            model = MagicMock()
            model.export.side_effect = [
                RuntimeError(
                    "Exporting the operator 'aten::grid_sampler' to ONNX opset version 13 is not supported. "
                    "Support for this operator was added in version 16."
                ),
                str(output),
            ]

            with patch("export.RTDETR", return_value=model), patch("export.normalize_onnx_ir_version") as mock_normalize:
                result = export.main(["--weights", str(weights), "--output", str(output)])

        self.assertEqual(model.export.call_count, 2)
        mock_normalize.assert_called_once_with(output.resolve())
        self.assertEqual(model.export.call_args_list[0].kwargs["opset"], 13)
        self.assertEqual(model.export.call_args_list[1].kwargs["opset"], 16)
        self.assertEqual(result, output)

    def test_main_moves_export_when_output_path_differs(self):
        import export

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "DAUB-R.pt"
            exported = root / "nested" / "DAUB-R.onnx"
            output = root / "custom" / "DAUB-R.onnx"
            weights.write_bytes(b"weights")

            model = MagicMock()
            model.export.return_value = str(exported)

            with patch("export.RTDETR", return_value=model), patch("export.shutil.move") as mock_move, patch(
                "export.normalize_onnx_ir_version"
            ) as mock_normalize:
                result = export.main(["--weights", str(weights), "--output", str(output)])

        mock_normalize.assert_called_once_with(exported.resolve())
        mock_move.assert_called_once_with(str(exported.resolve()), str(output))
        self.assertEqual(result, output)

    def test_main_normalizes_exported_onnx_ir_version(self):
        import export

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "DAUB-R.pt"
            output = root / "DAUB-R.onnx"
            weights.write_bytes(b"weights")

            model = MagicMock()
            model.export.return_value = str(output)

            with patch("export.RTDETR", return_value=model), patch("export.normalize_onnx_ir_version") as mock_normalize:
                result = export.main(["--weights", str(weights), "--output", str(output)])

        mock_normalize.assert_called_once_with(output.resolve())
        self.assertEqual(result, output)

    def test_main_uses_motion_stream_export_branch(self):
        import export

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "DAUB-R.pt"
            output = root / "DAUB-R-stream.onnx"
            weights.write_bytes(b"weights")

            with patch("export.export_motion_stream_onnx", return_value=output.resolve()) as mock_export:
                result = export.main(["--weights", str(weights), "--output", str(output), "--with-motion-stream"])

        mock_export.assert_called_once()
        self.assertEqual(result, output)

    def test_main_retries_motion_stream_export_with_opset_16_for_grid_sampler_failure(self):
        import export

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "DAUB-R.pt"
            output = root / "DAUB-R-stream.onnx"
            weights.write_bytes(b"weights")

            with patch(
                "export.export_motion_stream_onnx",
                side_effect=[
                    RuntimeError(
                        "Exporting the operator 'aten::grid_sampler' to ONNX opset version 13 is not supported. "
                        "Support for this operator was added in version 16."
                    ),
                    output.resolve(),
                ],
            ) as mock_export:
                result = export.main(["--weights", str(weights), "--output", str(output), "--with-motion-stream"])

        self.assertEqual(mock_export.call_count, 2)
        self.assertEqual(mock_export.call_args_list[0].kwargs["opset"], 13)
        self.assertEqual(mock_export.call_args_list[1].kwargs["opset"], 16)
        self.assertEqual(result, output)


class MotionStreamWrapperTests(unittest.TestCase):
    def test_wrapper_forward_returns_detector_output_and_next_states(self):
        import export

        class DummyMotion(torch.nn.Module):
            def forward(self, frame, adapt_state, memory_state, state_valid):
                motion = frame[:, :1]
                return motion, adapt_state + 1, memory_state + 2

        class DummyDetector(torch.nn.Module):
            def forward(self, x):
                self.last_input = x
                return torch.ones((x.shape[0], 4, 5), dtype=x.dtype)

        wrapper = export.MotionIntegratedRTDETRWrapper(DummyDetector(), DummyMotion())
        frame = torch.rand((1, 3, 16, 16), dtype=torch.float32)
        adapt = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        memory = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
        valid = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

        output0, next_adapt, next_memory = wrapper(frame, adapt, memory, valid)

        self.assertEqual(tuple(output0.shape), (1, 4, 5))
        self.assertEqual(tuple(next_adapt.shape), (1, 1, 16, 16))
        self.assertEqual(tuple(next_memory.shape), (1, 1, 16, 16))
        self.assertEqual(tuple(wrapper.detector_model.last_input.shape), (1, 6, 16, 16))

    def test_wrapper_invalid_state_matches_zero_state(self):
        import export

        class DummyMotion(torch.nn.Module):
            def forward(self, frame, adapt_state, memory_state, state_valid):
                valid = state_valid.to(frame.dtype)
                motion = frame[:, :1] + adapt_state * valid + memory_state * valid
                return motion, adapt_state + 1, memory_state + 2

        class DummyDetector(torch.nn.Module):
            def forward(self, x):
                return x.mean(dim=(2, 3), keepdim=False).unsqueeze(1)

        wrapper = export.MotionIntegratedRTDETRWrapper(DummyDetector(), DummyMotion())
        frame = torch.rand((1, 3, 8, 8), dtype=torch.float32)
        random_adapt = torch.rand((1, 1, 8, 8), dtype=torch.float32)
        random_memory = torch.rand((1, 1, 8, 8), dtype=torch.float32)
        invalid = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

        out_invalid = wrapper(frame, random_adapt, random_memory, invalid)
        out_zero = wrapper(frame, torch.zeros_like(random_adapt), torch.zeros_like(random_memory), invalid)

        for left, right in zip(out_invalid, out_zero):
            self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
