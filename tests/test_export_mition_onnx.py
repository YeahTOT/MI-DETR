import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch


class ExportMitionOnnxCliTests(unittest.TestCase):
    def test_parse_args_uses_expected_defaults(self):
        import export_mition_onnx

        args = export_mition_onnx.parse_args([])

        self.assertEqual(args.output, "motion_map.onnx")
        self.assertEqual(args.imgsz, 512)
        self.assertEqual(args.opset, 13)

    def test_main_exports_motion_map_module_with_expected_arguments(self):
        import export_mition_onnx

        with tempfile.TemporaryDirectory() as tmpdir:
            output = Path(tmpdir) / "custom_motion_map.onnx"

            with patch("export_mition_onnx.torch.onnx.export") as mock_export, patch(
                "export_mition_onnx.normalize_onnx_ir_version"
            ) as mock_normalize:
                result = export_mition_onnx.main(["--output", str(output), "--imgsz", "320"])

        self.assertEqual(result, output.resolve())
        mock_normalize.assert_called_once_with(output.resolve())
        mock_export.assert_called_once()

        args = mock_export.call_args.args
        kwargs = mock_export.call_args.kwargs

        self.assertEqual(args[2], str(output.resolve()))
        frame, adapt_state, memory_state, state_valid = args[1]
        self.assertEqual(tuple(frame.shape), (1, 1, 320, 320))
        self.assertEqual(tuple(adapt_state.shape), (1, 1, 320, 320))
        self.assertEqual(tuple(memory_state.shape), (1, 1, 320, 320))
        self.assertEqual(tuple(state_valid.shape), (1, 1, 1, 1))
        self.assertEqual(frame.dtype, torch.float32)
        self.assertEqual(kwargs["input_names"], ["frame", "adapt_state", "memory_state", "state_valid"])
        self.assertEqual(kwargs["output_names"], ["motion_core", "next_adapt_state", "next_memory_state"])
        self.assertEqual(kwargs["opset_version"], 13)
