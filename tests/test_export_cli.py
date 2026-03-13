import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


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

    def test_resolve_output_path_defaults_to_onnx_suffix(self):
        import export

        output = export.resolve_output_path(Path("checkpoints/DAUB-R.pt"), None)

        self.assertEqual(output, Path("checkpoints/DAUB-R.onnx"))

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


if __name__ == "__main__":
    unittest.main()
