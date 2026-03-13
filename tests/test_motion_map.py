import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch


class MotionMapApiTests(unittest.TestCase):
    @staticmethod
    def load_motion_map_module():
        module_path = Path("ultralytics/data/motion_map.py")
        if not module_path.is_file():
            return None

        spec = importlib.util.spec_from_file_location("motion_map_under_test", module_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    def test_motion_map_module_and_cli_exist(self):
        module_path = Path("ultralytics/data/motion_map.py")
        self.assertTrue(module_path.is_file(), f"Expected {module_path} to exist")
        self.assertTrue(Path("motion_map.py").is_file(), "Expected motion_map.py CLI entrypoint to exist")

    def test_motion_map_public_api_is_exposed(self):
        module = self.load_motion_map_module()
        self.assertIsNotNone(module, "Expected ultralytics/data/motion_map.py to exist")
        self.assertTrue(hasattr(module, "generate_sequence"))
        self.assertTrue(hasattr(module, "OnnxMotionMapCoreModule"))
        self.assertTrue(hasattr(module, "OnnxApproxMotionMapModule"))
        self.assertTrue(hasattr(module, "paper_postprocess"))
        self.assertTrue(hasattr(module, "onnx_approx_postprocess"))


class MotionMapBehaviorTests(unittest.TestCase):
    @staticmethod
    def load_motion_map_module():
        spec = importlib.util.spec_from_file_location("motion_map_under_test", Path("ultralytics/data/motion_map.py"))
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module

    @staticmethod
    def make_edge_frame(size=32):
        frame = torch.zeros(1, size, size, dtype=torch.float32)
        frame[:, :, size // 2 :] = 1.0
        return frame

    @staticmethod
    def make_moving_spot_sequence(length=4, size=32):
        frames = torch.zeros(length, 1, size, size, dtype=torch.float32)
        for index in range(length):
            frames[index, 0, size // 2, 8 + index] = 1.0
        return frames

    def test_gaussian_kernel_is_normalized(self):
        module = self.load_motion_map_module()
        kernel = module.gaussian_kernel2d(1.2)
        self.assertEqual(kernel.shape, (9, 9))
        self.assertAlmostEqual(float(kernel.sum().item()), 1.0, places=5)

    def test_motion_map_defaults_match_paper_parameters(self):
        module = self.load_motion_map_module()
        config = module.MotionMapConfig()

        self.assertAlmostEqual(config.theta_p, 0.1)
        self.assertAlmostEqual(config.g_p, 1.5)
        self.assertAlmostEqual(config.gaussian_sigma, 1.0)
        self.assertAlmostEqual(config.horizontal_inhibition, 0.3)
        self.assertAlmostEqual(config.theta_b, 0.2)
        self.assertAlmostEqual(config.g_b, 2.0)
        self.assertAlmostEqual(config.alpha, 0.8)
        self.assertAlmostEqual(config.beta, 1.2)
        self.assertAlmostEqual(config.theta_m, 0.3)
        self.assertAlmostEqual(config.g_m, 2.5)
        self.assertAlmostEqual(config.gamma_a, 0.5)
        self.assertAlmostEqual(config.gamma_tau, 0.7)
        self.assertAlmostEqual(config.eta_m, 0.7)
        self.assertAlmostEqual(config.mexican_hat_sigma_surround, 2.0)
        self.assertAlmostEqual(config.mexican_hat_surround_weight, 0.5)
        self.assertAlmostEqual(config.bilateral_sigma_color, 0.1)
        self.assertAlmostEqual(config.bilateral_sigma_space, 0.1)

    def test_mexican_hat_kernel_matches_paper_normalization(self):
        module = self.load_motion_map_module()
        kernel = module.mexican_hat_kernel2d(1.0, sigma_surround=2.0, size=5, surround_weight=0.5)
        self.assertEqual(kernel.shape, (5, 5))
        self.assertAlmostEqual(float(kernel.sum().item()), 0.0, places=4)
        self.assertAlmostEqual(float(kernel.abs().sum().item()), 1.0, places=4)
        self.assertGreater(float(kernel[2, 2].item()), 0.0)
        self.assertLess(float(kernel[0, 0].item()), 0.0)

    def test_paper_postprocess_keeps_sparse_response_darker_than_onnx_approx(self):
        module = self.load_motion_map_module()
        config = module.MotionMapConfig()
        smoothing = module.gaussian_kernel2d(config.smoothing_sigma, size=config.smoothing_kernel_size)
        motion = torch.zeros((1, 1, 17, 17), dtype=torch.float32)
        motion[..., 8, 8] = 1.0

        paper = module.paper_postprocess(motion, config)
        approx = module.onnx_approx_postprocess(motion, smoothing, config)

        self.assertLess(float(paper.mean().item()), float(approx.mean().item()))

    def test_sequence_layout_warnings_flag_flat_split_directories(self):
        module = self.load_motion_map_module()
        groups = [(Path("train"), [Path("000001.png"), Path("000002.png")])]

        warnings = module.sequence_layout_warnings(groups)

        self.assertEqual(len(warnings), 1)
        self.assertIn("train", warnings[0])
        self.assertIn("per-sequence", warnings[0])

    def test_generate_sequence_uses_distinct_first_frame_path(self):
        module = self.load_motion_map_module()
        frames = self.make_edge_frame().unsqueeze(0)

        with_sobel = module.generate_sequence(frames, use_sobel_first=True)
        without_sobel = module.generate_sequence(frames, use_sobel_first=False)

        self.assertEqual(with_sobel.shape, frames.shape)
        self.assertFalse(torch.allclose(with_sobel, without_sobel))

    def test_generate_sequence_static_sequence_decays_after_warmup(self):
        module = self.load_motion_map_module()
        frame = self.make_edge_frame()
        frames = frame.unsqueeze(0).repeat(5, 1, 1, 1)

        output = module.generate_sequence(frames)

        self.assertLess(float(output[-1].mean().item()), float(output[0].mean().item()))

    def test_generate_sequence_moving_spot_outperforms_static_spot(self):
        module = self.load_motion_map_module()
        moving = self.make_moving_spot_sequence()
        static = moving.clone()
        static[1:] = static[:1]

        moving_out = module.generate_sequence(moving)
        static_out = module.generate_sequence(static)

        self.assertGreater(float(moving_out[-1].max().item()), float(static_out[-1].max().item()))

    def test_onnx_module_ignores_previous_state_when_invalid(self):
        module = self.load_motion_map_module()
        model = module.OnnxMotionMapCoreModule()
        frame = self.make_edge_frame().unsqueeze(0)
        random_adapt = torch.rand_like(frame)
        random_memory = torch.rand_like(frame)
        invalid = torch.zeros(1, 1, 1, 1, dtype=frame.dtype)

        out_invalid = model(frame, random_adapt, random_memory, invalid)
        out_zero = model(frame, torch.zeros_like(frame), torch.zeros_like(frame), invalid)

        for left, right in zip(out_invalid, out_zero):
            self.assertTrue(torch.allclose(left, right, atol=1e-6, rtol=1e-6))

    @unittest.skipUnless(importlib.util.find_spec("onnx") is not None, "onnx is not installed")
    def test_onnx_module_exports(self):
        module = self.load_motion_map_module()
        model = module.OnnxMotionMapCoreModule().eval()
        frame = self.make_edge_frame().unsqueeze(0)
        adapt = torch.zeros_like(frame)
        memory = torch.zeros_like(frame)
        valid = torch.zeros(1, 1, 1, 1, dtype=frame.dtype)

        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "motion_map.onnx"
            torch.onnx.export(
                model,
                (frame, adapt, memory, valid),
                target,
                input_names=["frame", "adapt_state", "memory_state", "state_valid"],
                output_names=["motion", "next_adapt_state", "next_memory_state"],
                opset_version=13,
            )
            self.assertTrue(target.is_file())

    def test_cli_preserves_tree_and_saves_rgb_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_root = root / "images"
            output_root = root / "image"
            (source_root / "train").mkdir(parents=True)
            (source_root / "test").mkdir(parents=True)

            for relative_path, offset in (("train/0001.png", 0), ("train/0002.png", 1), ("test/0001.png", 2)):
                frame = np.zeros((16, 16, 3), dtype=np.uint8)
                frame[:, 4 + offset : 8 + offset] = 255
                cv2.imwrite(str(source_root / relative_path), frame)

            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path.cwd()) + os.pathsep + env.get("PYTHONPATH", "")
            result = subprocess.run(
                [
                    str(Path(".venv/bin/python")),
                    "motion_map.py",
                    "--source-root",
                    str(source_root),
                    "--output-root",
                    str(output_root),
                    "--mode",
                    "paper_onnx",
                    "--recursive",
                ],
                cwd=Path.cwd(),
                env=env,
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, msg=result.stderr)
            self.assertIn("Generating motion maps", result.stderr)
            for relative_path in ("train/0001.png", "train/0002.png", "test/0001.png"):
                saved = cv2.imread(str(output_root / relative_path), cv2.IMREAD_UNCHANGED)
                self.assertIsNotNone(saved, msg=f"Expected output for {relative_path}")
                self.assertEqual(saved.shape[-1], 3)
                self.assertEqual(saved.dtype, np.uint8)


if __name__ == "__main__":
    unittest.main()
