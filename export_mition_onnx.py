"""Export the standalone OnnxMotionMapCoreModule to ONNX."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

from export import normalize_onnx_ir_version


def _load_motion_map_module():
    module_path = Path(__file__).resolve().parent / "ultralytics" / "data" / "motion_map.py"
    spec = importlib.util.spec_from_file_location("motion_map_export_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOTION_MAP = _load_motion_map_module()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OnnxMotionMapCoreModule to ONNX.")
    parser.add_argument("--output", default="motion_map.onnx", help="Output ONNX path.")
    parser.add_argument("--imgsz", type=int, default=512, help="Static square input size.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    output = Path(args.output).expanduser().resolve()

    model = MOTION_MAP.OnnxMotionMapCoreModule().eval()
    frame = torch.zeros((1, 1, args.imgsz, args.imgsz), dtype=torch.float32)
    adapt_state = torch.zeros((1, 1, args.imgsz, args.imgsz), dtype=torch.float32)
    memory_state = torch.zeros((1, 1, args.imgsz, args.imgsz), dtype=torch.float32)
    state_valid = torch.zeros((1, 1, 1, 1), dtype=torch.float32)

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (frame, adapt_state, memory_state, state_valid),
        str(output),
        input_names=["frame", "adapt_state", "memory_state", "state_valid"],
        output_names=["motion_core", "next_adapt_state", "next_memory_state"],
        opset_version=args.opset,
        do_constant_folding=True,
    )
    normalize_onnx_ir_version(output)
    return output


if __name__ == "__main__":
    main()
