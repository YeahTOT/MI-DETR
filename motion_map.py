"""Offline motion-map generation CLI for MI-DETR."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch
from tqdm import tqdm


def _load_motion_map_module():
    module_path = Path(__file__).resolve().parent / "ultralytics" / "data" / "motion_map.py"
    spec = importlib.util.spec_from_file_location("motion_map_cli_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOTION_MAP = _load_motion_map_module()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate motion maps for MI-DETR.")
    parser.add_argument("--source-root", required=True, help="Appearance-image root, usually an images/ directory.")
    parser.add_argument("--output-root", required=True, help="Output root, usually an image/ directory.")
    parser.add_argument("--mode", choices=("reference", "onnx"), default="reference", help="Generation backend.")
    parser.add_argument("--recursive", action="store_true", help="Traverse source-root recursively.")
    parser.add_argument("--save-rgb", dest="save_rgb", action="store_true", help="Save motion maps as 3-channel PNGs.")
    parser.add_argument(
        "--no-save-rgb",
        dest="save_rgb",
        action="store_false",
        help="Save motion maps as single-channel PNGs.",
    )
    parser.set_defaults(save_rgb=True)
    return parser.parse_args()


def run_reference(groups: list[tuple[Path, list[Path]]], source_root: Path, output_root: Path, save_rgb: bool) -> int:
    written = 0
    total = sum(len(image_paths) for _, image_paths in groups)
    with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
        for _, image_paths in groups:
            frames = MOTION_MAP.load_grayscale_frames(image_paths)
            motion_maps = MOTION_MAP.generate_sequence(frames)
            for image_path, motion_map in zip(image_paths, motion_maps):
                target = output_root / image_path.relative_to(source_root)
                MOTION_MAP.save_motion_image(target, motion_map, save_rgb=save_rgb)
                written += 1
                progress.update(1)
    return written


def run_onnx(groups: list[tuple[Path, list[Path]]], source_root: Path, output_root: Path, save_rgb: bool) -> int:
    written = 0
    module = MOTION_MAP.OnnxMotionMapModule().eval()
    with torch.no_grad():
        total = sum(len(image_paths) for _, image_paths in groups)
        with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
            for _, image_paths in groups:
                frames = MOTION_MAP.load_grayscale_frames(image_paths)
                adapt_state = torch.zeros_like(frames[:1])
                memory_state = torch.zeros_like(frames[:1])
                state_valid = torch.zeros((1, 1, 1, 1), dtype=frames.dtype)

                for image_path, frame in zip(image_paths, frames):
                    motion_map, adapt_state, memory_state = module(
                        frame.unsqueeze(0),
                        adapt_state,
                        memory_state,
                        state_valid,
                    )
                    target = output_root / image_path.relative_to(source_root)
                    MOTION_MAP.save_motion_image(target, motion_map.squeeze(0), save_rgb=save_rgb)
                    state_valid.fill_(1.0)
                    written += 1
                    progress.update(1)
    return written


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    groups = MOTION_MAP.collect_image_groups(source_root, recursive=args.recursive)
    for warning in MOTION_MAP.sequence_layout_warnings(groups):
        print(f"WARNING: {warning}")

    if args.mode == "reference":
        written = run_reference(groups, source_root, output_root, args.save_rgb)
    else:
        written = run_onnx(groups, source_root, output_root, args.save_rgb)

    print(f"Saved {written} motion maps to {output_root}")


if __name__ == "__main__":
    main()
