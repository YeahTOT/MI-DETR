import argparse
import importlib.util
import shutil
import sys
import warnings
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

from ultralytics import RTDETR
from ultralytics.data.utils import VID_FORMATS
from ultralytics.utils.files import increment_path

warnings.filterwarnings("ignore")

DEFAULT_WEIGHTS = "checkpoints/DAUB-R.pt"


def _load_motion_map_module():
    module_path = Path(__file__).resolve().parent / "ultralytics" / "data" / "motion_map.py"
    spec = importlib.util.spec_from_file_location("video_motion_map_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOTION_MAP = _load_motion_map_module()


def parse_args():
    parser = argparse.ArgumentParser(description="Run MI-DETR inference for frame directories or MP4 videos.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Checkpoint path.")
    parser.add_argument("--source", required=True, help="Frame directory or MP4 video path.")
    parser.add_argument("--motion-mode", choices=("reference", "onnx"), default="reference", help="Motion-map backend.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan frame directories.")
    parser.add_argument("--device", default="", help="Prediction device, e.g. '0'.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--project", default="runs/video", help="Directory to save prediction runs.")
    parser.add_argument("--name", default="mi-detr", help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the run directory if it already exists.")
    parser.add_argument("--frame-ext", default="jpg", help="Frame extension for extracted videos.")
    parser.add_argument("--save-rgb", dest="save_rgb", action="store_true", help="Save motion maps as 3-channel PNGs.")
    parser.add_argument(
        "--no-save-rgb",
        dest="save_rgb",
        action="store_false",
        help="Save motion maps as single-channel PNGs.",
    )
    parser.set_defaults(save_rgb=True)
    return parser.parse_args()


def is_video_file(path: Path) -> bool:
    return path.suffix.lower().lstrip(".") in VID_FORMATS


def prepare_images_from_directory(source_dir: Path, images_root: Path, recursive: bool) -> list[Path]:
    groups = MOTION_MAP.collect_image_groups(source_dir, recursive=recursive)
    copied: list[Path] = []
    for _, image_paths in groups:
        for image_path in image_paths:
            target = images_root / image_path.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image_path, target)
            copied.append(target)
    return copied


def prepare_images_from_video(video_path: Path, images_root: Path, frame_ext: str) -> list[Path]:
    suffix = frame_ext.lower().lstrip(".")
    if not suffix:
        raise ValueError("frame-ext must be a valid extension like 'jpg' or 'png'.")

    sequence_dir = images_root / video_path.stem
    sequence_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    frame_paths: list[Path] = []
    frame_id = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame_id += 1
        frame_path = sequence_dir / f"{frame_id:06d}.{suffix}"
        if not cv2.imwrite(str(frame_path), frame):
            raise OSError(f"Failed to write frame: {frame_path}")
        frame_paths.append(frame_path)
    cap.release()

    if not frame_paths:
        raise RuntimeError(f"No frames extracted from video: {video_path}")
    return frame_paths


def build_source_list_file(frame_paths: list[Path], list_path: Path) -> Path:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(str(path) for path in sorted(frame_paths))
    list_path.write_text(payload, encoding="utf-8")
    return list_path


def generate_motion_maps(
    images_root: Path,
    motion_root: Path,
    mode: str,
    recursive: bool,
    save_rgb: bool,
) -> int:
    groups = MOTION_MAP.collect_image_groups(images_root, recursive=recursive)
    for warning in MOTION_MAP.sequence_layout_warnings(groups):
        print(f"WARNING: {warning}")

    written = 0
    total = sum(len(image_paths) for _, image_paths in groups)
    if mode == "reference":
        with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
            for _, image_paths in groups:
                frames = MOTION_MAP.load_grayscale_frames(image_paths)
                motion_maps = MOTION_MAP.generate_sequence(frames)
                for image_path, motion_map in zip(image_paths, motion_maps):
                    target = motion_root / image_path.relative_to(images_root)
                    MOTION_MAP.save_motion_image(target, motion_map, save_rgb=save_rgb)
                    written += 1
                    progress.update(1)
    else:
        module = MOTION_MAP.OnnxMotionMapModule().eval()
        with torch.no_grad():
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
                        target = motion_root / image_path.relative_to(images_root)
                        MOTION_MAP.save_motion_image(target, motion_map.squeeze(0), save_rgb=save_rgb)
                        state_valid.fill_(1.0)
                        written += 1
                        progress.update(1)
    return written


def main():
    args = parse_args()
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    run_dir = increment_path(Path(args.project).expanduser() / args.name, exist_ok=args.exist_ok, mkdir=True).resolve()
    images_root = run_dir / "input" / "images"
    motion_root = run_dir / "input" / "image"

    if source.is_dir():
        frame_paths = prepare_images_from_directory(source, images_root, recursive=args.recursive)
    elif source.is_file() and is_video_file(source):
        frame_paths = prepare_images_from_video(source, images_root, frame_ext=args.frame_ext)
    else:
        raise ValueError(f"Unsupported source: {source}. Please provide a frame directory or a supported video file.")

    source_list = build_source_list_file(frame_paths, run_dir / "input" / "source.txt")
    written = generate_motion_maps(
        images_root=images_root,
        motion_root=motion_root,
        mode=args.motion_mode,
        recursive=True,
        save_rgb=args.save_rgb,
    )
    print(f"Saved {written} motion maps to: {motion_root}")

    model = RTDETR(args.weights)
    result_count = 0
    for _ in model.predict(
        source=str(source_list),
        ch=6,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=str(run_dir.parent),
        name=run_dir.name,
        exist_ok=True,
        save=True,
        stream=True,
        verbose=True,
    ):
        result_count += 1

    if result_count == 0:
        raise RuntimeError("Prediction completed without returning any results.")

    save_dir = Path(model.predictor.save_dir).resolve()
    print(f"Processed {result_count} frames")
    print(f"Saved outputs to: {save_dir}")


if __name__ == "__main__":
    main()
