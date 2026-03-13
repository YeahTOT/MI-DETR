import argparse
import warnings
from pathlib import Path

from ultralytics import RTDETR
from ultralytics.utils.files import increment_path
from video_pipeline import build_source_list_file, generate_motion_maps, prepare_images_from_directory

warnings.filterwarnings("ignore")

DEFAULT_WEIGHTS = "checkpoints/DAUB-R.onnx"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run MI-DETR ONNX inference for frame directories.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="ONNX checkpoint path.")
    parser.add_argument("--source", required=True, help="Frame directory path.")
    parser.add_argument(
        "--motion-mode",
        choices=("reference", "paper_onnx", "onnx_approx"),
        default="reference",
        help="Motion-map backend.",
    )
    parser.add_argument("--recursive", action="store_true", help="Recursively scan frame directories.")
    parser.add_argument("--device", default="", help="Prediction device, e.g. '0'.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--project", default="runs/video_onnx", help="Directory to save prediction runs.")
    parser.add_argument("--name", default="mi-detr", help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the run directory if it already exists.")
    parser.add_argument("--save-rgb", dest="save_rgb", action="store_true", help="Save motion maps as 3-channel PNGs.")
    parser.add_argument(
        "--no-save-rgb",
        dest="save_rgb",
        action="store_false",
        help="Save motion maps as single-channel PNGs.",
    )
    parser.set_defaults(save_rgb=True)
    return parser.parse_args(argv)


def main(argv=None) -> Path:
    args = parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not source.is_dir():
        raise ValueError(f"仅支持帧目录输入: {source}")

    run_dir = increment_path(Path(args.project).expanduser() / args.name, exist_ok=args.exist_ok, mkdir=True).resolve()
    images_root = run_dir / "input" / "images"
    motion_root = run_dir / "input" / "image"

    frame_paths = prepare_images_from_directory(source, images_root, recursive=args.recursive)
    source_list = build_source_list_file(frame_paths, run_dir / "input" / "source.txt")
    written = generate_motion_maps(
        images_root=images_root,
        motion_root=motion_root,
        mode=args.motion_mode,
        recursive=True,
        save_rgb=args.save_rgb,
    )
    print(f"Saved {written} motion maps to: {motion_root}")

    model = RTDETR(str(Path(args.weights).expanduser().resolve()))
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
    return save_dir


if __name__ == "__main__":
    main()
