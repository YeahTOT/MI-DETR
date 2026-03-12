import argparse
import json
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

DEFAULT_WEIGHTS = "checkpoints/DAUB-R.pt"


def paired_modality_path(path: str) -> str:
    """Map an image path under `images/` to its paired modality path under `image/`."""
    token = f"{os.sep}images{os.sep}"
    if token not in path:
        raise FileNotFoundError(f"Expected paired-modality image under an 'images' directory, but got: {path}")
    return path.replace(token, f"{os.sep}image{os.sep}", 1)


def resolve_source_paths(source_path: str) -> tuple[Path, Path]:
    """Resolve the appearance image and its paired motion image."""
    source = Path(source_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source image not found: {source}")

    try:
        motion = Path(paired_modality_path(str(source))).resolve()
    except FileNotFoundError as exc:
        raise ValueError(
            f"Source image must be located under an 'images/' directory for automatic pairing: {source}"
        ) from exc

    if not motion.is_file():
        raise FileNotFoundError(f"Paired motion image not found: {motion}")

    return source, motion


def serialize_detections(result) -> list[dict]:
    """Convert Ultralytics Results boxes to JSON-friendly dictionaries."""
    detections = []
    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return detections

    for box in boxes:
        class_id = int(box.cls.item())
        detections.append(
            {
                "class_id": class_id,
                "class_name": result.names[class_id],
                "confidence": round(float(box.conf.item()), 6),
                "xyxy": [round(float(value), 3) for value in box.xyxy.view(-1).tolist()],
            }
        )
    return detections


def export_prediction_json(output_path: Path, payload: dict) -> Path:
    """Write structured prediction results to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run single-image MI-DETR prediction.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Checkpoint path.")
    parser.add_argument("--source", required=True, help="Appearance image path under an images/ directory.")
    parser.add_argument("--device", default="", help="Prediction device, e.g. '0'.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--project", default="runs/pred", help="Directory to save prediction runs.")
    parser.add_argument("--name", default="mi-detr", help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the run directory if it already exists.")
    return parser.parse_args()


def main():
    args = parse_args()
    source_path, motion_path = resolve_source_paths(args.source)
    from ultralytics import RTDETR

    model = RTDETR(args.weights)
    results = model.predict(
        source=str(source_path),
        ch=6,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        save=True,
        verbose=True,
    )

    if not results:
        raise RuntimeError("Prediction completed without returning any results.")

    appearance_result = results[0]
    motion_results = getattr(model.predictor, "ir_results", None) or []
    motion_result = motion_results[0] if motion_results else None
    save_dir = Path(model.predictor.save_dir)

    appearance_detections = serialize_detections(appearance_result)
    motion_detections = serialize_detections(motion_result) if motion_result is not None else []
    names = {str(key): value for key, value in appearance_result.names.items()}

    output_path = export_prediction_json(
        save_dir / f"{source_path.stem}.json",
        {
            "source_image": str(source_path),
            "motion_image": str(motion_path),
            "names": names,
            "appearance_detections": appearance_detections,
            "motion_detections": motion_detections,
        },
    )

    appearance_summary = appearance_result.verbose().strip().rstrip(",") or "(no detections)"
    print(f"Appearance: {appearance_summary}")
    if motion_result is not None:
        motion_summary = motion_result.verbose().strip().rstrip(",") or "(no detections)"
        print(f"Motion: {motion_summary}")
    print(f"Saved outputs to: {save_dir}")
    print(f"Saved JSON to: {output_path}")


if __name__ == "__main__":
    main()
