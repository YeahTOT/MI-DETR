import argparse
import ast
import warnings
from pathlib import Path

import onnx
import onnxruntime

from video_pipeline import (
    collect_stream_groups,
    create_motion_stream_module,
    generate_stream_motion_frame,
    load_stream_frame,
    postprocess_stream_predictions,
    preprocess_stream_frame,
    reset_motion_stream_state,
    save_stream_outputs,
    stream_layout_warnings,
)

warnings.filterwarnings("ignore")

DEFAULT_WEIGHTS = "checkpoints/DAUB-R.onnx"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run streaming MI-DETR ONNX inference for frame directories.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="ONNX checkpoint path.")
    parser.add_argument("--source", required=True, help="Frame directory path.")
    parser.add_argument("--recursive", action="store_true", help="Recursively scan frame directories.")
    parser.add_argument("--device", default="cpu", help="ONNX Runtime device, e.g. 'cpu' or '0'.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--project", default="runs/video_onnx_stream", help="Directory to save prediction runs.")
    parser.add_argument("--name", default="mi-detr", help="Run name.")
    parser.add_argument("--exist-ok", action="store_true", help="Reuse the run directory if it already exists.")
    return parser.parse_args(argv)


def load_onnx_class_names(weights: Path) -> dict[int, str]:
    model = onnx.load(weights)
    metadata = {prop.key: prop.value for prop in model.metadata_props}
    raw_names = metadata.get("names")
    if not raw_names:
        return {0: "class0"}
    parsed = ast.literal_eval(raw_names)
    return {int(key): str(value) for key, value in parsed.items()}


def build_onnx_session(weights: Path, device: str):
    providers = ["CPUExecutionProvider"]
    if device and device.lower() != "cpu":
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = onnxruntime.InferenceSession(str(weights), providers=providers)
    input_name = session.get_inputs()[0].name
    output_names = [output.name for output in session.get_outputs()]
    return session, input_name, output_names


def main(argv=None) -> Path:
    args = parse_args(argv)
    source = Path(args.source).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")
    if not source.is_dir():
        raise ValueError(f"仅支持帧目录输入: {source}")

    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    output_root = (Path(args.project).expanduser() / args.name).resolve()
    if output_root.exists() and not args.exist_ok:
        raise FileExistsError(f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    groups = collect_stream_groups(source, recursive=args.recursive)
    for warning in stream_layout_warnings(groups):
        print(f"WARNING: {warning}")

    session, input_name, output_names = build_onnx_session(weights, args.device)
    names = load_onnx_class_names(weights)
    motion_module = create_motion_stream_module()

    frame_index = 0
    # 按真实流式场景逐帧生成 motion 并立即推理。
    for _, image_paths in groups:
        state = None
        for frame_path in image_paths:
            if state is None:
                # 不同子目录视为不同视频片段，切换目录时重置流式状态。
                _, _, frame_tensor = load_stream_frame(frame_path)
                state = reset_motion_stream_state(frame_tensor)

            appearance_image, _, motion_image, state = generate_stream_motion_frame(frame_path, motion_module, state)
            frame_index += 1
            input_tensor = preprocess_stream_frame(appearance_image, motion_image, args.imgsz)
            preds = session.run(output_names, {input_name: input_tensor})
            result = postprocess_stream_predictions(preds, appearance_image, frame_path, names, args.conf)

            # 边推理边落盘，避免整段缓存。
            save_stream_outputs(result, output_root, source, frame_path, frame_index)

    if frame_index == 0:
        raise RuntimeError("Prediction completed without processing any frames.")

    print(f"Processed {frame_index} frames")
    print(f"Saved outputs to: {output_root}")
    return output_root


if __name__ == "__main__":
    main()
