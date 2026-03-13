import importlib.util
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from pred import export_prediction_json, serialize_detections
from ultralytics.data.augment import LetterBox
from ultralytics.engine.results import Results
from ultralytics.utils import ops


def load_motion_map_module():
    module_path = Path(__file__).resolve().parent / "ultralytics" / "data" / "motion_map.py"
    spec = importlib.util.spec_from_file_location("video_motion_map_impl", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MOTION_MAP = load_motion_map_module()


@dataclass
class MotionStreamState:
    # adapt_state 保存上一帧对比图，memory_state 保存运动记忆，state_valid 标记状态是否已初始化。
    adapt_state: torch.Tensor
    memory_state: torch.Tensor
    state_valid: torch.Tensor


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


def collect_stream_groups(source_root: Path, recursive: bool) -> list[tuple[Path, list[Path]]]:
    return MOTION_MAP.collect_image_groups(source_root, recursive=recursive)


def stream_layout_warnings(groups: list[tuple[Path, list[Path]]]) -> list[str]:
    return MOTION_MAP.sequence_layout_warnings(groups)


def create_motion_stream_module(mode: str = "paper_onnx"):
    if mode == "paper_onnx":
        return MOTION_MAP.OnnxMotionMapCoreModule().eval()
    if mode == "onnx_approx":
        return MOTION_MAP.OnnxApproxMotionMapModule().eval()
    raise ValueError(f"Unsupported motion stream mode: {mode}")


def load_stream_frame(path: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    frame = torch.from_numpy(gray).to(torch.float32).div(255.0).unsqueeze(0).unsqueeze(0)
    return image, gray, frame


def reset_motion_stream_state(frame: torch.Tensor) -> MotionStreamState:
    return MotionStreamState(
        adapt_state=torch.zeros_like(frame),
        memory_state=torch.zeros_like(frame),
        state_valid=torch.zeros((1, 1, 1, 1), dtype=frame.dtype),
    )


def _motion_tensor_to_bgr_image(motion: torch.Tensor) -> np.ndarray:
    motion_2d = motion.detach().cpu().squeeze().clamp_min(0.0).numpy()
    max_value = float(motion_2d.max(initial=0.0))
    if max_value > MOTION_MAP.MotionMapConfig().eps:
        motion_2d = motion_2d / max_value
    else:
        motion_2d = np.zeros_like(motion_2d, dtype=np.float32)
    image = np.clip(np.rint(motion_2d * 255.0), 0, 255).astype(np.uint8)
    return np.repeat(image[..., None], 3, axis=2)


def generate_stream_motion_frame(
    frame_path: Path,
    module,
    state: MotionStreamState,
    mode: str = "paper_onnx",
) -> tuple[np.ndarray, torch.Tensor, np.ndarray, MotionStreamState]:
    appearance_bgr, _, frame = load_stream_frame(frame_path)
    if state.adapt_state.shape != frame.shape:
        state = reset_motion_stream_state(frame)

    # 当前帧的 motion 输出依赖上一帧递推下来的状态。
    with torch.no_grad():
        motion, adapt_state, memory_state = module(
            frame,
            state.adapt_state,
            state.memory_state,
            state.state_valid,
        )
    if mode == "paper_onnx":
        motion = MOTION_MAP.paper_postprocess(motion)
    elif mode != "onnx_approx":
        raise ValueError(f"Unsupported motion stream mode: {mode}")
    next_state = MotionStreamState(
        adapt_state=adapt_state,
        memory_state=memory_state,
        state_valid=torch.ones((1, 1, 1, 1), dtype=frame.dtype),
    )
    return appearance_bgr, frame, _motion_tensor_to_bgr_image(motion), next_state


def preprocess_stream_frame(appearance_bgr: np.ndarray, motion_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    merged = np.concatenate((motion_bgr, appearance_bgr), axis=2)
    letterbox = LetterBox((imgsz, imgsz), auto=False, scaleFill=True)
    transformed = letterbox(image=merged)
    transformed = transformed[..., ::-1].transpose((2, 0, 1))
    return np.ascontiguousarray(transformed[None].astype(np.float32) / 255.0)


def preprocess_stream_appearance_frame(appearance_bgr: np.ndarray, imgsz: int) -> np.ndarray:
    letterbox = LetterBox((imgsz, imgsz), auto=False, scaleFill=True)
    transformed = letterbox(image=appearance_bgr)
    transformed = transformed[..., ::-1].transpose((2, 0, 1))
    return np.ascontiguousarray(transformed[None].astype(np.float32) / 255.0)


def frame_tensor_to_motion_grayscale(frame_rgb: torch.Tensor) -> torch.Tensor:
    if frame_rgb.ndim != 4 or frame_rgb.shape[1] != 3:
        raise ValueError(f"Expected frame tensor shape [N, 3, H, W], got {tuple(frame_rgb.shape)}")
    red = frame_rgb[:, 0:1]
    green = frame_rgb[:, 1:2]
    blue = frame_rgb[:, 2:3]
    return 0.299 * red + 0.587 * green + 0.114 * blue


def motion_tensor_to_rgb_tensor(motion: torch.Tensor) -> torch.Tensor:
    config = MOTION_MAP.MotionMapConfig()
    max_value = motion.amax(dim=(-2, -1), keepdim=True)
    normalized = motion / max_value.clamp_min(config.eps)
    normalized = torch.where(max_value > config.eps, normalized, torch.zeros_like(motion))
    return normalized.repeat(1, 3, 1, 1)


def build_stream_results(appearance_image: np.ndarray, frame_path: Path, names: dict[int, str], boxes: torch.Tensor) -> Results:
    return Results(appearance_image, path=str(frame_path), names=names, boxes=boxes)


def postprocess_stream_predictions(
    preds,
    appearance_image: np.ndarray,
    frame_path: Path,
    names: dict[int, str],
    conf: float,
) -> Results:
    prediction = preds[0] if isinstance(preds, (list, tuple)) else preds
    prediction = torch.as_tensor(prediction)
    bbox, score = prediction[0].split((4, prediction.shape[-1] - 4), dim=-1)
    bbox = ops.xywh2xyxy(bbox)
    max_score, cls = score.max(-1, keepdim=True)
    idx = max_score.squeeze(-1) > conf
    pred = torch.cat([bbox, max_score, cls], dim=-1)[idx]

    height, width = appearance_image.shape[:2]
    scaled = pred.clone()
    if len(scaled):
        # RT-DETR 输出是归一化框，这里映射回原始图像坐标。
        scaled[..., [0, 2]] *= width
        scaled[..., [1, 3]] *= height
    else:
        scaled = torch.zeros((0, 6), dtype=torch.float32)
    return build_stream_results(appearance_image, frame_path, names, scaled)


def save_stream_outputs(
    result: Results,
    output_root: Path,
    source_root: Path,
    frame_path: Path,
    frame_index: int,
) -> tuple[Path, Path]:
    relative_path = frame_path.relative_to(source_root)
    image_path = output_root / "images" / relative_path
    json_path = output_root / "json" / relative_path.with_suffix(".json")
    image_path.parent.mkdir(parents=True, exist_ok=True)

    # 保持与现有单图导出风格一致，便于后处理脚本复用。
    payload = {
        "source_image": str(frame_path),
        "frame_index": frame_index,
        "names": {str(key): value for key, value in result.names.items()},
        "detections": serialize_detections(result),
    }

    # 边推理边落盘，避免整段缓存。
    result.save(filename=str(image_path))
    export_prediction_json(json_path, payload)
    return image_path, json_path


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
    elif mode == "paper_onnx":
        with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
            for _, image_paths in groups:
                frames = MOTION_MAP.load_grayscale_frames(image_paths)
                motion_maps = MOTION_MAP.generate_paper_onnx_sequence(frames)
                for image_path, motion_map in zip(image_paths, motion_maps):
                    target = motion_root / image_path.relative_to(images_root)
                    MOTION_MAP.save_motion_image(target, motion_map, save_rgb=save_rgb)
                    written += 1
                    progress.update(1)
    elif mode == "onnx_approx":
        with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
            for _, image_paths in groups:
                frames = MOTION_MAP.load_grayscale_frames(image_paths)
                motion_maps = MOTION_MAP.generate_onnx_approx_sequence(frames)
                for image_path, motion_map in zip(image_paths, motion_maps):
                    target = motion_root / image_path.relative_to(images_root)
                    MOTION_MAP.save_motion_image(target, motion_map, save_rgb=save_rgb)
                    written += 1
                    progress.update(1)
    else:
        raise ValueError(f"Unsupported motion-map mode: {mode}")
    return written
