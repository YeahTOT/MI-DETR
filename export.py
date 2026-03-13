import argparse
import shutil
from copy import deepcopy
from pathlib import Path

import onnx
import torch
from torch import nn
from ultralytics import RTDETR
from ultralytics.nn.modules.head import RTDETRDecoder
from video_pipeline import create_motion_stream_module, frame_tensor_to_motion_grayscale, motion_tensor_to_rgb_tensor

DEFAULT_WEIGHTS = "checkpoints/DAUB-R.pt"
MAX_COMPATIBLE_IR_VERSION = 11
STREAM_SUFFIX = "-stream.onnx"


class MotionIntegratedRTDETRWrapper(nn.Module):
    def __init__(self, detector_model: nn.Module, motion_module: nn.Module):
        super().__init__()
        self.detector_model = detector_model
        self.motion_module = motion_module

    def forward(
        self,
        frame: torch.Tensor,
        adapt_state: torch.Tensor,
        memory_state: torch.Tensor,
        state_valid: torch.Tensor,
    ):
        # 当前帧的 motion 依赖上一帧状态，next_* 需要由部署端持续回灌。
        valid = state_valid.to(dtype=frame.dtype)
        motion_input = frame_tensor_to_motion_grayscale(frame)
        motion, next_adapt_state, next_memory_state = self.motion_module(
            motion_input,
            adapt_state * valid,
            memory_state * valid,
            valid,
        )
        # detector 仍然吃 6 通道，只是 6 通道在图内由 appearance 和 motion 拼接出来。
        detector_input = torch.cat((frame, motion_tensor_to_rgb_tensor(motion)), dim=1)
        output0 = self.detector_model(detector_input)
        if isinstance(output0, (list, tuple)):
            output0 = output0[0]
        return output0, next_adapt_state, next_memory_state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Export an MI-DETR checkpoint to ONNX.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Checkpoint path.")
    parser.add_argument("--output", default=None, help="Output ONNX path. Defaults to weights with an .onnx suffix.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--device", default="", help="Export device, e.g. '0'.")
    parser.add_argument("--opset", type=int, default=13, help="ONNX opset version.")
    parser.add_argument("--simplify", dest="simplify", action="store_true", help="Simplify the ONNX graph.")
    parser.add_argument("--no-simplify", dest="simplify", action="store_false", help="Do not simplify the ONNX graph.")
    parser.add_argument("--dynamic", action="store_true", help="Export with dynamic input dimensions.")
    parser.add_argument("--batch", type=int, default=1, help="Export batch size.")
    parser.add_argument("--half", action="store_true", help="Export in half precision when supported.")
    parser.add_argument(
        "--with-motion-stream",
        action="store_true",
        help="Export a streaming ONNX with motion generation inside the graph.",
    )
    parser.set_defaults(simplify=True)
    return parser.parse_args(argv)


def resolve_output_path(weights: Path, output: str | None, with_motion_stream: bool = False) -> Path:
    if output:
        return Path(output)
    if with_motion_stream:
        return weights.with_name(f"{weights.stem}{STREAM_SUFFIX}")
    return weights.with_suffix(".onnx")


def export_onnx(model: RTDETR, *, imgsz: int, device: str, opset: int, simplify: bool, dynamic: bool, batch: int, half: bool):
    return model.export(
        format="onnx",
        imgsz=imgsz,
        device=device,
        opset=opset,
        simplify=simplify,
        dynamic=dynamic,
        batch=batch,
        half=half,
    )


def normalize_onnx_ir_version(path: Path, max_ir_version: int = MAX_COMPATIBLE_IR_VERSION) -> Path:
    model = onnx.load(path)
    if model.ir_version > max_ir_version:
        model.ir_version = max_ir_version
        onnx.save(model, path)
    return path


def _add_onnx_metadata(path: Path, metadata: dict[str, str]) -> Path:
    model = onnx.load(path)
    existing = {prop.key: prop.value for prop in model.metadata_props}
    for prop in list(model.metadata_props):
        model.metadata_props.remove(prop)
    existing.update(metadata)
    for key, value in existing.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = str(value)
    onnx.save(model, path)
    return path


def _resolve_torch_device(device: str) -> torch.device:
    if device and device.lower() != "cpu" and torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def _prepare_detector_model(weights: Path, device: torch.device, dynamic: bool) -> tuple[RTDETR, nn.Module]:
    model = RTDETR(str(weights))
    detector_model = deepcopy(model.model).to(device)
    for parameter in detector_model.parameters():
        parameter.requires_grad = False
    detector_model.eval()
    detector_model.float()
    detector_model = detector_model.fuse()
    for module in detector_model.modules():
        if isinstance(module, RTDETRDecoder):
            module.dynamic = dynamic
            module.export = True
            module.format = "onnx"
    return model, detector_model


def should_retry_grid_sampler(exc: RuntimeError, opset: int) -> bool:
    message = str(exc)
    return opset == 13 and "grid_sampler" in message and "version 16" in message


def export_motion_stream_onnx(
    weights: Path,
    output: Path,
    imgsz: int,
    device: str,
    opset: int,
    dynamic: bool,
    batch: int,
    half: bool,
) -> Path:
    torch_device = _resolve_torch_device(device)
    if dynamic:
        torch_device = torch.device("cpu")

    model, detector_model = _prepare_detector_model(weights, torch_device, dynamic)
    motion_module = create_motion_stream_module(mode="onnx_approx").to(torch_device).eval()
    wrapper = MotionIntegratedRTDETRWrapper(detector_model, motion_module).to(torch_device).eval()

    frame = torch.zeros((batch, 3, imgsz, imgsz), device=torch_device)
    adapt_state = torch.zeros((batch, 1, imgsz, imgsz), device=torch_device)
    memory_state = torch.zeros((batch, 1, imgsz, imgsz), device=torch_device)
    state_valid = torch.zeros((batch, 1, 1, 1), device=torch_device)

    if half and torch_device.type != "cpu":
        wrapper = wrapper.half()
        frame = frame.half()
        adapt_state = adapt_state.half()
        memory_state = memory_state.half()
        state_valid = state_valid.half()

    dynamic_axes = None
    if dynamic:
        dynamic_axes = {
            "frame": {0: "batch", 2: "height", 3: "width"},
            "adapt_state": {0: "batch", 2: "height", 3: "width"},
            "memory_state": {0: "batch", 2: "height", 3: "width"},
            "state_valid": {0: "batch"},
            "output0": {0: "batch"},
            "next_adapt_state": {0: "batch", 2: "height", 3: "width"},
            "next_memory_state": {0: "batch", 2: "height", 3: "width"},
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper.cpu() if dynamic else wrapper,
        (
            frame.cpu() if dynamic else frame,
            adapt_state.cpu() if dynamic else adapt_state,
            memory_state.cpu() if dynamic else memory_state,
            state_valid.cpu() if dynamic else state_valid,
        ),
        str(output),
        verbose=False,
        opset_version=opset,
        do_constant_folding=True,
        input_names=["frame", "adapt_state", "memory_state", "state_valid"],
        output_names=["output0", "next_adapt_state", "next_memory_state"],
        dynamic_axes=dynamic_axes,
    )
    normalize_onnx_ir_version(output)
    metadata = {
        "description": "Ultralytics RT-DETR stream-integrated model trained on DAUB-R.yaml",
        "author": "Ultralytics",
        "task": "detect",
        "batch": batch,
        "imgsz": [imgsz, imgsz],
        "names": model.model.names,
        "motion_stream_integrated": True,
    }
    _add_onnx_metadata(output, metadata)
    return output


def main(argv=None) -> Path:
    args = parse_args(argv)
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    output = resolve_output_path(weights, args.output, with_motion_stream=args.with_motion_stream).expanduser().resolve()
    if args.with_motion_stream:
        try:
            exported = export_motion_stream_onnx(
                weights=weights,
                output=output,
                imgsz=args.imgsz,
                device=args.device,
                opset=args.opset,
                dynamic=args.dynamic,
                batch=args.batch,
                half=args.half,
            )
        except RuntimeError as exc:
            if should_retry_grid_sampler(exc, args.opset):
                print("Retrying ONNX export with opset 16 because grid_sampler is unsupported in opset 13.")
                exported = export_motion_stream_onnx(
                    weights=weights,
                    output=output,
                    imgsz=args.imgsz,
                    device=args.device,
                    opset=16,
                    dynamic=args.dynamic,
                    batch=args.batch,
                    half=args.half,
                )
            else:
                raise
        print(f"Exported ONNX to: {exported}")
        return exported

    model = RTDETR(str(weights))
    try:
        exported_path = export_onnx(
            model,
            imgsz=args.imgsz,
            device=args.device,
            opset=args.opset,
            simplify=args.simplify,
            dynamic=args.dynamic,
            batch=args.batch,
            half=args.half,
        )
    except RuntimeError as exc:
        if should_retry_grid_sampler(exc, args.opset):
            print("Retrying ONNX export with opset 16 because grid_sampler is unsupported in opset 13.")
            exported_path = export_onnx(
                model,
                imgsz=args.imgsz,
                device=args.device,
                opset=16,
                simplify=args.simplify,
                dynamic=args.dynamic,
                batch=args.batch,
                half=args.half,
            )
        else:
            raise

    exported = Path(exported_path).resolve()
    normalize_onnx_ir_version(exported)

    if exported != output:
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(exported), str(output))
    print(f"Exported ONNX to: {output}")
    return output


if __name__ == "__main__":
    main()
