import argparse
import shutil
from pathlib import Path

import onnx
from ultralytics import RTDETR

DEFAULT_WEIGHTS = "checkpoints/DAUB-R.pt"
MAX_COMPATIBLE_IR_VERSION = 11


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
    parser.set_defaults(simplify=True)
    return parser.parse_args(argv)


def resolve_output_path(weights: Path, output: str | None) -> Path:
    return Path(output) if output else weights.with_suffix(".onnx")


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


def main(argv=None) -> Path:
    args = parse_args(argv)
    weights = Path(args.weights).expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"Weights not found: {weights}")

    output = resolve_output_path(weights, args.output).expanduser().resolve()
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
        message = str(exc)
        if args.opset == 13 and "grid_sampler" in message and "version 16" in message:
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
