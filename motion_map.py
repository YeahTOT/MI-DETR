"""Offline motion-map generation CLI for MI-DETR."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

from tqdm import tqdm


def _load_motion_map_module():
    # CLI 文件放在仓库根目录，而实际实现位于 ultralytics/data/motion_map.py。
    # 这里显式按路径加载模块，可以避免依赖包安装方式或 PYTHONPATH 配置。
    # 这样脚本既能直接 `python motion_map.py` 运行，也不会和其他同名模块冲突。
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
    # source-root 指向外观图像目录，脚本会从这里读取原始帧。
    parser.add_argument("--source-root", required=True, help="Appearance-image root, usually an images/ directory.")
    # output-root 指向 motion map 输出目录，输出路径会尽量保持与 source-root 相同的相对结构。
    parser.add_argument("--output-root", required=True, help="Output root, usually an image/ directory.")
    # reference 是论文参考实现；
    # paper_onnx 使用 ONNX-friendly 递推 core，再在宿主侧执行论文里的 bilateral enhance；
    # onnx_approx 则保留纯图内近似版，便于整图导出。
    parser.add_argument(
        "--mode",
        choices=("reference", "paper_onnx", "onnx_approx"),
        default="reference",
        help="Generation backend.",
    )
    # recursive 打开后，会递归遍历 source-root 下的所有子目录。
    parser.add_argument("--recursive", action="store_true", help="Traverse source-root recursively.")
    # 默认保存为 3 通道 PNG，方便与常见视觉工具和数据管线兼容。
    parser.add_argument("--save-rgb", dest="save_rgb", action="store_true", help="Save motion camaps as 3-channel PNGs.")
    parser.add_argument(
        "--no-save-rgb",
        dest="save_rgb",
        action="store_false",
        help="Save motion maps as single-channel PNGs.",
    )
    # 默认值显式放在这里，便于和 --no-save-rgb 对读。
    parser.set_defaults(save_rgb=True)
    return parser.parse_args()


def run_reference(groups: list[tuple[Path, list[Path]]], source_root: Path, output_root: Path, save_rgb: bool) -> int:
    # reference 路径按“目录分组”整段处理：
    # 1. 整组读入灰度帧
    # 2. 一次性生成整段 motion map
    # 3. 再逐张写回磁盘
    written = 0
    total = sum(len(image_paths) for _, image_paths in groups)
    with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
        for _, image_paths in groups:
            # frames 形状为 [T, 1, H, W]，T 是当前目录下的帧数。
            frames = MOTION_MAP.load_grayscale_frames(image_paths)
            # generate_sequence 会在内部维护前一帧状态，输出同长度的 motion map 序列。
            motion_maps = MOTION_MAP.generate_sequence(frames)
            for image_path, motion_map in zip(image_paths, motion_maps):
                # 输出目录保持和输入目录一致的相对路径，方便直接替换数据根目录使用。
                target = output_root / image_path.relative_to(source_root)
                MOTION_MAP.save_motion_image(target, motion_map, save_rgb=save_rgb)
                written += 1
                progress.update(1)
    return written


def _write_generated_sequence(
    groups: list[tuple[Path, list[Path]]],
    source_root: Path,
    output_root: Path,
    save_rgb: bool,
    generator,
) -> int:
    written = 0
    total = sum(len(image_paths) for _, image_paths in groups)
    with tqdm(total=total, desc="Generating motion maps", unit="image") as progress:
        for _, image_paths in groups:
            frames = MOTION_MAP.load_grayscale_frames(image_paths)
            motion_maps = generator(frames)
            for image_path, motion_map in zip(image_paths, motion_maps):
                target = output_root / image_path.relative_to(source_root)
                MOTION_MAP.save_motion_image(target, motion_map, save_rgb=save_rgb)
                written += 1
                progress.update(1)
    return written


def run_paper_onnx(groups: list[tuple[Path, list[Path]]], source_root: Path, output_root: Path, save_rgb: bool) -> int:
    return _write_generated_sequence(
        groups,
        source_root,
        output_root,
        save_rgb,
        MOTION_MAP.generate_paper_onnx_sequence,
    )


def run_onnx_approx(groups: list[tuple[Path, list[Path]]], source_root: Path, output_root: Path, save_rgb: bool) -> int:
    return _write_generated_sequence(
        groups,
        source_root,
        output_root,
        save_rgb,
        MOTION_MAP.generate_onnx_approx_sequence,
    )


def main() -> None:
    # 总调度顺序：
    # 1. 解析命令行
    # 2. 规范化输入输出路径
    # 3. 收集并按目录分组图片
    # 4. 检查目录布局是否会导致状态错误复用
    # 5. 根据 mode 选择 reference / paper_onnx / onnx_approx 后端
    # 6. 输出汇总信息
    args = parse_args()
    source_root = Path(args.source_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    groups = MOTION_MAP.collect_image_groups(source_root, recursive=args.recursive)
    for warning in MOTION_MAP.sequence_layout_warnings(groups):
        print(f"WARNING: {warning}")

    if args.mode == "reference":
        written = run_reference(groups, source_root, output_root, args.save_rgb)
    elif args.mode == "paper_onnx":
        written = run_paper_onnx(groups, source_root, output_root, args.save_rgb)
    else:
        written = run_onnx_approx(groups, source_root, output_root, args.save_rgb)

    print(f"Saved {written} motion maps to {output_root}")


if __name__ == "__main__":
    main()
