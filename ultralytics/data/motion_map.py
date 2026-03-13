"""Motion-map generation utilities for MI-DETR."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class MotionMapConfig:
    """Default RCA parameters aligned with the MI-DETR paper."""

    # 感光/适应阶段参数：控制输入亮度经过非线性拉伸时的阈值和增益。
    theta_p: float = 0.1
    g_p: float = 1.5
    # 双极细胞阶段参数：控制 ON/OFF 响应的阈值和放大倍数。
    theta_b: float = 0.2
    g_b: float = 2.0
    # 空间抑制阶段参数：高斯核大小与水平抑制强度。
    gaussian_sigma: float = 1.0
    sigma_h: float = 0.3
    # 运动响应阶段参数：空间增强后的阈值和最终增益。
    theta_m: float = 0.3
    g_m: float = 2.5
    # 时间记忆阶段参数：
    # alpha 越大，越依赖上一时刻记忆；beta 控制帧间差响应幅度。
    alpha: float = 0.8
    beta: float = 1.2
    # 融合阶段参数：
    # gamma_a / gamma_tau 控制 contrast 与 memory 的耦合，eta_m 决定 motion 和 memory 的融合占比。
    gamma_a: float = 0.5
    gamma_tau: float = 0.7
    eta_m: float = 0.7
    # 后处理参数：gamma_p 控制非线性增强，其余参数决定 reference / onnx 各自使用的平滑方式。
    gamma_p: float = 0.8
    gaussian_kernel_size: int = 3
    mexican_hat_size: int = 5
    mexican_hat_sigma_center: float = 1.0
    mexican_hat_sigma_surround: float = 2.0
    mexican_hat_surround_weight: float = 0.5
    smoothing_kernel_size: int = 5
    smoothing_sigma: float = 1.0
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 0.1
    bilateral_sigma_space: float = 0.1
    eps: float = 1e-6

    @property
    def horizontal_inhibition(self) -> float:
        # 给 sigma_h 一个语义更明确的访问名，便于在响应函数里直读其含义。
        return self.sigma_h


def kernel_size_from_sigma(sigma: float) -> int:
    """Return the odd kernel size covering +/- 3 sigma."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    return int(2 * math.ceil(3 * sigma) + 1)


def gaussian_kernel2d(
    sigma: float,
    *,
    size: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a normalized 2D Gaussian kernel."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    # 如果调用方没有显式指定大小，就用覆盖 +/- 3 sigma 的最小奇数窗口。
    size = kernel_size_from_sigma(sigma) if size is None else int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)


def mexican_hat_kernel2d(
    sigma_center: float,
    sigma_surround: float = 2.0,
    *,
    surround_weight: float = 0.5,
    size: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create the paper-style Mexican-hat kernel with zero-sum and unit-L1 normalization."""
    if sigma_center <= 0 or sigma_surround <= 0 or surround_weight <= 0:
        raise ValueError("sigma_center, sigma_surround and surround_weight must be positive")
    # 论文中的核在有限窗口上定义，因此这里显式使用离散采样，
    # 再强制零和并做 L1 归一化，避免截断带来的偏置。
    size = kernel_size_from_sigma(sigma_center) if size is None else int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    radius2 = xx.square() + yy.square()
    center = torch.exp(-radius2 / (2 * sigma_center * sigma_center)) / (2 * math.pi * sigma_center * sigma_center)
    surround = torch.exp(-radius2 / (2 * sigma_surround * sigma_surround)) / (2 * math.pi * sigma_surround * sigma_surround)
    kernel = center - surround_weight * surround
    kernel = kernel - kernel.mean()
    return kernel / kernel.abs().sum().clamp_min(torch.finfo(kernel.dtype).eps)


def collect_image_groups(source_root: str | Path, recursive: bool = False) -> list[tuple[Path, list[Path]]]:
    """Collect image paths grouped by their relative parent directory."""
    # 所有状态都会在“目录边界”重置，所以这里要先把同一目录下的帧收成一组。
    # 上层逻辑把每个目录视作一段独立视频序列。
    source_root = Path(source_root).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    pattern = "**/*" if recursive else "*"
    image_paths = sorted(
        path
        for path in source_root.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not image_paths:
        raise FileNotFoundError(f"No images found under: {source_root}")

    groups: dict[Path, list[Path]] = {}
    for path in image_paths:
        # 用相对父目录做 key，保证输出目录结构能与输入对齐。
        relative_parent = path.relative_to(source_root).parent
        groups.setdefault(relative_parent, []).append(path)
    return sorted(groups.items(), key=lambda item: item[0].as_posix())


def sequence_layout_warnings(groups: list[tuple[Path, list[Path]]]) -> list[str]:
    """Flag layouts where RCA state is reset only by split-like directories."""
    # 如果一个目录名只是 train/val/test/images 这种“数据集层级”，
    # 但里面又塞了很多帧，那么算法会把它误当成一条连续序列来递推状态。
    warnings = []
    split_like = {"train", "val", "valid", "test", "images", "image"}

    for relative_parent, image_paths in groups:
        if len(image_paths) < 2:
            continue
        parts = [part.lower() for part in relative_parent.parts if part not in {"", "."}]
        if not parts or parts[-1] in split_like:
            label = relative_parent.as_posix() if parts else "."
            warnings.append(
                f"Directory group '{label}' contains {len(image_paths)} frames; "
                "RCA resets state per directory only, so split mixed clips into per-sequence subdirectories."
            )

    return warnings


def load_grayscale_frames(paths: list[Path]) -> torch.Tensor:
    """Load images as a [T, 1, H, W] grayscale float tensor in [0, 1]."""
    # 输出张量：
    # T 表示时间维，单通道放在第 2 维，像素值统一缩放到 [0, 1]。
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if image.ndim == 2:
            # 原图本来就是单通道时直接使用。
            gray = image
        else:
            # 彩色图统一转成灰度，因为后续 RCA 实现只接收单通道输入。
            gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
        # 每张图从 [H, W] 扩成 [1, H, W]，再在最后 stack 成 [T, 1, H, W]。
        frames.append(torch.from_numpy(gray).to(torch.float32).div(255.0).unsqueeze(0))
    return torch.stack(frames, dim=0)


def save_motion_image(path: str | Path, motion: torch.Tensor, save_rgb: bool = True) -> Path:
    """Save a normalized motion map as uint8 PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # 保存前统一拉回 CPU，压到 2D，并且丢掉所有负值。
    motion_2d = motion.detach().cpu().squeeze().clamp_min(0.0).numpy()
    # 每张图按自身最大值归一化，确保不同序列都能充分用满 0~255 动态范围。
    motion_2d = _max_normalize_array(motion_2d, MotionMapConfig().eps)
    image = np.clip(np.rint(motion_2d * 255.0), 0, 255).astype(np.uint8)
    # save_rgb=True 时把灰度 motion map 复制成 3 通道，方便部分数据管线直接读取。
    image = np.repeat(image[..., None], 3, axis=2) if save_rgb else image
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write motion image: {path}")
    return path


def _conv2d_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    # 把 [kh, kw] 核整理成 conv2d 所需的 [out_c, in_c, kh, kw]，
    # 然后通过对称 padding 保持输出空间尺寸与输入一致。
    kernel_4d = kernel.to(dtype=x.dtype, device=x.device).view(1, 1, kernel.shape[0], kernel.shape[1])
    padding = (int(kernel_4d.shape[-2] // 2), int(kernel_4d.shape[-1] // 2))
    return F.conv2d(x, kernel_4d, padding=padding)


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
    # Sobel 只在首帧初始化时使用：没有前一帧可比较时，用边缘强度近似提供“显著性”响应。
    dtype, device = x.dtype, x.device
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=dtype,
        device=device,
    )
    grad_x = _conv2d_same(x, sobel_x)
    grad_y = _conv2d_same(x, sobel_y)
    return torch.sqrt(grad_x.square() + grad_y.square() + torch.finfo(dtype).eps)


def _max_normalize_tensor(x: torch.Tensor, eps: float) -> torch.Tensor:
    # 按每张图自己的最大值归一化；如果整张图几乎全 0，则直接返回 0，避免除零噪声。
    max_value = x.amax(dim=(-2, -1), keepdim=True)
    normalized = x / max_value.clamp_min(eps)
    return torch.where(max_value > eps, normalized, torch.zeros_like(x))


def _max_normalize_array(x: np.ndarray, eps: float) -> np.ndarray:
    max_value = float(x.max(initial=0.0))
    if max_value <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / max_value).astype(np.float32, copy=False)


def _adapt(frame: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
    # 适应层把输入亮度分成两段处理：
    # 高于阈值的区域做 tanh 拉伸，低亮度区域保留一个很弱的线性响应。
    high_response = config.g_p * torch.tanh(frame - config.theta_p)
    low_response = frame * 0.1
    return torch.where(frame > config.theta_p, high_response, low_response)


def _horizontal_cell_response(frame: torch.Tensor, gaussian_kernel: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
    # 水平细胞阶段先做亮度适应，再做局部空间抑制，
    # 用于突出局部变化而压制大范围平滑背景。
    adapted = _adapt(frame, config)
    suppressed = adapted - config.horizontal_inhibition * _conv2d_same(adapted, gaussian_kernel)
    return F.relu(suppressed)


def _bipolar_cell_response(horizontal: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
    # 双极细胞阶段分成 ON / OFF 两路：
    # ON 关注正向增强，OFF 关注负向增强，最后合并成统一的 contrast 图。
    on = F.relu(config.g_b * (horizontal - config.theta_b))
    off = F.relu(config.g_b * (-horizontal - config.theta_b))
    return on + off


def _temporal_response(
    contrast: torch.Tensor,
    prev_contrast: torch.Tensor,
    prev_memory: torch.Tensor,
    state_valid: torch.Tensor,
    config: MotionMapConfig,
    *,
    use_sobel_first: bool,
) -> torch.Tensor:
    # 时间响应的核心是“当前 contrast 与上一帧 contrast 的差”。
    difference = config.beta * torch.abs(contrast - prev_contrast)
    if use_sobel_first:
        # 第一帧没有历史 contrast 可比较时，state_valid=0，
        # 此时退化为 Sobel 边缘幅值，避免第一帧整张图都接近 0。
        sobel = config.beta * _sobel_magnitude(contrast)
        response = state_valid * difference + (1.0 - state_valid) * sobel
    else:
        response = difference
    # 与上一时刻 memory 做指数平滑，形成时间记忆。
    return config.alpha * prev_memory + (1.0 - config.alpha) * response


def _motion_response(
    contrast: torch.Tensor,
    memory: torch.Tensor,
    mexican_hat_kernel: torch.Tensor,
    config: MotionMapConfig,
) -> torch.Tensor:
    # 先把当前 contrast 和历史 memory 融合，再用 Mexican-hat 做中心-周围增强，
    # 最后经过阈值和增益，形成更聚焦的运动响应图。
    integrated = contrast + config.gamma_a * memory
    spatial = _conv2d_same(integrated, mexican_hat_kernel)
    thresholded = F.relu(torch.tanh(spatial + config.gamma_tau * memory - config.theta_m))
    return config.g_m * thresholded


def paper_postprocess(motion: torch.Tensor, config: MotionMapConfig | None = None) -> torch.Tensor:
    """Apply the paper-style host-side enhancement without changing per-frame dynamic range."""
    if motion.ndim != 4 or motion.shape[1] != 1:
        raise ValueError("motion must have shape [N, 1, H, W]")

    config = config or MotionMapConfig()
    original_device = motion.device
    outputs = []
    for sample in motion.detach().to(dtype=torch.float32, device="cpu"):
        powered = sample.squeeze(0).clamp_min(0.0).pow(config.gamma_p).numpy().astype(np.float32, copy=False)
        smoothed = cv2.bilateralFilter(
            powered,
            d=config.bilateral_diameter,
            sigmaColor=config.bilateral_sigma_color,
            sigmaSpace=config.bilateral_sigma_space,
        )
        outputs.append(torch.from_numpy(smoothed).unsqueeze(0))
    return torch.stack(outputs, dim=0).to(dtype=motion.dtype, device=original_device)


def onnx_approx_postprocess(
    motion: torch.Tensor,
    smoothing_kernel: torch.Tensor | None = None,
    config: MotionMapConfig | None = None,
) -> torch.Tensor:
    """Apply the graph-only approximate enhancement used for export-friendly motion maps."""
    if motion.ndim != 4 or motion.shape[1] != 1:
        raise ValueError("motion must have shape [N, 1, H, W]")

    config = config or MotionMapConfig()
    if smoothing_kernel is None:
        smoothing_kernel = gaussian_kernel2d(
            config.smoothing_sigma,
            size=config.smoothing_kernel_size,
            dtype=motion.dtype,
            device=motion.device,
        )
    # onnx 路径用“幂次增强 + 高斯平滑 + 最大值归一化”替代双边滤波，
    # 避免引入不易导出的 OpenCV 后处理算子。
    powered = motion.clamp_min(0.0).pow(config.gamma_p)
    smoothed = _conv2d_same(powered, smoothing_kernel)
    return _max_normalize_tensor(smoothed, config.eps)


def _rca_step(
    frame: torch.Tensor,
    prev_contrast: torch.Tensor,
    prev_memory: torch.Tensor,
    state_valid: torch.Tensor,
    gaussian_kernel: torch.Tensor,
    mexican_hat_kernel: torch.Tensor,
    config: MotionMapConfig,
    *,
    use_sobel_first: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # 单步 RCA 链路：
    # frame -> horizontal -> contrast -> memory -> motion -> fused
    # 返回值里：
    # fused 是当前要输出/后处理的结果，
    # contrast 和 memory 会作为下一帧递推状态继续传下去。
    horizontal = _horizontal_cell_response(frame, gaussian_kernel, config)
    contrast = _bipolar_cell_response(horizontal, config)
    memory = _temporal_response(
        contrast,
        prev_contrast,
        prev_memory,
        state_valid,
        config,
        use_sobel_first=use_sobel_first,
    )
    motion = _motion_response(contrast, memory, mexican_hat_kernel, config)
    fused = config.eta_m * motion + (1.0 - config.eta_m) * memory
    return fused, contrast, memory


def generate_sequence(
    frames: torch.Tensor,
    *,
    use_sobel_first: bool = True,
    config: MotionMapConfig | None = None,
) -> torch.Tensor:
    """Generate the paper/reference motion-map sequence from grayscale frames."""
    # 这里接收完整序列，统一在 CPU 上执行 reference 版本，避免双边滤波和小张量循环带来的设备切换复杂度。
    if frames.ndim != 4 or frames.shape[1] != 1:
        raise ValueError("frames must have shape [T, 1, H, W]")

    config = config or MotionMapConfig()
    original_device = frames.device
    frames_cpu = frames.detach().to(dtype=torch.float32, device="cpu")
    gaussian_kernel = gaussian_kernel2d(config.gaussian_sigma, size=config.gaussian_kernel_size)
    mexican_hat_kernel = mexican_hat_kernel2d(
        config.mexican_hat_sigma_center,
        config.mexican_hat_sigma_surround,
        size=config.mexican_hat_size,
    )

    prev_contrast = torch.zeros_like(frames_cpu[:1])
    prev_memory = torch.zeros_like(frames_cpu[:1])
    outputs = []

    for index in range(frames_cpu.shape[0]):
        frame = frames_cpu[index : index + 1]
        # 只有第一帧 state_valid=0，表示没有历史状态可用；
        # 从第二帧开始，prev_contrast 和 prev_memory 都来自上一轮输出。
        state_valid = torch.full((1, 1, 1, 1), float(index > 0), dtype=frame.dtype)
        fused, prev_contrast, prev_memory = _rca_step(
            frame,
            prev_contrast,
            prev_memory,
            state_valid,
            gaussian_kernel,
            mexican_hat_kernel,
            config,
            use_sobel_first=use_sobel_first,
        )
        # reference 路径逐帧做论文里的 host-side enhance，再收集成完整序列。
        outputs.append(paper_postprocess(fused, config).squeeze(0))

    return torch.stack(outputs, dim=0).to(original_device)


def generate_paper_onnx_sequence(
    frames: torch.Tensor,
    *,
    config: MotionMapConfig | None = None,
) -> torch.Tensor:
    """Generate motion maps with ONNX-friendly recurrent core and paper host-side postprocess."""
    if frames.ndim != 4 or frames.shape[1] != 1:
        raise ValueError("frames must have shape [T, 1, H, W]")

    config = config or MotionMapConfig()
    original_device = frames.device
    frames_cpu = frames.detach().to(dtype=torch.float32, device="cpu")
    module = OnnxMotionMapCoreModule(config).eval()
    adapt_state = torch.zeros_like(frames_cpu[:1])
    memory_state = torch.zeros_like(frames_cpu[:1])
    state_valid = torch.zeros((1, 1, 1, 1), dtype=frames_cpu.dtype)
    outputs = []

    with torch.no_grad():
        for frame in frames_cpu:
            fused, adapt_state, memory_state = module(frame.unsqueeze(0), adapt_state, memory_state, state_valid)
            outputs.append(paper_postprocess(fused, config).squeeze(0))
            state_valid.fill_(1.0)

    return torch.stack(outputs, dim=0).to(original_device)


def generate_onnx_approx_sequence(
    frames: torch.Tensor,
    *,
    config: MotionMapConfig | None = None,
) -> torch.Tensor:
    """Generate graph-only approximate motion maps frame-by-frame."""
    if frames.ndim != 4 or frames.shape[1] != 1:
        raise ValueError("frames must have shape [T, 1, H, W]")

    config = config or MotionMapConfig()
    original_device = frames.device
    frames_cpu = frames.detach().to(dtype=torch.float32, device="cpu")
    module = OnnxApproxMotionMapModule(config).eval()
    adapt_state = torch.zeros_like(frames_cpu[:1])
    memory_state = torch.zeros_like(frames_cpu[:1])
    state_valid = torch.zeros((1, 1, 1, 1), dtype=frames_cpu.dtype)
    outputs = []

    with torch.no_grad():
        for frame in frames_cpu:
            motion, adapt_state, memory_state = module(frame.unsqueeze(0), adapt_state, memory_state, state_valid)
            outputs.append(motion.squeeze(0))
            state_valid.fill_(1.0)

    return torch.stack(outputs, dim=0).to(original_device)


class OnnxMotionMapCoreModule(nn.Module):
    """ONNX-friendly single-step RCA core without paper host-side postprocess."""

    def __init__(self, config: MotionMapConfig | None = None):
        super().__init__()
        self.config = config or MotionMapConfig()
        # 这些核注册成 buffer，而不是每次 forward 临时创建，
        # 这样既能随模块一起导出，也能避免重复分配。
        self.register_buffer(
            "gaussian_kernel",
            gaussian_kernel2d(self.config.gaussian_sigma, size=self.config.gaussian_kernel_size),
        )
        self.register_buffer(
            "mexican_hat_kernel",
            mexican_hat_kernel2d(
                self.config.mexican_hat_sigma_center,
                self.config.mexican_hat_sigma_surround,
                surround_weight=self.config.mexican_hat_surround_weight,
                size=self.config.mexican_hat_size,
            ),
        )

    def forward(
        self,
        frame: torch.Tensor,
        adapt_state: torch.Tensor,
        memory_state: torch.Tensor,
        state_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one motion map and the next recurrent states.

        The API keeps the planned `adapt_state` name for compatibility, but the tensor
        stores the previous contrast map required by the temporal difference stage.
        """
        # state_valid 决定历史状态是否参与本轮计算：
        # 第一帧传 0，后续帧传 1。
        valid = state_valid.to(dtype=frame.dtype)
        # adapt_state 是兼容旧接口留下的名字，真正存放的是上一帧 contrast。
        prev_contrast = adapt_state * valid
        prev_memory = memory_state * valid
        fused, next_contrast, next_memory = _rca_step(
            frame,
            prev_contrast,
            prev_memory,
            valid,
            self.gaussian_kernel,
            self.mexican_hat_kernel,
            self.config,
            use_sobel_first=True,
        )
        return fused, next_contrast, next_memory


class OnnxApproxMotionMapModule(nn.Module):
    """Graph-only approximation of the paper motion map suitable for full ONNX export."""

    def __init__(self, config: MotionMapConfig | None = None):
        super().__init__()
        self.config = config or MotionMapConfig()
        self.core = OnnxMotionMapCoreModule(self.config)
        self.register_buffer(
            "smoothing_kernel",
            gaussian_kernel2d(self.config.smoothing_sigma, size=self.config.smoothing_kernel_size),
        )

    def forward(
        self,
        frame: torch.Tensor,
        adapt_state: torch.Tensor,
        memory_state: torch.Tensor,
        state_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused, next_contrast, next_memory = self.core(frame, adapt_state, memory_state, state_valid)
        motion = onnx_approx_postprocess(fused, self.smoothing_kernel, self.config)
        return motion, next_contrast, next_memory


OnnxMotionMapModule = OnnxApproxMotionMapModule


__all__ = (
    "IMAGE_SUFFIXES",
    "MotionMapConfig",
    "OnnxApproxMotionMapModule",
    "OnnxMotionMapCoreModule",
    "OnnxMotionMapModule",
    "collect_image_groups",
    "gaussian_kernel2d",
    "generate_onnx_approx_sequence",
    "generate_paper_onnx_sequence",
    "generate_sequence",
    "kernel_size_from_sigma",
    "load_grayscale_frames",
    "mexican_hat_kernel2d",
    "onnx_approx_postprocess",
    "paper_postprocess",
    "save_motion_image",
    "sequence_layout_warnings",
)
