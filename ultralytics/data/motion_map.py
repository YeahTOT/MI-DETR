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
    """Default RCA parameters from the MI-DETR paper."""

    theta_p: float = 0.1
    g_p: float = 1.5
    theta_b: float = 0.2
    g_b: float = 2.0
    gaussian_sigma: float = 1.0
    sigma_h: float = 0.3
    theta_m: float = 0.4
    g_m: float = 5.0
    alpha: float = 0.8
    beta: float = 1.2
    gamma_a: float = 0.7
    gamma_tau: float = 1.0
    eta_m: float = 0.2
    gamma_p: float = 0.8
    gaussian_kernel_size: int = 3
    mexican_hat_size: int = 5
    mexican_hat_sigma_center: float = 1.0
    mexican_hat_sigma_surround: float = 1.5
    smoothing_kernel_size: int = 5
    smoothing_sigma: float = 1.0
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 0.1
    bilateral_sigma_space: float = 0.1
    eps: float = 1e-6

    @property
    def horizontal_inhibition(self) -> float:
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
    size = kernel_size_from_sigma(sigma) if size is None else int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    coords = torch.arange(size, dtype=dtype, device=device) - size // 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(torch.finfo(kernel.dtype).eps)


def mexican_hat_kernel2d(
    sigma_center: float,
    sigma_surround: float = 1.5,
    *,
    size: int | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Create a zero-sum Mexican-hat kernel as a difference of Gaussians."""
    if sigma_center <= 0 or sigma_surround <= 0:
        raise ValueError("sigma_center and sigma_surround must be positive")
    size = kernel_size_from_sigma(sigma_center) if size is None else int(size)
    if size <= 0 or size % 2 == 0:
        raise ValueError("size must be a positive odd integer")

    center = gaussian_kernel2d(sigma_center, size=size, dtype=dtype, device=device)
    surround = gaussian_kernel2d(sigma_surround, size=size, dtype=dtype, device=device)
    return center - surround


def collect_image_groups(source_root: str | Path, recursive: bool = False) -> list[tuple[Path, list[Path]]]:
    """Collect image paths grouped by their relative parent directory."""
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
        relative_parent = path.relative_to(source_root).parent
        groups.setdefault(relative_parent, []).append(path)
    return sorted(groups.items(), key=lambda item: item[0].as_posix())


def sequence_layout_warnings(groups: list[tuple[Path, list[Path]]]) -> list[str]:
    """Flag layouts where RCA state is reset only by split-like directories."""
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
    frames = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {path}")
        if image.ndim == 2:
            gray = image
        else:
            gray = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2GRAY)
        frames.append(torch.from_numpy(gray).to(torch.float32).div(255.0).unsqueeze(0))
    return torch.stack(frames, dim=0)


def save_motion_image(path: str | Path, motion: torch.Tensor, save_rgb: bool = True) -> Path:
    """Save a normalized motion map as uint8 PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    motion_2d = motion.detach().cpu().squeeze().clamp_min(0.0).numpy()
    motion_2d = _max_normalize_array(motion_2d, MotionMapConfig().eps)
    image = np.clip(np.rint(motion_2d * 255.0), 0, 255).astype(np.uint8)
    image = np.repeat(image[..., None], 3, axis=2) if save_rgb else image
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Failed to write motion image: {path}")
    return path


def _conv2d_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kernel_4d = kernel.to(dtype=x.dtype, device=x.device).view(1, 1, kernel.shape[0], kernel.shape[1])
    padding = (int(kernel_4d.shape[-2] // 2), int(kernel_4d.shape[-1] // 2))
    return F.conv2d(x, kernel_4d, padding=padding)


def _sobel_magnitude(x: torch.Tensor) -> torch.Tensor:
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
    max_value = x.amax(dim=(-2, -1), keepdim=True)
    normalized = x / max_value.clamp_min(eps)
    return torch.where(max_value > eps, normalized, torch.zeros_like(x))


def _max_normalize_array(x: np.ndarray, eps: float) -> np.ndarray:
    max_value = float(x.max(initial=0.0))
    if max_value <= eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x / max_value).astype(np.float32, copy=False)


def _adapt(frame: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
    high_response = config.g_p * torch.tanh(frame - config.theta_p)
    low_response = frame * 0.1
    return torch.where(frame > config.theta_p, high_response, low_response)


def _horizontal_cell_response(frame: torch.Tensor, gaussian_kernel: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
    adapted = _adapt(frame, config)
    suppressed = adapted - config.horizontal_inhibition * _conv2d_same(adapted, gaussian_kernel)
    return F.relu(suppressed)


def _bipolar_cell_response(horizontal: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
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
    difference = config.beta * torch.abs(contrast - prev_contrast)
    if use_sobel_first:
        sobel = config.beta * _sobel_magnitude(contrast)
        response = state_valid * difference + (1.0 - state_valid) * sobel
    else:
        response = difference
    return config.alpha * prev_memory + (1.0 - config.alpha) * response


def _motion_response(
    contrast: torch.Tensor,
    memory: torch.Tensor,
    mexican_hat_kernel: torch.Tensor,
    config: MotionMapConfig,
) -> torch.Tensor:
    integrated = contrast + config.gamma_a * memory
    spatial = _conv2d_same(integrated, mexican_hat_kernel)
    thresholded = F.relu(torch.tanh(spatial + config.gamma_tau * memory - config.theta_m))
    return config.g_m * thresholded


def _reference_enhance(motion: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
    powered = motion.detach().cpu().squeeze().clamp_min(0.0).pow(config.gamma_p).numpy().astype(np.float32, copy=False)
    smoothed = cv2.bilateralFilter(
        powered,
        d=config.bilateral_diameter,
        sigmaColor=config.bilateral_sigma_color,
        sigmaSpace=config.bilateral_sigma_space,
    )
    return torch.from_numpy(smoothed).unsqueeze(0).unsqueeze(0)


def _onnx_enhance(motion: torch.Tensor, smoothing_kernel: torch.Tensor, config: MotionMapConfig) -> torch.Tensor:
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
    """Generate a reference motion-map sequence from grayscale frames."""
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
        outputs.append(_reference_enhance(fused, config).squeeze(0))

    return torch.stack(outputs, dim=0).to(original_device)


class OnnxMotionMapModule(nn.Module):
    """ONNX-friendly single-step RCA motion-map module."""

    def __init__(self, config: MotionMapConfig | None = None):
        super().__init__()
        self.config = config or MotionMapConfig()
        self.register_buffer(
            "gaussian_kernel",
            gaussian_kernel2d(self.config.gaussian_sigma, size=self.config.gaussian_kernel_size),
        )
        self.register_buffer(
            "mexican_hat_kernel",
            mexican_hat_kernel2d(
                self.config.mexican_hat_sigma_center,
                self.config.mexican_hat_sigma_surround,
                size=self.config.mexican_hat_size,
            ),
        )
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
        """Compute one motion map and the next recurrent states.

        The API keeps the planned `adapt_state` name for compatibility, but the tensor
        stores the previous contrast map required by the temporal difference stage.
        """
        valid = state_valid.to(dtype=frame.dtype)
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
        motion = _onnx_enhance(fused, self.smoothing_kernel, self.config)
        return motion, next_contrast, next_memory


__all__ = (
    "IMAGE_SUFFIXES",
    "MotionMapConfig",
    "OnnxMotionMapModule",
    "collect_image_groups",
    "gaussian_kernel2d",
    "generate_sequence",
    "kernel_size_from_sigma",
    "load_grayscale_frames",
    "mexican_hat_kernel2d",
    "save_motion_image",
    "sequence_layout_warnings",
)
