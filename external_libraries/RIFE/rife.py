"""
Minimal vendored RIFE 4.25 inference backend.

Public API is exported through this package's __init__.py:

    calculate_optical_flow(image_1, image_2, model="4.25")
    interpolate_between_two_frames(image_1, image_2, model="4.25")

Input image tensors:
    [H, W, C], floating point, normally [0, 1]

Optical-flow outputs:
    [H, W, 2]

Supported model variants:
    - "4.25"
    - "4.25.lite"

Expected weight files:
    models/flownet_v4.25.pkl
    models/flownet_v4.25.lite.pkl
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_KASKI_RIFE_PACKAGE_DIR = Path(__file__).resolve().parent
_KASKI_RIFE_MODELS_DIR = _KASKI_RIFE_PACKAGE_DIR / "models"


# =============================================================================
# Model configuration
# =============================================================================

@dataclass(frozen=True)
class _ModelConfig:
    weights_file: str
    block_channels: Tuple[int, int, int, int, int]
    scale_list: Tuple[float, float, float, float, float]
    modulo: int


_KASKI_RIFE_MODEL_CONFIGS: Dict[str, _ModelConfig] = {
    "4.25": _ModelConfig(
        weights_file="flownet_v4.25.pkl",
        block_channels=(192, 128, 96, 64, 32),
        scale_list=(16.0, 8.0, 4.0, 2.0, 1.0),
        modulo=64,
    ),
    "4.25.lite": _ModelConfig(
        weights_file="flownet_v4.25.lite.pkl",
        block_channels=(192, 128, 96, 64, 24),
        scale_list=(32.0, 16.0, 8.0, 4.0, 1.0),
        modulo=128,
    ),
}


# =============================================================================
# Caches
# =============================================================================

_KASKI_MODEL_CACHE: Dict[
    Tuple[str, str, torch.dtype],
    "_IFNet",
] = {}
_KASKI_MODEL_CACHE_LOCK = Lock()

_KASKI_GRID_CACHE: Dict[
    Tuple[str, torch.dtype, int, int],
    torch.Tensor,
] = {}
_KASKI_GRID_CACHE_LOCK = Lock()


# =============================================================================
# Network blocks
# =============================================================================

def _conv(
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    dilation: int = 1,
) -> nn.Sequential:

    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=True,
        ),
        nn.LeakyReLU(0.2, inplace=True),
    )


class _Head(nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.cnn0 = nn.Conv2d(3, 16, 3, 2, 1)
        self.cnn1 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn2 = nn.Conv2d(16, 16, 3, 1, 1)
        self.cnn3 = nn.ConvTranspose2d(16, 4, 4, 2, 1)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clamp(0.0, 1.0)
        x = self.relu(self.cnn0(x))
        x = self.relu(self.cnn1(x))
        x = self.relu(self.cnn2(x))
        return self.cnn3(x)


class _ResConv(nn.Module):

    def __init__(
        self,
        channels: int,
        dilation: int = 1,
    ) -> None:

        super().__init__()

        self.conv = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            groups=1,
        )

        self.beta = nn.Parameter(
            torch.ones((1, channels, 1, 1))
        )

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(
            self.conv(x) * self.beta + x
        )


class _IFBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        channels: int,
    ) -> None:

        super().__init__()

        self.conv0 = nn.Sequential(
            _conv(in_channels, channels // 2, 3, 2, 1),
            _conv(channels // 2, channels, 3, 2, 1),
        )

        self.convblock = nn.Sequential(
            *[_ResConv(channels) for _ in range(8)]
        )

        # 52 channels -> PixelShuffle(2) -> 13 channels:
        # 4 flow, 1 mask, 8 recurrent features.
        self.lastconv = nn.Sequential(
            nn.ConvTranspose2d(
                channels,
                4 * 13,
                4,
                2,
                1,
            ),
            nn.PixelShuffle(2),
        )

    def forward(
        self,
        x: torch.Tensor,
        flow: torch.Tensor | None,
        scale: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        x = F.interpolate(
            x,
            scale_factor=1.0 / scale,
            mode="bilinear",
            align_corners=False,
        )

        if flow is not None:
            scaled_flow = F.interpolate(
                flow,
                scale_factor=1.0 / scale,
                mode="bilinear",
                align_corners=False,
            ) / scale

            x = torch.cat(
                (x, scaled_flow),
                dim=1,
            )

        features = self.convblock(
            self.conv0(x)
        )

        output = self.lastconv(features)

        output = F.interpolate(
            output,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
        )

        delta_flow = output[:, :4] * scale
        mask = output[:, 4:5]
        recurrent_features = output[:, 5:]

        return (
            delta_flow,
            mask,
            recurrent_features,
        )


# =============================================================================
# Warping
# =============================================================================

def _get_base_grid(
    device: torch.device,
    dtype: torch.dtype,
    height: int,
    width: int,
) -> torch.Tensor:

    key = (
        str(device),
        dtype,
        height,
        width,
    )

    with _KASKI_GRID_CACHE_LOCK:

        cached = _KASKI_GRID_CACHE.get(key)

        if cached is not None:
            return cached

        horizontal = torch.linspace(
            -1.0,
            1.0,
            width,
            device=device,
            dtype=dtype,
        ).view(
            1, 1, 1, width
        ).expand(
            1, 1, height, width
        )

        vertical = torch.linspace(
            -1.0,
            1.0,
            height,
            device=device,
            dtype=dtype,
        ).view(
            1, 1, height, 1
        ).expand(
            1, 1, height, width
        )

        grid = torch.cat(
            (horizontal, vertical),
            dim=1,
        )

        _KASKI_GRID_CACHE[key] = grid

        return grid


def _warp(
    image: torch.Tensor,
    flow: torch.Tensor,
) -> torch.Tensor:
    """
    Backward warp using RIFE's normalized optical-flow convention.
    """

    _, _, height, width = image.shape

    if flow.dtype != image.dtype:
        flow = flow.to(dtype=image.dtype)

    base_grid = _get_base_grid(
        device=image.device,
        dtype=image.dtype,
        height=height,
        width=width,
    )

    x_divisor = max(
        (width - 1.0) / 2.0,
        1e-6,
    )

    y_divisor = max(
        (height - 1.0) / 2.0,
        1e-6,
    )

    normalized_flow = torch.cat(
        (
            flow[:, 0:1] / x_divisor,
            flow[:, 1:2] / y_divisor,
        ),
        dim=1,
    )

    sampling_grid = (
        base_grid
        + normalized_flow
    ).permute(
        0, 2, 3, 1
    )

    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


# =============================================================================
# IFNet
# =============================================================================

class _IFNet(nn.Module):

    def __init__(
        self,
        config: _ModelConfig,
    ) -> None:

        super().__init__()

        c0, c1, c2, c3, c4 = (
            config.block_channels
        )

        self.block0 = _IFBlock(15, c0)
        self.block1 = _IFBlock(28, c1)
        self.block2 = _IFBlock(28, c2)
        self.block3 = _IFBlock(28, c3)
        self.block4 = _IFBlock(28, c4)

        self.encode = _Head()
        self.scale_list = config.scale_list

    def forward(
        self,
        image_0: torch.Tensor,
        image_1: torch.Tensor,
        timestep: float = 0.5,
        return_flow: bool = False,
    ) -> torch.Tensor:

        image_0 = image_0.clamp(0.0, 1.0)
        image_1 = image_1.clamp(0.0, 1.0)

        features_0 = self.encode(image_0)
        features_1 = self.encode(image_1)

        time_map = torch.full(
            (
                image_0.shape[0],
                1,
                image_0.shape[2],
                image_0.shape[3],
            ),
            float(timestep),
            device=image_0.device,
            dtype=image_0.dtype,
        )

        blocks = (
            self.block0,
            self.block1,
            self.block2,
            self.block3,
            self.block4,
        )

        warped_0 = image_0
        warped_1 = image_1

        flow = None
        mask = None
        recurrent_features = None

        for index, block in enumerate(blocks):

            if flow is None:

                block_input = torch.cat(
                    (
                        image_0,
                        image_1,
                        features_0,
                        features_1,
                        time_map,
                    ),
                    dim=1,
                )

                flow, mask, recurrent_features = block(
                    block_input,
                    flow=None,
                    scale=self.scale_list[index],
                )

            else:

                warped_features_0 = _warp(
                    features_0,
                    flow[:, :2],
                )

                warped_features_1 = _warp(
                    features_1,
                    flow[:, 2:4],
                )

                block_input = torch.cat(
                    (
                        warped_0,
                        warped_1,
                        warped_features_0,
                        warped_features_1,
                        time_map,
                        mask,
                        recurrent_features,
                    ),
                    dim=1,
                )

                delta_flow, mask, recurrent_features = block(
                    block_input,
                    flow=flow,
                    scale=self.scale_list[index],
                )

                flow = flow + delta_flow

            warped_0 = _warp(
                image_0,
                flow[:, :2],
            )

            warped_1 = _warp(
                image_1,
                flow[:, 2:4],
            )

        if return_flow:
            return flow

        blend_mask = torch.sigmoid(mask)

        return (
            warped_0 * blend_mask
            + warped_1 * (1.0 - blend_mask)
        )


# =============================================================================
# Weight loading
# =============================================================================

def _normalize_state_dict(
    raw_state: object,
) -> Dict[str, torch.Tensor]:

    if isinstance(raw_state, dict):

        for candidate_key in (
            "state_dict",
            "model",
            "flownet",
        ):
            candidate = raw_state.get(
                candidate_key
            )

            if isinstance(candidate, dict):
                raw_state = candidate
                break

    if not isinstance(raw_state, dict):
        raise RuntimeError(
            "RIFE weights do not contain a state dictionary."
        )

    cleaned: Dict[
        str,
        torch.Tensor,
    ] = {}

    for key, value in raw_state.items():

        if (
            not isinstance(key, str)
            or not torch.is_tensor(value)
        ):
            continue

        clean_key = key

        while clean_key.startswith(
            "module."
        ):
            clean_key = clean_key[
                len("module.") :
            ]

        if clean_key.startswith(
            "flownet."
        ):
            clean_key = clean_key[
                len("flownet.") :
            ]

        cleaned[clean_key] = value

    if not cleaned:
        raise RuntimeError(
            "RIFE state dictionary is empty after key normalization."
        )

    return cleaned


def _load_state_dict(
    path: Path,
) -> Dict[str, torch.Tensor]:

    if not path.is_file():
        raise FileNotFoundError(
            f"RIFE weights not found: {path}\n"
            "Expected both model files inside "
            "external_libraries/RIFE/models/."
        )

    try:
        raw_state = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

    except TypeError:
        raw_state = torch.load(
            path,
            map_location="cpu",
        )

    return _normalize_state_dict(
        raw_state
    )


# =============================================================================
# Device / dtype
# =============================================================================

def _get_inference_device(
    input_device: torch.device,
) -> torch.device:
    """
    Prefer CUDA.

    If input already lives on CUDA, use that device.
    Otherwise use the current CUDA device if available.
    Fall back to CPU if CUDA is unavailable.
    """

    if input_device.type == "cuda":
        return input_device

    if torch.cuda.is_available():
        return torch.device(
            "cuda",
            torch.cuda.current_device(),
        )

    print(
        "[KASKI RIFE] CUDA unavailable. "
        "Falling back to CPU inference."
    )

    return torch.device("cpu")


def _select_inference_dtype(
    device: torch.device,
) -> torch.dtype:

    if device.type == "cuda":
        return torch.float16

    return torch.float32


# =============================================================================
# Model cache
# =============================================================================

def _get_model(
    model_name: str,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[_IFNet, _ModelConfig]:

    try:
        config = _KASKI_RIFE_MODEL_CONFIGS[
            model_name
        ]

    except KeyError as exc:
        supported = ", ".join(
            sorted(
                _KASKI_RIFE_MODEL_CONFIGS
            )
        )

        raise ValueError(
            f"Unsupported RIFE model '{model_name}'. "
            f"Supported: {supported}"
        ) from exc

    cache_key = (
        model_name,
        str(device),
        dtype,
    )

    with _KASKI_MODEL_CACHE_LOCK:

        cached = _KASKI_MODEL_CACHE.get(
            cache_key
        )

        if cached is not None:
            return cached, config

        model = _IFNet(config)

        state_dict = _load_state_dict(
            _KASKI_RIFE_MODELS_DIR
            / config.weights_file
        )

        missing_keys, _unexpected_keys = (
            model.load_state_dict(
                state_dict,
                strict=False,
            )
        )

        if missing_keys:

            missing_preview = ", ".join(
                missing_keys[:8]
            )

            if len(missing_keys) > 8:
                missing_preview += ", ..."

            raise RuntimeError(
                f"RIFE checkpoint '{config.weights_file}' "
                f"is incompatible with the pinned "
                f"{model_name} architecture. "
                f"Missing keys: {missing_preview}"
            )

        model.eval()
        model.requires_grad_(False)

        model.to(
            device=device,
            dtype=dtype,
        )

        _KASKI_MODEL_CACHE[
            cache_key
        ] = model

        return model, config


# =============================================================================
# Padding
# =============================================================================

def _pad_to_modulo(
    image_0: torch.Tensor,
    image_1: torch.Tensor,
    modulo: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    int,
    int,
]:

    height = image_0.shape[2]
    width = image_0.shape[3]

    padded_height = (
        (height + modulo - 1)
        // modulo
    ) * modulo

    padded_width = (
        (width + modulo - 1)
        // modulo
    ) * modulo

    pad_right = padded_width - width
    pad_bottom = padded_height - height

    if pad_right or pad_bottom:

        padding = (
            0,
            pad_right,
            0,
            pad_bottom,
        )

        image_0 = F.pad(
            image_0,
            padding,
        )

        image_1 = F.pad(
            image_1,
            padding,
        )

    return (
        image_0,
        image_1,
        height,
        width,
    )


# =============================================================================
# Input validation / preparation
# =============================================================================

def _validate_input_frames(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
) -> None:

    if (
        not torch.is_tensor(image_1)
        or not torch.is_tensor(image_2)
    ):
        raise TypeError(
            "RIFE inputs must be torch.Tensor instances."
        )

    if (
        image_1.ndim != 3
        or image_2.ndim != 3
    ):
        raise ValueError(
            "RIFE expects individual HWC frames. "
            f"Received {tuple(image_1.shape)} "
            f"and {tuple(image_2.shape)}."
        )

    if image_1.shape != image_2.shape:
        raise ValueError(
            "RIFE input frames must have identical shapes. "
            f"Received {tuple(image_1.shape)} "
            f"and {tuple(image_2.shape)}."
        )

    if image_1.shape[-1] < 3:
        raise ValueError(
            "RIFE requires at least 3 channels, "
            f"got {image_1.shape[-1]}."
        )

    if image_1.device != image_2.device:
        raise ValueError(
            "RIFE input frames must be on the same device. "
            f"Received {image_1.device} "
            f"and {image_2.device}."
        )

    if (
        not image_1.is_floating_point()
        or not image_2.is_floating_point()
    ):
        raise TypeError(
            "RIFE expects floating-point frame tensors in [0, 1]."
        )


def _prepare_rife_inputs(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    model: str,
):
    """
    Shared preparation for optical-flow and interpolation calls.
    """

    _validate_input_frames(
        image_1,
        image_2,
    )

    original_device = image_1.device
    original_dtype = image_1.dtype

    inference_device = _get_inference_device(
        original_device
    )

    inference_dtype = _select_inference_dtype(
        inference_device
    )

    rife_model, config = _get_model(
        model,
        inference_device,
        inference_dtype,
    )

    rgb_1 = (
        image_1[..., :3]
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(
            device=inference_device,
            dtype=inference_dtype,
        )
    )

    rgb_2 = (
        image_2[..., :3]
        .permute(2, 0, 1)
        .unsqueeze(0)
        .contiguous()
        .to(
            device=inference_device,
            dtype=inference_dtype,
        )
    )

    (
        rgb_1,
        rgb_2,
        original_height,
        original_width,
    ) = _pad_to_modulo(
        rgb_1,
        rgb_2,
        config.modulo,
    )

    return (
        rife_model,
        rgb_1,
        rgb_2,
        original_height,
        original_width,
        original_device,
        original_dtype,
    )


# =============================================================================
# Public API: Optical Flow
# =============================================================================

def calculate_optical_flow(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    model: str = "4.25",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Calculate RIFE optical flow between two HWC frames.

    Args:
        image_1:
            First frame [H,W,C].

        image_2:
            Second frame [H,W,C].

        model:
            "4.25" or "4.25.lite".

    Returns:
        flow_image_1:
            [H,W,2]

        flow_image_2:
            [H,W,2]

    The two flow fields are the final midpoint-directed flow fields
    RIFE uses internally to warp image_1 and image_2.

    Outputs are returned on the same device and in the same dtype
    as image_1.
    """

    (
        rife_model,
        rgb_1,
        rgb_2,
        original_height,
        original_width,
        original_device,
        original_dtype,
    ) = _prepare_rife_inputs(
        image_1,
        image_2,
        model,
    )

    with torch.inference_mode():

        flow = rife_model(
            rgb_1,
            rgb_2,
            timestep=0.5,
            return_flow=True,
        )

        flow = flow[
            :,
            :,
            :original_height,
            :original_width,
        ]

    flow_image_1 = (
        flow[0, :2]
        .permute(1, 2, 0)
        .to(
            device=original_device,
            dtype=original_dtype,
        )
        .contiguous()
    )

    flow_image_2 = (
        flow[0, 2:4]
        .permute(1, 2, 0)
        .to(
            device=original_device,
            dtype=original_dtype,
        )
        .contiguous()
    )

    return (
        flow_image_1,
        flow_image_2,
    )


# =============================================================================
# Public API: Frame interpolation
# =============================================================================

def interpolate_between_two_frames(
    image_1: torch.Tensor,
    image_2: torch.Tensor,
    model: str = "4.25",
) -> torch.Tensor:
    """
    Interpolate the temporal midpoint between two HWC image tensors.

    Args:
        image_1:
            First frame [H,W,C].

        image_2:
            Second frame [H,W,C].

        model:
            "4.25" or "4.25.lite".

    Returns:
        Midpoint frame [H,W,C].

    The result is returned on the same device and in the same dtype
    as image_1.

    CUDA:
        FP16 inference.

    CPU fallback:
        FP32 inference.
    """

    channel_count = image_1.shape[-1]

    (
        rife_model,
        rgb_1,
        rgb_2,
        original_height,
        original_width,
        original_device,
        original_dtype,
    ) = _prepare_rife_inputs(
        image_1,
        image_2,
        model,
    )

    with torch.inference_mode():

        midpoint_rgb = rife_model(
            rgb_1,
            rgb_2,
            timestep=0.5,
            return_flow=False,
        )

        midpoint_rgb = midpoint_rgb[
            :,
            :,
            :original_height,
            :original_width,
        ]

        midpoint_rgb = midpoint_rgb.clamp(
            0.0,
            1.0,
        )

    midpoint = (
        midpoint_rgb[0]
        .permute(1, 2, 0)
        .to(
            device=original_device,
            dtype=original_dtype,
        )
        .contiguous()
    )

    # RIFE itself only handles RGB.
    if channel_count > 3:

        extra_channels = torch.lerp(
            image_1[..., 3:],
            image_2[..., 3:],
            0.5,
        )

        midpoint = torch.cat(
            (
                midpoint,
                extra_channels,
            ),
            dim=-1,
        )

    return midpoint