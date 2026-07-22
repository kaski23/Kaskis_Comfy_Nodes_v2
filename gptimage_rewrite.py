import base64
import traceback
from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

from comfy.utils import common_upscale
from comfy_api.latest import IO, Input
from comfy_api_nodes.apis.openai import (
    OpenAIImageEditRequest,
    OpenAIImageGenerationRequest,
    OpenAIImageGenerationResponse,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    download_url_to_bytesio,
    downscale_image_tensor,
    sync_op,
    validate_string,
)


CATEGORY = "KASKI/api-adaptions/openai"
SETTINGS_TYPE = "KASKI_GPT_IMAGE_SETTINGS"
MAX_REFERENCE_IMAGES = 16

VALID_GPT_IMAGE_MODELS = {
    "gpt-image-1",
    "gpt-image-1.5",
    "gpt-image-2",
}

VALID_GPT_IMAGE_QUALITIES = {
    "low",
    "medium",
    "high",
}

GPT_IMAGE_2_BACKGROUNDS = {
    "auto",
    "opaque",
}

LEGACY_GPT_IMAGE_BACKGROUNDS = {
    "auto",
    "opaque",
    "transparent",
}

GPT_IMAGE_2_SIZES = {
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "2048x2048",
    "2048x1152",
    "1152x2048",
    "3840x2160",
    "2160x3840",
    "Custom",
}

LEGACY_GPT_IMAGE_SIZES = {
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
}


async def validate_and_cast_response(
    response: OpenAIImageGenerationResponse,
    timeout: int | None = None,
) -> torch.Tensor:
    data = response.data
    if not data:
        raise ValueError("No images returned from API endpoint")

    image_tensors: list[torch.Tensor] = []

    for img_data in data:
        if img_data.b64_json:
            img_io = BytesIO(base64.b64decode(img_data.b64_json))
        elif img_data.url:
            img_io = BytesIO()
            await download_url_to_bytesio(
                img_data.url,
                img_io,
                timeout=timeout,
            )
        else:
            raise ValueError(
                "Invalid image payload – neither URL nor base64 data present."
            )

        pil_img = Image.open(img_io).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        image_tensors.append(torch.from_numpy(arr))

    # size="auto" can return images whose dimensions differ slightly.
    # ComfyUI batches must have one consistent tensor shape, so match the
    # current upstream behavior and resize every result to the first image.
    ref_h, ref_w = image_tensors[0].shape[:2]
    for index, tensor in enumerate(image_tensors):
        if tensor.shape[:2] == (ref_h, ref_w):
            continue

        samples = tensor.unsqueeze(0).movedim(-1, 1)
        samples = common_upscale(
            samples,
            ref_w,
            ref_h,
            "bilinear",
            "center",
        )
        image_tensors[index] = samples.movedim(1, -1).squeeze(0)

    return torch.stack(image_tensors, dim=0)


def calculate_tokens_price_image_1(
    response: OpenAIImageGenerationResponse,
) -> float | None:
    return (
        (response.usage.input_tokens * 10.0)
        + (response.usage.output_tokens * 40.0)
    ) / 1_000_000.0


def calculate_tokens_price_image_1_5(
    response: OpenAIImageGenerationResponse,
) -> float | None:
    return (
        (response.usage.input_tokens * 8.0)
        + (response.usage.output_tokens * 32.0)
    ) / 1_000_000.0


def calculate_tokens_price_image_2_0(
    response: OpenAIImageGenerationResponse,
) -> float | None:
    return (
        (response.usage.input_tokens * 8.0)
        + (response.usage.output_tokens * 30.0)
    ) / 1_000_000.0


def _black_image(width: int = 1024, height: int = 1024) -> torch.Tensor:
    return torch.zeros((1, height, width, 4), dtype=torch.float32)


def _log_soft_error(where: str, error: Exception) -> None:
    print(f"[KASKI GPTImage2] {where}: {type(error).__name__}: {error}")
    traceback.print_exc()


def _validate_custom_size(width: int, height: int) -> None:
    if type(width) is not int or type(height) is not int:
        raise TypeError("custom_width and custom_height must be integers.")

    if not 1024 <= width <= 3840 or not 1024 <= height <= 3840:
        raise ValueError(
            "Custom width and height must each be between 1024 and 3840; "
            f"received {width}x{height}."
        )

    if width % 16 != 0 or height % 16 != 0:
        raise ValueError(
            "Custom width and height must be multiples of 16; "
            f"received {width}x{height}."
        )

    if max(width, height) > 3840:
        raise ValueError(
            "Custom resolution max edge must be <= 3840; "
            f"received {width}x{height}."
        )

    ratio = max(width, height) / min(width, height)
    if ratio > 3:
        raise ValueError(
            "Custom resolution aspect ratio must not exceed 3:1; "
            f"received {width}x{height}."
        )

    total_pixels = width * height
    if not 655_360 <= total_pixels <= 8_294_400:
        raise ValueError(
            "Custom resolution total pixels must be between 655,360 and "
            f"8,294,400; received {total_pixels}."
        )


def _validate_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError(
            "settings must come from the OpenAI GPT Image Settings node."
        )

    required_keys = {
        "model_id",
        "quality",
        "background",
        "size",
        "custom_width",
        "custom_height",
        "n",
    }
    missing_keys = required_keys.difference(settings)
    if missing_keys:
        raise ValueError(
            "Settings object is incomplete. Missing: "
            + ", ".join(sorted(missing_keys))
        )

    normalized = dict(settings)

    model_id = normalized["model_id"]
    quality = normalized["quality"]
    background = normalized["background"]
    size = normalized["size"]
    custom_width = normalized["custom_width"]
    custom_height = normalized["custom_height"]
    n = normalized["n"]

    if not isinstance(model_id, str) or model_id not in VALID_GPT_IMAGE_MODELS:
        raise ValueError(
            f"Invalid model '{model_id}'. Allowed: "
            f"{sorted(VALID_GPT_IMAGE_MODELS)}"
        )

    if not isinstance(quality, str):
        raise TypeError("quality must be a string.")
    quality = quality.strip().lower()
    if quality not in VALID_GPT_IMAGE_QUALITIES:
        raise ValueError(
            f"Invalid quality '{quality}'. Allowed: "
            f"{sorted(VALID_GPT_IMAGE_QUALITIES)}"
        )

    if not isinstance(background, str):
        raise TypeError("background must be a string.")
    background = background.strip().lower()

    if not isinstance(size, str):
        raise TypeError("size must be a string.")
    size = size.strip()

    if type(n) is not int or not 1 <= n <= 8:
        raise ValueError("n must be an integer between 1 and 8.")

    if model_id == "gpt-image-2":
        if background not in GPT_IMAGE_2_BACKGROUNDS:
            raise ValueError(
                "GPT Image 2 supports only auto or opaque backgrounds."
            )

        if size not in GPT_IMAGE_2_SIZES:
            raise ValueError(
                f"Invalid GPT Image 2 size '{size}'. Allowed: "
                f"{sorted(GPT_IMAGE_2_SIZES)}"
            )

        if size == "Custom":
            _validate_custom_size(custom_width, custom_height)
    else:
        if background not in LEGACY_GPT_IMAGE_BACKGROUNDS:
            raise ValueError(
                f"Invalid legacy GPT Image background '{background}'."
            )

        if size not in LEGACY_GPT_IMAGE_SIZES:
            raise ValueError(
                f"Resolution '{size}' is only supported by GPT Image 2."
            )

        if custom_width is not None or custom_height is not None:
            raise ValueError(
                "custom_width and custom_height must be None for "
                f"{model_id}."
            )

    normalized.update(
        {
            "model_id": model_id,
            "quality": quality,
            "background": background,
            "size": size,
            "n": n,
        }
    )
    return normalized


def _default_settings() -> dict[str, Any]:
    return {
        "model_id": "gpt-image-2",
        "quality": "low",
        "background": "auto",
        "size": "auto",
        "custom_width": 1024,
        "custom_height": 1024,
        "n": 1,
    }


def _resolve_request_size(settings: dict[str, Any]) -> str:
    if settings["size"] != "Custom":
        return settings["size"]

    width = settings["custom_width"]
    height = settings["custom_height"]
    _validate_custom_size(width, height)
    return f"{width}x{height}"


def _price_extractor_for_model(model_id: str):
    if model_id == "gpt-image-1":
        return calculate_tokens_price_image_1
    if model_id == "gpt-image-1.5":
        return calculate_tokens_price_image_1_5
    if model_id == "gpt-image-2":
        return calculate_tokens_price_image_2_0
    raise ValueError(f"Unknown model: {model_id}")


def _collect_image_tensors(
    images: (
        Input.Image
        | list[Input.Image]
        | dict[str, Input.Image]
        | None
    ),
) -> list[torch.Tensor]:
    if images is None:
        return []

    if isinstance(images, dict):
        image_values = [
            tensor for tensor in images.values() if tensor is not None
        ]
    elif isinstance(images, list):
        image_values = [tensor for tensor in images if tensor is not None]
    else:
        image_values = [images]

    flat: list[torch.Tensor] = []
    for tensor in image_values:
        if len(tensor.shape) == 4:
            flat.extend(
                tensor[index : index + 1]
                for index in range(tensor.shape[0])
            )
        elif len(tensor.shape) == 3:
            flat.append(tensor.unsqueeze(0))
        else:
            raise ValueError(
                "Reference images must be HWC or batched BHWC tensors."
            )

    if len(flat) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"GPT Image supports at most {MAX_REFERENCE_IMAGES} reference "
            f"images; received {len(flat)}."
        )

    return flat


def _settings_inputs_for_legacy_model() -> list[Input]:
    return [
        IO.Combo.Input(
            "size",
            default="auto",
            options=[
                "auto",
                "1024x1024",
                "1024x1536",
                "1536x1024",
            ],
            tooltip="Image size.",
        ),
        IO.Combo.Input(
            "background",
            default="auto",
            options=["auto", "opaque", "transparent"],
            tooltip="Return image with or without a transparent background.",
        ),
        IO.Combo.Input(
            "quality",
            default="low",
            options=["low", "medium", "high"],
            tooltip="Image quality, affecting cost and generation time.",
        ),
    ]


class OpenAIGPTImageSettings(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImageSettings_KASKI",
            display_name="OpenAI GPT Image Settings",
            category=CATEGORY,
            description=(
                "Central settings object for one or more KASKI GPT Image "
                "generator nodes. Fan this output out to every generator "
                "that should share the same model configuration."
            ),
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "gpt-image-2",
                            [
                                IO.Combo.Input(
                                    "size",
                                    default="auto",
                                    options=[
                                        "auto",
                                        "1024x1024",
                                        "1024x1536",
                                        "1536x1024",
                                        "2048x2048",
                                        "2048x1152",
                                        "1152x2048",
                                        "3840x2160",
                                        "2160x3840",
                                        "Custom",
                                    ],
                                    tooltip=(
                                        "Image size. Select Custom to use "
                                        "custom width and height."
                                    ),
                                ),
                                IO.Int.Input(
                                    "custom_width",
                                    default=1024,
                                    min=1024,
                                    max=3840,
                                    step=16,
                                    tooltip=(
                                        "Used only when size is Custom. "
                                        "Must be a multiple of 16."
                                    ),
                                ),
                                IO.Int.Input(
                                    "custom_height",
                                    default=1024,
                                    min=1024,
                                    max=3840,
                                    step=16,
                                    tooltip=(
                                        "Used only when size is Custom. "
                                        "Must be a multiple of 16."
                                    ),
                                ),
                                IO.Combo.Input(
                                    "background",
                                    default="auto",
                                    options=["auto", "opaque"],
                                    tooltip=(
                                        "GPT Image 2 does not support "
                                        "transparent output."
                                    ),
                                ),
                                IO.Combo.Input(
                                    "quality",
                                    default="low",
                                    options=["low", "medium", "high"],
                                    tooltip=(
                                        "Image quality, affecting cost and "
                                        "generation time."
                                    ),
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            "gpt-image-1.5",
                            _settings_inputs_for_legacy_model(),
                        ),
                        IO.DynamicCombo.Option(
                            "gpt-image-1",
                            _settings_inputs_for_legacy_model(),
                        ),
                    ],
                    tooltip="Model and model-specific image settings.",
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="How many images to generate per request.",
                    display_mode=IO.NumberDisplay.number,
                ),
            ],
            outputs=[
                IO.Custom(SETTINGS_TYPE).Output(
                    display_name="settings"
                ),
            ],
        )

    @classmethod
    def execute(
        cls,
        model: dict[str, Any],
        n: int,
    ) -> IO.NodeOutput:
        try:
            model_id = model["model"]
            is_gpt_image_2 = model_id == "gpt-image-2"

            settings = {
                "model_id": model_id,
                "quality": model["quality"],
                "background": model["background"],
                "size": model["size"],
                "custom_width": (
                    int(model.get("custom_width", 1024))
                    if is_gpt_image_2
                    else None
                ),
                "custom_height": (
                    int(model.get("custom_height", 1024))
                    if is_gpt_image_2
                    else None
                ),
                "n": int(n),
            }

            return IO.NodeOutput(_validate_settings(settings))
        except Exception as error:
            _log_soft_error("OpenAIGPTImageSettings.execute", error)
            return IO.NodeOutput(_default_settings())


class OpenAIGPTImage1(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            # Keep the existing KASKI node ID to minimize workflow breakage.
            node_id="OpenAIGPTImage1_KASKI",
            display_name="OpenAI GPT Image IO-unlocked",
            category=CATEGORY,
            description=(
                "Generate or edit images via OpenAI's GPT Image endpoint. "
                "Shared model configuration is supplied by the OpenAI GPT "
                "Image Settings node."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for GPT Image.",
                ),
                IO.Custom(SETTINGS_TYPE).Input(
                    "settings",
                    tooltip=(
                        "Connect one central OpenAI GPT Image Settings node. "
                        "The same output can feed multiple generators."
                    ),
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2147483647,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip=(
                        "ComfyUI cache-buster and control-after-generate "
                        "value. The current GPT Image request schema does "
                        "not send it to OpenAI."
                    ),
                ),
                IO.Autogrow.Input(
                    "images",
                    template=IO.Autogrow.TemplateNames(
                        IO.Image.Input("image"),
                        names=[
                            f"image_{index}"
                            for index in range(1, MAX_REFERENCE_IMAGES + 1)
                        ],
                        min=0,
                    ),
                    tooltip=(
                        "Optional reference images for editing. Up to 16 "
                        "individual images or batched IMAGE tensors."
                    ),
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional inpainting mask. White areas are replaced. "
                        "Requires exactly one reference image."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        settings: dict[str, Any],
        seed: int,
        images: (
            Input.Image
            | list[Input.Image]
            | dict[str, Input.Image]
            | None
        ) = None,
        mask: Input.Image | None = None,
    ) -> IO.NodeOutput:
        # Upstream currently exposes seed only as a ComfyUI cache-buster.
        del seed

        try:
            validate_string(prompt, strip_whitespace=False)
            settings = _validate_settings(settings)

            model_id = settings["model_id"]
            quality = settings["quality"]
            background = settings["background"]
            n = settings["n"]
            size = _resolve_request_size(settings)
            price_extractor = _price_extractor_for_model(model_id)

            flat_images = _collect_image_tensors(images)

            if mask is not None and not flat_images:
                raise ValueError("Cannot use a mask without an input image.")

            if flat_images:
                files = []

                for index, single_image in enumerate(flat_images):
                    scaled_image = downscale_image_tensor(
                        single_image,
                        total_pixels=2048 * 2048,
                    ).squeeze()

                    image_np = (
                        scaled_image.numpy() * 255
                    ).astype(np.uint8)
                    pil_image = Image.fromarray(image_np)

                    image_bytes = BytesIO()
                    pil_image.save(image_bytes, format="PNG")
                    image_bytes.seek(0)

                    field_name = (
                        "image" if len(flat_images) == 1 else "image[]"
                    )
                    files.append(
                        (
                            field_name,
                            (
                                f"image_{index}.png",
                                image_bytes,
                                "image/png",
                            ),
                        )
                    )

                if mask is not None:
                    if len(flat_images) != 1:
                        raise ValueError(
                            "Cannot use a mask with multiple images."
                        )

                    reference_image = flat_images[0]
                    if mask.shape[1:] != reference_image.shape[1:-1]:
                        raise ValueError(
                            "Mask and image must have the same size."
                        )

                    _, height, width = mask.shape
                    rgba_mask = torch.zeros(
                        height,
                        width,
                        4,
                        device="cpu",
                    )
                    rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

                    scaled_mask = downscale_image_tensor(
                        rgba_mask.unsqueeze(0),
                        total_pixels=2048 * 2048,
                    ).squeeze()

                    mask_np = (
                        scaled_mask.numpy() * 255
                    ).astype(np.uint8)
                    mask_image = Image.fromarray(mask_np)

                    mask_bytes = BytesIO()
                    mask_image.save(mask_bytes, format="PNG")
                    mask_bytes.seek(0)

                    files.append(
                        (
                            "mask",
                            ("mask.png", mask_bytes, "image/png"),
                        )
                    )

                response = await sync_op(
                    cls,
                    ApiEndpoint(
                        path="/proxy/openai/images/edits",
                        method="POST",
                    ),
                    response_model=OpenAIImageGenerationResponse,
                    data=OpenAIImageEditRequest(
                        model=model_id,
                        prompt=prompt,
                        quality=quality,
                        background=background,
                        n=n,
                        size=size,
                        moderation="low",
                    ),
                    content_type="multipart/form-data",
                    files=files,
                    price_extractor=price_extractor,
                )
            else:
                response = await sync_op(
                    cls,
                    ApiEndpoint(
                        path="/proxy/openai/images/generations",
                        method="POST",
                    ),
                    response_model=OpenAIImageGenerationResponse,
                    data=OpenAIImageGenerationRequest(
                        model=model_id,
                        prompt=prompt,
                        quality=quality,
                        background=background,
                        n=n,
                        size=size,
                        moderation="low",
                    ),
                    price_extractor=price_extractor,
                )

            return IO.NodeOutput(
                await validate_and_cast_response(response)
            )
        except Exception as error:
            _log_soft_error("OpenAIGPTImage1.execute", error)
            return IO.NodeOutput(_black_image())


OPENAI_GPT_IMAGE_REWRITE_NODE_CLASS_MAPPINGS = {
    "OpenAIGPTImage1_KASKI": OpenAIGPTImage1,
    "OpenAIGPTImageSettings_KASKI": OpenAIGPTImageSettings,
}

OPENAI_GPT_IMAGE_REWRITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIGPTImage1_KASKI": "OpenAI GPT Image IO-unlocked",
    "OpenAIGPTImageSettings_KASKI": "OpenAI GPT Image Settings",
}
