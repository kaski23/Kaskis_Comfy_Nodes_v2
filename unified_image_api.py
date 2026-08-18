from __future__ import annotations

import base64
import traceback
from fnmatch import fnmatch
from io import BytesIO
from typing import Any, Literal

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
from comfy_api_nodes.apis.gemini import (
    GeminiContent,
    GeminiFileData,
    GeminiGenerateContentResponse,
    GeminiImageConfig,
    GeminiImageGenerateContentRequest,
    GeminiImageGenerationConfig,
    GeminiInlineData,
    GeminiMimeType,
    GeminiPart,
    GeminiRole,
    GeminiSystemInstructionContent,
    GeminiTextPart,
    GeminiThinkingConfig,
    Modality,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    bytesio_to_image_tensor,
    download_url_to_bytesio,
    download_url_to_image_tensor,
    get_number_of_images,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    validate_string,
)


CATEGORY = "KASKI/api-adaptions/image"
SETTINGS_TYPE = "KASKI_IMAGE_API_SETTINGS"

PROVIDER_OPENAI = "openai"
PROVIDER_GEMINI = "gemini"

OPENAI_PROVIDER_LABEL = "OpenAI GPT Image"
GEMINI_PROVIDER_LABEL = "Gemini / Nanobanana"

MAX_OPENAI_REFERENCE_IMAGES = 16
MAX_GEMINI_REFERENCE_IMAGES = 14
GEMINI_URL_IMAGE_BUDGET = 10
GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"


# -----------------------------------------------------------------------------
# Shared defaults / constants
# -----------------------------------------------------------------------------

GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)

OPENAI_MODEL_IDS = {
    "gpt-image-2",
    "gpt-image-1.5",
    "gpt-image-1",
}

OPENAI_QUALITIES = {
    "low",
    "medium",
    "high",
}

OPENAI_GPT_IMAGE_2_BACKGROUNDS = {
    "auto",
    "opaque",
}

OPENAI_LEGACY_BACKGROUNDS = {
    "auto",
    "opaque",
    "transparent",
}

OPENAI_GPT_IMAGE_2_SIZES = {
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

OPENAI_LEGACY_SIZES = {
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
}

GEMINI_MODEL_IDS = {
    "Gemini 3 Pro Image": "gemini-3-pro-image-preview",
    "Nano Banana 2 (Gemini 3.1 Flash Image)": "gemini-3.1-flash-image-preview",
    "Nano Banana 2 Lite": "gemini-3.1-flash-lite-image",
}

GEMINI_BASE_ASPECT_RATIOS = [
    "auto",
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
]

GEMINI_EXTENDED_ASPECT_RATIOS = GEMINI_BASE_ASPECT_RATIOS + [
    "1:4",
    "4:1",
    "8:1",
    "1:8",
]

GEMINI_VALID_MODALITIES = {"IMAGE", "IMAGE+TEXT"}
GEMINI_VALID_THINKING_LEVELS = {"MINIMAL", "HIGH"}


# -----------------------------------------------------------------------------
# Shared helpers
# -----------------------------------------------------------------------------


def _black_image(width: int = 1024, height: int = 1024) -> torch.Tensor:
    return torch.zeros((1, height, width, 4), dtype=torch.float32)


def _black_like(image: torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor) and len(image.shape) >= 3:
        if len(image.shape) == 4:
            height = int(image.shape[1])
            width = int(image.shape[2])
        else:
            height = int(image.shape[0])
            width = int(image.shape[1])
        return _black_image(width=width, height=height)
    return _black_image()


def _log_soft_error(where: str, error: Exception) -> None:
    print(f"[KASKI Image API] {where}: {type(error).__name__}: {error}")
    traceback.print_exc()


def _collect_image_tensors(
    images: Input.Image | None,
) -> list[torch.Tensor]:
    """
    Accept one optional Comfy IMAGE input and normalize it to a list of
    single-image BHWC tensors.

    This node deliberately uses one IMAGE socket only. If that socket carries a
    batch tensor shaped (B, H, W, C), each batch entry becomes one reference
    image for the downstream API request. If no tensor is connected, the
    request runs without reference images.
    """
    if images is None:
        return []

    if len(images.shape) == 4:
        return [images[index : index + 1] for index in range(images.shape[0])]

    if len(images.shape) == 3:
        return [images.unsqueeze(0)]

    raise ValueError(
        "Reference images input must be a single HWC image or a batched "
        "BHWC tensor."
    )


# -----------------------------------------------------------------------------
# OpenAI helpers
# -----------------------------------------------------------------------------


async def _openai_validate_and_cast_response(
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

    # size="auto" can return slightly different image dimensions. A ComfyUI
    # batch needs one consistent tensor shape, so resize all results to the
    # first image exactly like the original rewrite did.
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


def _openai_price_image_1(
    response: OpenAIImageGenerationResponse,
) -> float | None:
    return (
        (response.usage.input_tokens * 10.0)
        + (response.usage.output_tokens * 40.0)
    ) / 1_000_000.0


def _openai_price_image_1_5(
    response: OpenAIImageGenerationResponse,
) -> float | None:
    return (
        (response.usage.input_tokens * 8.0)
        + (response.usage.output_tokens * 32.0)
    ) / 1_000_000.0


def _openai_price_image_2(
    response: OpenAIImageGenerationResponse,
) -> float | None:
    return (
        (response.usage.input_tokens * 8.0)
        + (response.usage.output_tokens * 30.0)
    ) / 1_000_000.0


def _openai_price_extractor(model_id: str):
    if model_id == "gpt-image-1":
        return _openai_price_image_1
    if model_id == "gpt-image-1.5":
        return _openai_price_image_1_5
    if model_id == "gpt-image-2":
        return _openai_price_image_2
    raise ValueError(f"Unknown OpenAI model: {model_id}")


def _validate_openai_custom_size(width: int, height: int) -> None:
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


def _resolve_openai_request_size(settings: dict[str, Any]) -> str:
    if settings["size"] != "Custom":
        return settings["size"]

    width = settings["custom_width"]
    height = settings["custom_height"]
    _validate_openai_custom_size(width, height)
    return f"{width}x{height}"


def _validate_openai_settings(settings: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "provider",
        "model_id",
        "quality",
        "background",
        "size",
        "custom_width",
        "custom_height",
        "n",
        "system_prompt",
    }
    missing_keys = required_keys.difference(settings)
    if missing_keys:
        raise ValueError(
            "OpenAI settings are incomplete. Missing: "
            + ", ".join(sorted(missing_keys))
        )

    if settings["provider"] != PROVIDER_OPENAI:
        raise ValueError(
            f"OpenAI adapter received provider '{settings['provider']}'."
        )

    model_id = settings["model_id"]
    quality = settings["quality"]
    background = settings["background"]
    size = settings["size"]
    custom_width = settings["custom_width"]
    custom_height = settings["custom_height"]
    n = settings["n"]

    if not isinstance(model_id, str) or model_id not in OPENAI_MODEL_IDS:
        raise ValueError(
            f"Invalid OpenAI model '{model_id}'. Allowed: "
            f"{sorted(OPENAI_MODEL_IDS)}"
        )

    if not isinstance(quality, str):
        raise TypeError("quality must be a string.")
    quality = quality.strip().lower()
    if quality not in OPENAI_QUALITIES:
        raise ValueError(
            f"Invalid quality '{quality}'. Allowed: "
            f"{sorted(OPENAI_QUALITIES)}"
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
        if background not in OPENAI_GPT_IMAGE_2_BACKGROUNDS:
            raise ValueError(
                "GPT Image 2 supports only auto or opaque backgrounds."
            )

        if size not in OPENAI_GPT_IMAGE_2_SIZES:
            raise ValueError(
                f"Invalid GPT Image 2 size '{size}'. Allowed: "
                f"{sorted(OPENAI_GPT_IMAGE_2_SIZES)}"
            )

        if size == "Custom":
            _validate_openai_custom_size(custom_width, custom_height)
    else:
        if background not in OPENAI_LEGACY_BACKGROUNDS:
            raise ValueError(
                f"Invalid legacy GPT Image background '{background}'."
            )

        if size not in OPENAI_LEGACY_SIZES:
            raise ValueError(
                f"Resolution '{size}' is only supported by GPT Image 2."
            )

        if custom_width is not None or custom_height is not None:
            raise ValueError(
                "custom_width and custom_height must be None for "
                f"{model_id}."
            )

    if not isinstance(settings["system_prompt"], str):
        raise TypeError("system_prompt must be a string.")

    normalized = dict(settings)
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


# -----------------------------------------------------------------------------
# Gemini helpers
# -----------------------------------------------------------------------------


def _gemini_mime_matches(
    mime: GeminiMimeType | str | None,
    pattern: str,
) -> bool:
    if mime is None:
        return False

    mime_value = mime.value if hasattr(mime, "value") else str(mime)
    return fnmatch(mime_value, pattern)


def _gemini_get_parts_by_type(
    response: GeminiGenerateContentResponse,
    part_type: Literal["text"] | str,
) -> list[GeminiPart]:
    if not response.candidates:
        if response.promptFeedback and response.promptFeedback.blockReason:
            feedback = response.promptFeedback
            raise ValueError(
                "Gemini API blocked the request. "
                f"Reason: {feedback.blockReason} ({feedback.blockReasonMessage})"
            )

        raise ValueError(
            "Gemini API returned no response candidates. "
            "Try IMAGE+TEXT to expose a possible model explanation."
        )

    parts: list[GeminiPart] = []
    blocked_reasons: list[str] = []

    for candidate in response.candidates:
        if (
            candidate.finishReason
            and candidate.finishReason.upper() == "IMAGE_PROHIBITED_CONTENT"
        ):
            blocked_reasons.append(candidate.finishReason)
            continue

        if candidate.content is None or candidate.content.parts is None:
            continue

        for part in candidate.content.parts:
            if part_type == "text" and part.text:
                parts.append(part)
            elif part.inlineData and _gemini_mime_matches(
                part.inlineData.mimeType,
                part_type,
            ):
                parts.append(part)
            elif part.fileData and _gemini_mime_matches(
                part.fileData.mimeType,
                part_type,
            ):
                parts.append(part)

    if not parts and blocked_reasons:
        raise ValueError(
            f"Gemini API blocked the request. Reasons: {blocked_reasons}"
        )

    return parts


def _gemini_get_text_from_response(
    response: GeminiGenerateContentResponse,
) -> str:
    return "\n".join(
        part.text
        for part in _gemini_get_parts_by_type(response, "text")
        if part.text
    )


async def _gemini_get_image_from_response(
    response: GeminiGenerateContentResponse,
    *,
    thought: bool = False,
) -> Input.Image:
    image_tensors: list[Input.Image] = []

    for part in _gemini_get_parts_by_type(response, "image/*"):
        if (part.thought is True) != thought:
            continue

        if part.inlineData and part.inlineData.data:
            image_data = base64.b64decode(part.inlineData.data)
            image_tensor = bytesio_to_image_tensor(BytesIO(image_data))
        elif part.fileData and part.fileData.fileUri:
            image_tensor = await download_url_to_image_tensor(
                part.fileData.fileUri
            )
        else:
            continue

        image_tensors.append(image_tensor)

    if image_tensors:
        return torch.cat(image_tensors, dim=0)

    if thought:
        return _black_image()

    model_message = _gemini_get_text_from_response(response).strip()
    if model_message:
        raise ValueError(
            "Gemini did not generate an image. "
            f"Model response: {model_message}"
        )

    raise ValueError(
        "Gemini did not generate an image. Rephrase the prompt or use "
        "IMAGE+TEXT to expose a possible model explanation."
    )


async def _gemini_create_image_parts(
    cls: type[IO.ComfyNode],
    images: Input.Image,
    image_limit: int = 0,
) -> list[GeminiPart]:
    if image_limit < 0:
        raise ValueError("image_limit must be greater than or equal to 0.")

    total_images = get_number_of_images(images)
    if total_images <= 0:
        raise ValueError("At least one reference image is required.")

    effective_max = (
        total_images
        if image_limit == 0
        else min(total_images, image_limit)
    )
    url_image_count = min(effective_max, GEMINI_URL_IMAGE_BUDGET)

    upload_kwargs: dict[str, Any] = {
        "wait_label": "Uploading reference images"
    }
    if effective_max > url_image_count:
        upload_kwargs = {
            "wait_label": f"Uploading reference images ({url_image_count}+)",
            "show_batch_index": False,
        }

    image_urls = await upload_images_to_comfyapi(
        cls,
        [images],
        max_images=url_image_count,
        **upload_kwargs,
    )

    parts: list[GeminiPart] = [
        GeminiPart(
            fileData=GeminiFileData(
                mimeType=GeminiMimeType.image_png,
                fileUri=image_url,
            )
        )
        for image_url in image_urls
    ]

    if effective_max > url_image_count:
        flattened_images = _collect_image_tensors(images)

        for index in range(url_image_count, effective_max):
            parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.image_png,
                        data=tensor_to_base64_string(
                            flattened_images[index].squeeze(0)
                        ),
                    )
                )
            )

    return parts


def _gemini_price_extractor(
    response: GeminiGenerateContentResponse,
) -> float | None:
    if not response.modelVersion or not response.usageMetadata:
        return None

    prices = {
        "gemini-3-pro-image-preview": (2.0, 12.0, 120.0),
        "gemini-3-pro-image": (2.0, 12.0, 120.0),
        "gemini-3.1-flash-image-preview": (0.5, 3.0, 60.0),
        "gemini-3.1-flash-image": (0.5, 3.0, 60.0),
        "gemini-3.1-flash-lite-image": (0.25, 1.5, 30.0),
    }

    model_prices = prices.get(response.modelVersion)
    if model_prices is None:
        return None

    input_price, output_text_price, output_image_price = model_prices
    usage = response.usageMetadata

    total_price = (usage.promptTokenCount or 0) * input_price

    if usage.candidatesTokensDetails:
        for token_detail in usage.candidatesTokensDetails:
            if token_detail.modality == Modality.IMAGE:
                total_price += (
                    token_detail.tokenCount or 0
                ) * output_image_price
            else:
                total_price += (
                    token_detail.tokenCount or 0
                ) * output_text_price

    if usage.thoughtsTokenCount:
        total_price += usage.thoughtsTokenCount * output_text_price

    return total_price / 1_000_000.0


def _validate_gemini_settings(settings: dict[str, Any]) -> dict[str, Any]:
    required_keys = {
        "provider",
        "model_id",
        "model_label",
        "aspect_ratio",
        "resolution",
        "response_modalities",
        "temperature",
        "top_p",
        "system_prompt",
        "thinking_level",
    }
    missing_keys = required_keys.difference(settings)
    if missing_keys:
        raise ValueError(
            "Gemini settings are incomplete. Missing: "
            + ", ".join(sorted(missing_keys))
        )

    if settings["provider"] != PROVIDER_GEMINI:
        raise ValueError(
            f"Gemini adapter received provider '{settings['provider']}'."
        )

    model_id = settings["model_id"]
    model_label = settings["model_label"]
    aspect_ratio = settings["aspect_ratio"]
    resolution = settings["resolution"]
    modalities = settings["response_modalities"]
    thinking_level = settings["thinking_level"]
    temperature = settings["temperature"]
    top_p = settings["top_p"]
    system_prompt = settings["system_prompt"]

    if model_label not in GEMINI_MODEL_IDS:
        raise ValueError(f"Invalid Gemini model label '{model_label}'.")

    if model_id != GEMINI_MODEL_IDS[model_label]:
        raise ValueError(
            "Gemini model label and model ID in the settings object do not match."
        )

    valid_ratios = (
        GEMINI_BASE_ASPECT_RATIOS
        if model_id == "gemini-3-pro-image-preview"
        else GEMINI_EXTENDED_ASPECT_RATIOS
    )
    if aspect_ratio not in valid_ratios:
        raise ValueError(
            f"Aspect ratio '{aspect_ratio}' is not valid for {model_label}."
        )

    valid_resolutions = (
        {"1K"}
        if model_id == "gemini-3.1-flash-lite-image"
        else {"1K", "2K", "4K"}
    )
    if resolution not in valid_resolutions:
        raise ValueError(
            f"Resolution '{resolution}' is not valid for {model_label}."
        )

    if modalities not in GEMINI_VALID_MODALITIES:
        raise ValueError(
            f"Invalid response modality '{modalities}'."
        )

    if thinking_level is not None:
        if model_id == "gemini-3-pro-image-preview":
            raise ValueError(
                "Gemini 3 Pro Image does not use thinking_level in this node."
            )
        if thinking_level not in GEMINI_VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking level '{thinking_level}'."
            )

    if (
        not isinstance(temperature, (int, float))
        or not 0.0 <= temperature <= 2.0
    ):
        raise ValueError("temperature must be between 0.0 and 2.0.")

    if not isinstance(top_p, (int, float)) or not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0.")

    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string.")

    return dict(settings)


# -----------------------------------------------------------------------------
# Unified settings validation / defaults
# -----------------------------------------------------------------------------


def _validate_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError(
            "settings must come from the KASKI Image API Settings node."
        )

    provider = settings.get("provider")
    if provider == PROVIDER_OPENAI:
        return _validate_openai_settings(settings)
    if provider == PROVIDER_GEMINI:
        return _validate_gemini_settings(settings)

    raise ValueError(
        f"Unknown image provider '{provider}'. Expected '{PROVIDER_OPENAI}' "
        f"or '{PROVIDER_GEMINI}'."
    )


def _default_settings() -> dict[str, Any]:
    return {
        "provider": PROVIDER_OPENAI,
        "model_id": "gpt-image-2",
        "quality": "low",
        "background": "auto",
        "size": "auto",
        "custom_width": 1024,
        "custom_height": 1024,
        "n": 1,
        # Deliberately carried through the unified settings object even though
        # the OpenAI adapter does not send it to the GPT Image endpoint.
        "system_prompt": GEMINI_IMAGE_SYS_PROMPT,
    }


# -----------------------------------------------------------------------------
# Settings UI builders
# -----------------------------------------------------------------------------


def _openai_legacy_model_inputs() -> list[Input]:
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


def _openai_model_selector() -> Input:
    return IO.DynamicCombo.Input(
        "model_settings",
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
                            "Image size. Select Custom to use custom width "
                            "and height."
                        ),
                    ),
                    IO.Int.Input(
                        "custom_width",
                        default=1024,
                        min=1024,
                        max=3840,
                        step=16,
                        tooltip=(
                            "Used only when size is Custom. Must be a "
                            "multiple of 16."
                        ),
                    ),
                    IO.Int.Input(
                        "custom_height",
                        default=1024,
                        min=1024,
                        max=3840,
                        step=16,
                        tooltip=(
                            "Used only when size is Custom. Must be a "
                            "multiple of 16."
                        ),
                    ),
                    IO.Combo.Input(
                        "background",
                        default="auto",
                        options=["auto", "opaque"],
                        tooltip=(
                            "GPT Image 2 does not support transparent output."
                        ),
                    ),
                    IO.Combo.Input(
                        "quality",
                        default="low",
                        options=["low", "medium", "high"],
                        tooltip=(
                            "Image quality, affecting cost and generation time."
                        ),
                    ),
                ],
            ),
            IO.DynamicCombo.Option(
                "gpt-image-1.5",
                _openai_legacy_model_inputs(),
            ),
            IO.DynamicCombo.Option(
                "gpt-image-1",
                _openai_legacy_model_inputs(),
            ),
        ],
        tooltip="OpenAI GPT Image model and model-specific settings.",
    )


def _gemini_model_inputs(
    *,
    aspect_ratios: list[str],
    resolutions: list[str],
    supports_thinking: bool,
) -> list[Input]:
    inputs: list[Input] = [
        IO.Combo.Input(
            "aspect_ratio",
            options=aspect_ratios,
            default="auto",
            tooltip=(
                "Output aspect ratio. 'auto' lets Gemini infer it from the "
                "input or prompt."
            ),
        ),
        IO.Combo.Input(
            "resolution",
            options=resolutions,
            default=resolutions[0],
            tooltip="Target output resolution.",
        ),
    ]

    if supports_thinking:
        inputs.append(
            IO.Combo.Input(
                "thinking_level",
                options=["MINIMAL", "HIGH"],
                default="MINIMAL",
                tooltip=(
                    "HIGH can improve difficult generations but may cost "
                    "more and take longer."
                ),
            )
        )

    return inputs


def _gemini_model_selector() -> Input:
    return IO.DynamicCombo.Input(
        "model_settings",
        options=[
            IO.DynamicCombo.Option(
                "Gemini 3 Pro Image",
                _gemini_model_inputs(
                    aspect_ratios=GEMINI_BASE_ASPECT_RATIOS,
                    resolutions=["1K", "2K", "4K"],
                    supports_thinking=False,
                ),
            ),
            IO.DynamicCombo.Option(
                "Nano Banana 2 (Gemini 3.1 Flash Image)",
                _gemini_model_inputs(
                    aspect_ratios=GEMINI_EXTENDED_ASPECT_RATIOS,
                    resolutions=["1K", "2K", "4K"],
                    supports_thinking=True,
                ),
            ),
            IO.DynamicCombo.Option(
                "Nano Banana 2 Lite",
                _gemini_model_inputs(
                    aspect_ratios=GEMINI_EXTENDED_ASPECT_RATIOS,
                    resolutions=["1K"],
                    supports_thinking=True,
                ),
            ),
        ],
        tooltip="Gemini / Nanobanana model and model-specific settings.",
    )


# -----------------------------------------------------------------------------
# Unified settings node
# -----------------------------------------------------------------------------


class KASKIImageAPISettings(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageAPISettings_KASKI",
            display_name="KASKI Image API Settings",
            category=CATEGORY,
            description=(
                "One shared settings node for OpenAI GPT Image and Gemini / "
                "Nanobanana. Choose the provider and model here, then fan this "
                "settings output out to one or more KASKI Image API Generator "
                "nodes."
            ),
            inputs=[
                IO.DynamicCombo.Input(
                    "backend",
                    options=[
                        IO.DynamicCombo.Option(
                            OPENAI_PROVIDER_LABEL,
                            [
                                _openai_model_selector(),
                                IO.Int.Input(
                                    "n",
                                    default=1,
                                    min=1,
                                    max=8,
                                    step=1,
                                    tooltip=(
                                        "How many images OpenAI should generate "
                                        "per request."
                                    ),
                                    display_mode=IO.NumberDisplay.number,
                                ),
                            ],
                        ),
                        IO.DynamicCombo.Option(
                            GEMINI_PROVIDER_LABEL,
                            [
                                _gemini_model_selector(),
                                IO.Combo.Input(
                                    "response_modalities",
                                    options=["IMAGE", "IMAGE+TEXT"],
                                    default="IMAGE",
                                    tooltip=(
                                        "IMAGE returns only generated images. "
                                        "IMAGE+TEXT also returns the model's "
                                        "text response."
                                    ),
                                ),
                                IO.Float.Input(
                                    "temperature",
                                    default=1.0,
                                    min=0.0,
                                    max=2.0,
                                    step=0.01,
                                    tooltip=(
                                        "Controls generation randomness. Lower "
                                        "is more focused; higher is more variable."
                                    ),
                                    advanced=True,
                                ),
                                IO.Float.Input(
                                    "top_p",
                                    default=0.95,
                                    min=0.0,
                                    max=1.0,
                                    step=0.01,
                                    tooltip="Nucleus sampling threshold.",
                                    advanced=True,
                                ),
                            ],
                        ),
                    ],
                    tooltip=(
                        "Select the image API backend. Only settings relevant "
                        "to that backend are shown."
                    ),
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    tooltip=(
                        "Shared system prompt. Gemini sends it as systemInstruction. "
                        "OpenAI keeps it in the shared settings object but does "
                        "not send it to the GPT Image endpoint."
                    ),
                    advanced=True,
                ),
            ],
            outputs=[
                IO.Custom(SETTINGS_TYPE).Output(display_name="settings"),
            ],
        )

    @classmethod
    def execute(
        cls,
        backend: dict[str, Any],
        system_prompt: str,
    ) -> IO.NodeOutput:
        try:
            provider_label = backend["backend"]
            model_settings = backend["model_settings"]

            if provider_label == OPENAI_PROVIDER_LABEL:
                model_id = model_settings["model_settings"]
                is_gpt_image_2 = model_id == "gpt-image-2"

                settings = {
                    "provider": PROVIDER_OPENAI,
                    "model_id": model_id,
                    "quality": model_settings["quality"],
                    "background": model_settings["background"],
                    "size": model_settings["size"],
                    "custom_width": (
                        int(model_settings.get("custom_width", 1024))
                        if is_gpt_image_2
                        else None
                    ),
                    "custom_height": (
                        int(model_settings.get("custom_height", 1024))
                        if is_gpt_image_2
                        else None
                    ),
                    "n": int(backend["n"]),
                    # Intentionally retained for a stable unified settings
                    # payload, but ignored by the OpenAI request adapter.
                    "system_prompt": system_prompt,
                }
                return IO.NodeOutput(_validate_settings(settings))

            if provider_label == GEMINI_PROVIDER_LABEL:
                model_label = model_settings["model_settings"]
                model_id = GEMINI_MODEL_IDS[model_label]

                settings = {
                    "provider": PROVIDER_GEMINI,
                    "model_label": model_label,
                    "model_id": model_id,
                    "aspect_ratio": model_settings["aspect_ratio"],
                    "resolution": model_settings["resolution"],
                    "response_modalities": backend["response_modalities"],
                    "thinking_level": model_settings.get("thinking_level"),
                    "temperature": float(backend["temperature"]),
                    "top_p": float(backend["top_p"]),
                    "system_prompt": system_prompt,
                }
                return IO.NodeOutput(_validate_settings(settings))

            raise ValueError(f"Unknown backend label '{provider_label}'.")

        except Exception as error:
            _log_soft_error("KASKIImageAPISettings.execute", error)
            return IO.NodeOutput(_default_settings())


# -----------------------------------------------------------------------------
# Provider adapters
# -----------------------------------------------------------------------------


async def _execute_openai(
    cls: type[IO.ComfyNode],
    *,
    prompt: str,
    settings: dict[str, Any],
    images: Input.Image | None,
    mask: Input.Image | None,
) -> tuple[torch.Tensor, str, torch.Tensor]:
    validate_string(prompt, strip_whitespace=False)
    settings = _validate_openai_settings(settings)

    model_id = settings["model_id"]
    quality = settings["quality"]
    background = settings["background"]
    n = settings["n"]
    size = _resolve_openai_request_size(settings)
    price_extractor = _openai_price_extractor(model_id)

    flat_images = _collect_image_tensors(images)
    if len(flat_images) > MAX_OPENAI_REFERENCE_IMAGES:
        raise ValueError(
            f"GPT Image supports at most {MAX_OPENAI_REFERENCE_IMAGES} "
            f"reference images; received {len(flat_images)}."
        )

    if mask is not None and not flat_images:
        raise ValueError("Cannot use a mask without an input image.")

    if flat_images:
        files = []

        for index, single_image in enumerate(flat_images):
            scaled_image = downscale_image_tensor_compat(single_image)

            image_np = (scaled_image.numpy() * 255).astype(np.uint8)
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
                raise ValueError("Cannot use a mask with multiple images.")

            reference_image = flat_images[0]
            if mask.shape[1:] != reference_image.shape[1:-1]:
                raise ValueError("Mask and image must have the same size.")

            _, height, width = mask.shape
            rgba_mask = torch.zeros(
                height,
                width,
                4,
                device="cpu",
            )
            rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

            scaled_mask = downscale_image_tensor_compat(
                rgba_mask.unsqueeze(0)
            )

            mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
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
        # system_prompt intentionally does NOT get forwarded here. It exists in
        # the unified settings node purely so the settings contract is stable
        # when switching providers.
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

    image = await _openai_validate_and_cast_response(response)
    return image, "", _black_like(image)


def downscale_image_tensor_compat(image: torch.Tensor) -> torch.Tensor:
    """Keep the original OpenAI rewrite's 2048² upload downscale behavior."""
    # Imported lazily here only to keep all provider-specific behavior together.
    from comfy_api_nodes.util import downscale_image_tensor

    return downscale_image_tensor(
        image,
        total_pixels=2048 * 2048,
    ).squeeze()


async def _execute_gemini(
    cls: type[IO.ComfyNode],
    *,
    prompt: str,
    settings: dict[str, Any],
    images: Input.Image | None,
    files: list[GeminiPart] | None,
) -> tuple[torch.Tensor, str, torch.Tensor]:
    validate_string(prompt, strip_whitespace=True, min_length=1)
    settings = _validate_gemini_settings(settings)

    flat_images = _collect_image_tensors(images)
    total_images = len(flat_images)

    if total_images > MAX_GEMINI_REFERENCE_IMAGES:
        raise ValueError(
            "The current maximum number of supported Gemini reference images "
            f"is {MAX_GEMINI_REFERENCE_IMAGES}; received {total_images}."
        )

    parts: list[GeminiPart] = [GeminiPart(text=prompt)]

    if images is not None and total_images > 0:
        parts.extend(
            await _gemini_create_image_parts(
                cls,
                images,
                image_limit=MAX_GEMINI_REFERENCE_IMAGES,
            )
        )

    if files is not None:
        parts.extend(files)

    image_config = GeminiImageConfig(
        imageSize=settings["resolution"]
    )
    if settings["aspect_ratio"] != "auto":
        image_config.aspectRatio = settings["aspect_ratio"]

    system_instruction = None
    if settings["system_prompt"].strip():
        system_instruction = GeminiSystemInstructionContent(
            parts=[
                GeminiTextPart(text=settings["system_prompt"])
            ],
            role=None,
        )

    generation_config_kwargs: dict[str, Any] = {
        "responseModalities": (
            ["IMAGE"]
            if settings["response_modalities"] == "IMAGE"
            else ["TEXT", "IMAGE"]
        ),
        "imageConfig": image_config,
        "temperature": settings["temperature"],
        "topP": settings["top_p"],
    }

    if settings["thinking_level"] is not None:
        generation_config_kwargs["thinkingConfig"] = (
            GeminiThinkingConfig(
                thinkingLevel=settings["thinking_level"]
            )
        )

    response = await sync_op(
        cls,
        ApiEndpoint(
            path=(
                f"{GEMINI_BASE_ENDPOINT}/"
                f"{settings['model_id']}"
            ),
            method="POST",
        ),
        data=GeminiImageGenerateContentRequest(
            contents=[
                GeminiContent(
                    role=GeminiRole.user,
                    parts=parts,
                )
            ],
            generationConfig=GeminiImageGenerationConfig(
                **generation_config_kwargs
            ),
            systemInstruction=system_instruction,
        ),
        response_model=GeminiGenerateContentResponse,
        price_extractor=_gemini_price_extractor,
    )

    return (
        await _gemini_get_image_from_response(response),
        _gemini_get_text_from_response(response),
        await _gemini_get_image_from_response(response, thought=True),
    )


# -----------------------------------------------------------------------------
# Unified generator node
# -----------------------------------------------------------------------------


class KASKIImageAPIGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="ImageAPIGenerator_KASKI",
            display_name="KASKI Image API Generator",
            category=CATEGORY,
            description=(
                "Unified image generator. The connected KASKI Image API Settings "
                "node decides whether this request is sent through OpenAI GPT "
                "Image or Gemini / Nanobanana."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Image generation / edit prompt.",
                ),
                IO.Custom(SETTINGS_TYPE).Input(
                    "settings",
                    tooltip=(
                        "Connect a KASKI Image API Settings node. Switching the "
                        "provider there switches this generator backend."
                    ),
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=0x7FFFFFFFFFFFFFFF,
                    step=1,
                    control_after_generate=True,
                    display_mode=IO.NumberDisplay.number,
                    tooltip=(
                        "ComfyUI cache-buster / control-after-generate value. "
                        "It is not sent to either image API by the current "
                        "request schemas."
                    ),
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional reference images via exactly one IMAGE socket. "
                        "Pass a single image or a batched tensor shaped "
                        "(B, H, W, C). Each batch entry is sent as its own "
                        "reference image. OpenAI supports up to 16 references; "
                        "Gemini currently supports up to 14."
                    ),
                ),
                IO.Mask.Input(
                    "mask",
                    optional=True,
                    tooltip=(
                        "Optional OpenAI inpainting mask. Ignored when the "
                        "selected backend is Gemini. White areas are replaced; "
                        "requires exactly one OpenAI reference image."
                    ),
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip=(
                        "Optional Gemini input files from a compatible file node. "
                        "Ignored when the selected backend is OpenAI."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.String.Output(display_name="text"),
                IO.Image.Output(
                    display_name="thought_image",
                    tooltip=(
                        "Gemini thinking-process image when available. OpenAI "
                        "returns a black placeholder on this output."
                    ),
                ),
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
        images: Input.Image | None = None,
        mask: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
    ) -> IO.NodeOutput:
        # Both upstream rewrites use seed only to invalidate ComfyUI's cache.
        del seed

        try:
            settings = _validate_settings(settings)
            provider = settings["provider"]

            if provider == PROVIDER_OPENAI:
                image, text, thought_image = await _execute_openai(
                    cls,
                    prompt=prompt,
                    settings=settings,
                    images=images,
                    mask=mask,
                )
                return IO.NodeOutput(image, text, thought_image)

            if provider == PROVIDER_GEMINI:
                image, text, thought_image = await _execute_gemini(
                    cls,
                    prompt=prompt,
                    settings=settings,
                    images=images,
                    files=files,
                )
                return IO.NodeOutput(image, text, thought_image)

            raise ValueError(f"Unsupported provider '{provider}'.")

        except Exception as error:
            _log_soft_error("KASKIImageAPIGenerator.execute", error)
            black = _black_image()
            return IO.NodeOutput(black, "", black)


# -----------------------------------------------------------------------------
# ComfyUI registration
# -----------------------------------------------------------------------------

UNIFIED_IMAGE_API_NODE_CLASS_MAPPINGS = {
    "ImageAPISettings_KASKI": KASKIImageAPISettings,
    "ImageAPIGenerator_KASKI": KASKIImageAPIGenerator,
}

UNIFIED_IMAGE_API_NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageAPISettings_KASKI": "KASKI Image API Settings",
    "ImageAPIGenerator_KASKI": "KASKI Image API Generator",
}
