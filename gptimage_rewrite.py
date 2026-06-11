import base64
from io import BytesIO

import numpy as np
import torch
from PIL import Image

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

VALID_GPT_IMAGE_BACKGROUNDS = {
    "auto",
    "opaque",
    "transparent",
}

VALID_GPT_IMAGE_SIZES = {
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "2048x2048",
    "2048x1152",
    "1152x2048",
    "3840x2160",
    "2160x3840",
}

VALID_GPT_IMAGE_1_SIZES = {
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
            await download_url_to_bytesio(img_data.url, img_io, timeout=timeout)
        else:
            raise ValueError("Invalid image payload – neither URL nor base64 data present.")

        pil_img = Image.open(img_io).convert("RGBA")
        arr = np.asarray(pil_img).astype(np.float32) / 255.0
        image_tensors.append(torch.from_numpy(arr))

    return torch.stack(image_tensors, dim=0)


def calculate_tokens_price_image_1(response: OpenAIImageGenerationResponse) -> float | None:
    return ((response.usage.input_tokens * 10.0) + (response.usage.output_tokens * 40.0)) / 1_000_000.0


def calculate_tokens_price_image_1_5(response: OpenAIImageGenerationResponse) -> float | None:
    return ((response.usage.input_tokens * 8.0) + (response.usage.output_tokens * 32.0)) / 1_000_000.0


def calculate_tokens_price_image_2_0(response: OpenAIImageGenerationResponse) -> float | None:
    return ((response.usage.input_tokens * 8.0) + (response.usage.output_tokens * 30.0)) / 1_000_000.0


def normalize_gpt_image_model(model: str) -> str:
    if not isinstance(model, str):
        raise TypeError("model must be a string")

    model = model.strip()

    if not model:
        raise ValueError("model must not be empty")

    if model not in VALID_GPT_IMAGE_MODELS:
        raise ValueError(f"Invalid model '{model}'. Allowed: {VALID_GPT_IMAGE_MODELS}")

    return model


def normalize_gpt_image_quality(quality: str) -> str:
    if not isinstance(quality, str):
        raise TypeError("quality must be a string")

    quality = quality.strip().lower()

    if not quality:
        raise ValueError("quality must not be empty")

    if quality not in VALID_GPT_IMAGE_QUALITIES:
        raise ValueError(f"Invalid quality '{quality}'. Allowed: {VALID_GPT_IMAGE_QUALITIES}")

    return quality


def normalize_gpt_image_background(background: str) -> str:
    if not isinstance(background, str):
        raise TypeError("background must be a string")

    background = background.strip().lower()

    if not background:
        raise ValueError("background must not be empty")

    if background not in VALID_GPT_IMAGE_BACKGROUNDS:
        raise ValueError(f"Invalid background '{background}'. Allowed: {VALID_GPT_IMAGE_BACKGROUNDS}")

    return background


def normalize_gpt_image_size(size: str) -> str:
    if not isinstance(size, str):
        raise TypeError("size must be a string")

    size = size.strip().lower()

    if not size:
        raise ValueError("size must not be empty")

    if size not in VALID_GPT_IMAGE_SIZES:
        raise ValueError(f"Invalid size '{size}'. Allowed: {VALID_GPT_IMAGE_SIZES}")

    return size


class OpenAIGPTImage1(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="OpenAIGPTImage1_KASKI",
            display_name="OpenAI GPT Image IO-unlocked",
            category="KASKI/api-adaptions/openai",
            description="Generate or edit images synchronously via OpenAI's GPT Image endpoint.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    default="",
                    multiline=True,
                    tooltip="Text prompt for GPT Image.",
                ),
                IO.Int.Input(
                    "seed",
                    default=0,
                    min=0,
                    max=2**31 - 1,
                    step=1,
                    display_mode=IO.NumberDisplay.number,
                    control_after_generate=True,
                    tooltip="Not implemented yet in backend.",
                    optional=True,
                ),
                IO.String.Input(
                    "quality",
                    default="low",
                    tooltip="Image quality: low, medium, high.",
                    optional=True,
                ),
                IO.String.Input(
                    "background",
                    default="opaque",
                    tooltip="Background mode: auto, opaque, transparent.",
                    optional=True,
                ),
                IO.String.Input(
                    "size",
                    default="1024x1024",
                    tooltip="Image size, e.g. auto, 1024x1024, 1536x1024, 2048x2048, 3840x2160.",
                    optional=True,
                ),
                IO.Int.Input(
                    "n",
                    default=1,
                    min=1,
                    max=8,
                    step=1,
                    tooltip="How many images to generate.",
                    display_mode=IO.NumberDisplay.number,
                    optional=True,
                ),
                IO.Image.Input(
                    "image",
                    tooltip="Optional reference image for image editing.",
                    optional=True,
                ),
                IO.Mask.Input(
                    "mask",
                    tooltip="Optional mask for inpainting. White areas will be replaced.",
                    optional=True,
                ),
                IO.String.Input(
                    "model",
                    default="gpt-image-2",
                    tooltip="Model: gpt-image-1, gpt-image-1.5, gpt-image-2.",
                    optional=True,
                ),
            ],
            outputs=[
                IO.Image.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["quality", "n", "model"]),
                expr="""
                (
                  $ranges := {
                    "gpt-image-1": {
                      "low":    [0.011, 0.02],
                      "medium": [0.042, 0.07],
                      "high":   [0.167, 0.25]
                    },
                    "gpt-image-1.5": {
                      "low":    [0.009, 0.02],
                      "medium": [0.034, 0.062],
                      "high":   [0.133, 0.22]
                    },
                    "gpt-image-2": {
                      "low":    [0.0048, 0.012],
                      "medium": [0.041, 0.112],
                      "high":   [0.165, 0.43]
                    }
                  };
                  $range := $lookup($lookup($ranges, widgets.model), widgets.quality);
                  $nRaw := widgets.n;
                  $n := ($nRaw != null and $nRaw != 0) ? $nRaw : 1;
                  ($n = 1)
                    ? {"type":"range_usd","min_usd": $range[0], "max_usd": $range[1], "format": {"approximate": true}}
                    : {
                        "type":"range_usd",
                        "min_usd": $range[0] * $n,
                        "max_usd": $range[1] * $n,
                        "format": { "suffix": "/Run", "approximate": true }
                      }
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        seed: int = 0,
        quality: str = "low",
        background: str = "opaque",
        size: str = "1024x1024",
        n: int = 1,
        image: Input.Image | None = None,
        mask: Input.Image | None = None,
        model: str = "gpt-image-2",
    ) -> IO.NodeOutput:
        validate_string(prompt, strip_whitespace=True, min_length=1)

        model = normalize_gpt_image_model(model)
        quality = normalize_gpt_image_quality(quality)
        background = normalize_gpt_image_background(background)
        size = normalize_gpt_image_size(size)

        if mask is not None and image is None:
            raise ValueError("Cannot use a mask without an input image")

        if model in ("gpt-image-1", "gpt-image-1.5"):
            if size not in VALID_GPT_IMAGE_1_SIZES:
                raise ValueError(f"Resolution '{size}' is only supported by GPT Image 2 model")

        if model == "gpt-image-1":
            price_extractor = calculate_tokens_price_image_1
        elif model == "gpt-image-1.5":
            price_extractor = calculate_tokens_price_image_1_5
        elif model == "gpt-image-2":
            price_extractor = calculate_tokens_price_image_2_0
            if background == "transparent":
                raise ValueError("Transparent background is not supported for GPT Image 2 model")
        else:
            raise ValueError(f"Unknown model: {model}")

        if image is not None:
            files = []
            batch_size = image.shape[0]

            for i in range(batch_size):
                single_image = image[i : i + 1]
                scaled_image = downscale_image_tensor(
                    single_image,
                    total_pixels=2048 * 2048,
                ).squeeze()

                image_np = (scaled_image.numpy() * 255).astype(np.uint8)
                img = Image.fromarray(image_np)

                img_byte_arr = BytesIO()
                img.save(img_byte_arr, format="PNG")
                img_byte_arr.seek(0)

                if batch_size == 1:
                    files.append(("image", (f"image_{i}.png", img_byte_arr, "image/png")))
                else:
                    files.append(("image[]", (f"image_{i}.png", img_byte_arr, "image/png")))

            if mask is not None:
                if image.shape[0] != 1:
                    raise Exception("Cannot use a mask with multiple image")

                if mask.shape[1:] != image.shape[1:-1]:
                    raise Exception("Mask and Image must be the same size")

                _, height, width = mask.shape

                rgba_mask = torch.zeros(height, width, 4, device="cpu")
                rgba_mask[:, :, 3] = 1 - mask.squeeze().cpu()

                scaled_mask = downscale_image_tensor(
                    rgba_mask.unsqueeze(0),
                    total_pixels=2048 * 2048,
                ).squeeze()

                mask_np = (scaled_mask.numpy() * 255).astype(np.uint8)
                mask_img = Image.fromarray(mask_np)

                mask_img_byte_arr = BytesIO()
                mask_img.save(mask_img_byte_arr, format="PNG")
                mask_img_byte_arr.seek(0)

                files.append(("mask", ("mask.png", mask_img_byte_arr, "image/png")))

            response = await sync_op(
                cls,
                ApiEndpoint(path="/proxy/openai/images/edits", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                data=OpenAIImageEditRequest(
                    model=model,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    seed=seed,
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
                ApiEndpoint(path="/proxy/openai/images/generations", method="POST"),
                response_model=OpenAIImageGenerationResponse,
                data=OpenAIImageGenerationRequest(
                    model=model,
                    prompt=prompt,
                    quality=quality,
                    background=background,
                    n=n,
                    seed=seed,
                    size=size,
                    moderation="low",
                ),
                price_extractor=price_extractor,
            )

        return IO.NodeOutput(await validate_and_cast_response(response))


class OpenAIGPTImageSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gpt-image-1",
                    "gpt-image-1.5",
                    "gpt-image-2",
                ], {
                    "default": "gpt-image-2",
                }),
                "quality": ([
                    "low",
                    "medium",
                    "high",
                ], {
                    "default": "low",
                }),
                "background": ([
                    "auto",
                    "opaque",
                    "transparent",
                ], {
                    "default": "opaque",
                }),
                "size": ([
                    "auto",
                    "1024x1024",
                    "1024x1536",
                    "1536x1024",
                    "2048x2048",
                    "2048x1152",
                    "1152x2048",
                    "3840x2160",
                    "2160x3840",
                ], {
                    "default": "1024x1024",
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model", "quality", "background", "size")
    FUNCTION = "get_settings"
    CATEGORY = "KASKI/api-adaptions/openai"

    def get_settings(
        self,
        model: str,
        quality: str,
        background: str,
        size: str,
    ):
        model = normalize_gpt_image_model(model)
        quality = normalize_gpt_image_quality(quality)
        background = normalize_gpt_image_background(background)
        size = normalize_gpt_image_size(size)

        if model in ("gpt-image-1", "gpt-image-1.5"):
            if size not in VALID_GPT_IMAGE_1_SIZES:
                raise ValueError(f"Resolution '{size}' is only supported by GPT Image 2 model")

        if model == "gpt-image-2" and background == "transparent":
            raise ValueError("Transparent background is not supported for GPT Image 2 model")

        return (model, quality, background, size)


OPENAI_GPT_IMAGE_REWRITE_NODE_CLASS_MAPPINGS = {
    "OpenAIGPTImage1_KASKI": OpenAIGPTImage1,
    "OpenAIGPTImageSettings_KASKI": OpenAIGPTImageSettings,
}


OPENAI_GPT_IMAGE_REWRITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "OpenAIGPTImage1_KASKI": "OpenAI GPT Image IO-unlocked",
    "OpenAIGPTImageSettings_KASKI": "OpenAI GPT Image Settings",
}