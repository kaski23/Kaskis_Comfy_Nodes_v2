import base64
import traceback
from fnmatch import fnmatch
from io import BytesIO
from typing import Any, Literal

import torch

from comfy_api.latest import IO, Input
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
    download_url_to_image_tensor,
    get_number_of_images,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    validate_string,
)


CATEGORY = "KASKI/api-adaptions/nanobanana"
SETTINGS_TYPE = "KASKI_GEMINI_IMAGE_SETTINGS"
GEMINI_BASE_ENDPOINT = "/proxy/vertexai/gemini"
MAX_REFERENCE_IMAGES = 14
URL_IMAGE_BUDGET = 10

GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of format, intent, or abstraction—as literal "
    "visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, you must creatively "
    "invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or "
    "conversational requests."
)

MODEL_IDS = {
    "Gemini 3 Pro Image": "gemini-3-pro-image-preview",
    "Nano Banana 2 (Gemini 3.1 Flash Image)": "gemini-3.1-flash-image-preview",
    "Nano Banana 2 Lite": "gemini-3.1-flash-lite-image",
}

BASE_ASPECT_RATIOS = [
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

EXTENDED_ASPECT_RATIOS = BASE_ASPECT_RATIOS + [
    "1:4",
    "4:1",
    "8:1",
    "1:8",
]

VALID_MODALITIES = {"IMAGE", "IMAGE+TEXT"}
VALID_THINKING_LEVELS = {"MINIMAL", "HIGH"}


def _model_settings_inputs(
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
            tooltip="Output aspect ratio. 'auto' lets Gemini infer it from the input or prompt.",
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
                tooltip="HIGH can improve difficult generations but may cost more and take longer.",
            )
        )

    return inputs


def _black_image(width: int = 1024, height: int = 1024) -> torch.Tensor:
    return torch.zeros((1, height, width, 4), dtype=torch.float32)


def _log_soft_error(where: str, error: Exception) -> None:
    print(f"[KASKI GeminiImage2] {where}: {type(error).__name__}: {error}")
    traceback.print_exc()


def _mime_matches(mime: GeminiMimeType | str | None, pattern: str) -> bool:
    if mime is None:
        return False

    mime_value = mime.value if hasattr(mime, "value") else str(mime)
    return fnmatch(mime_value, pattern)


def get_parts_by_type(
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
            elif part.inlineData and _mime_matches(
                part.inlineData.mimeType,
                part_type,
            ):
                parts.append(part)
            elif part.fileData and _mime_matches(
                part.fileData.mimeType,
                part_type,
            ):
                parts.append(part)

    if not parts and blocked_reasons:
        raise ValueError(
            f"Gemini API blocked the request. Reasons: {blocked_reasons}"
        )

    return parts


def get_text_from_response(response: GeminiGenerateContentResponse) -> str:
    return "\n".join(
        part.text
        for part in get_parts_by_type(response, "text")
        if part.text
    )


async def get_image_from_response(
    response: GeminiGenerateContentResponse,
    *,
    thought: bool = False,
) -> Input.Image:
    image_tensors: list[Input.Image] = []

    for part in get_parts_by_type(response, "image/*"):
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

    model_message = get_text_from_response(response).strip()
    if model_message:
        raise ValueError(
            "Gemini did not generate an image. "
            f"Model response: {model_message}"
        )

    raise ValueError(
        "Gemini did not generate an image. Rephrase the prompt or use "
        "IMAGE+TEXT to expose a possible model explanation."
    )


def _flatten_images(images: list[Input.Image]) -> list[torch.Tensor]:
    frames: list[torch.Tensor] = []

    for tensor in images:
        if len(tensor.shape) == 4:
            frames.extend(tensor[index] for index in range(tensor.shape[0]))
        else:
            frames.append(tensor)

    return frames


async def create_image_parts(
    cls: type[IO.ComfyNode],
    images: Input.Image | list[Input.Image],
    image_limit: int = 0,
) -> list[GeminiPart]:
    if image_limit < 0:
        raise ValueError("image_limit must be greater than or equal to 0.")

    image_list = images if isinstance(images, list) else [images]
    total_images = sum(get_number_of_images(image) for image in image_list)

    if total_images <= 0:
        raise ValueError("At least one reference image is required.")

    effective_max = (
        total_images
        if image_limit == 0
        else min(total_images, image_limit)
    )
    url_image_count = min(effective_max, URL_IMAGE_BUDGET)

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
        image_list,
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
        flattened_images = _flatten_images(image_list)

        for index in range(url_image_count, effective_max):
            parts.append(
                GeminiPart(
                    inlineData=GeminiInlineData(
                        mimeType=GeminiMimeType.image_png,
                        data=tensor_to_base64_string(
                            flattened_images[index]
                        ),
                    )
                )
            )

    return parts


def calculate_tokens_price(
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


def _validate_settings(settings: Any) -> dict[str, Any]:
    if not isinstance(settings, dict):
        raise TypeError(
            "settings must come from the Nanobanana Settings node."
        )

    required_keys = {
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
            "Settings object is incomplete. Missing: "
            + ", ".join(sorted(missing_keys))
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

    if model_label not in MODEL_IDS:
        raise ValueError(f"Invalid model label '{model_label}'.")

    if model_id != MODEL_IDS[model_label]:
        raise ValueError(
            "Model label and model ID in the settings object do not match."
        )

    valid_ratios = (
        BASE_ASPECT_RATIOS
        if model_id == "gemini-3-pro-image-preview"
        else EXTENDED_ASPECT_RATIOS
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

    if modalities not in VALID_MODALITIES:
        raise ValueError(
            f"Invalid response modality '{modalities}'."
        )

    if thinking_level is not None:
        if model_id == "gemini-3-pro-image-preview":
            raise ValueError(
                "Gemini 3 Pro Image does not use thinking_level in this node."
            )
        if thinking_level not in VALID_THINKING_LEVELS:
            raise ValueError(
                f"Invalid thinking level '{thinking_level}'."
            )

    if not isinstance(temperature, (int, float)) or not 0.0 <= temperature <= 2.0:
        raise ValueError("temperature must be between 0.0 and 2.0.")

    if not isinstance(top_p, (int, float)) or not 0.0 <= top_p <= 1.0:
        raise ValueError("top_p must be between 0.0 and 1.0.")

    if not isinstance(system_prompt, str):
        raise TypeError("system_prompt must be a string.")

    return settings


def _default_settings() -> dict[str, Any]:
    return {
        "model_label": "Gemini 3 Pro Image",
        "model_id": MODEL_IDS["Gemini 3 Pro Image"],
        "aspect_ratio": "auto",
        "resolution": "1K",
        "response_modalities": "IMAGE",
        "thinking_level": None,
        "temperature": 1.0,
        "top_p": 0.95,
        "system_prompt": GEMINI_IMAGE_SYS_PROMPT,
    }


class GeminiImageSettings(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NanobananaSettings_KASKI",
            display_name="Nanobanana Settings",
            category=CATEGORY,
            description=(
                "Central settings object for one or more KASKI GeminiImage2 "
                "nodes. Fan this output out to every generator node that should "
                "share the same configuration."
            ),
            inputs=[
                IO.DynamicCombo.Input(
                    "model",
                    options=[
                        IO.DynamicCombo.Option(
                            "Gemini 3 Pro Image",
                            _model_settings_inputs(
                                aspect_ratios=BASE_ASPECT_RATIOS,
                                resolutions=["1K", "2K", "4K"],
                                supports_thinking=False,
                            ),
                        ),
                        IO.DynamicCombo.Option(
                            "Nano Banana 2 (Gemini 3.1 Flash Image)",
                            _model_settings_inputs(
                                aspect_ratios=EXTENDED_ASPECT_RATIOS,
                                resolutions=["1K", "2K", "4K"],
                                supports_thinking=True,
                            ),
                        ),
                        IO.DynamicCombo.Option(
                            "Nano Banana 2 Lite",
                            _model_settings_inputs(
                                aspect_ratios=EXTENDED_ASPECT_RATIOS,
                                resolutions=["1K"],
                                supports_thinking=True,
                            ),
                        ),
                    ],
                    tooltip="Model and model-specific image settings.",
                ),
                IO.Combo.Input(
                    "response_modalities",
                    options=["IMAGE", "IMAGE+TEXT"],
                    default="IMAGE",
                    tooltip=(
                        "IMAGE returns only generated images. IMAGE+TEXT also "
                        "returns the model's text response."
                    ),
                ),
                IO.Float.Input(
                    "temperature",
                    default=1.0,
                    min=0.0,
                    max=2.0,
                    step=0.01,
                    tooltip=(
                        "Controls generation randomness. Lower is more focused; "
                        "higher is more variable."
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
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    tooltip="Shared system prompt for all connected generators.",
                    advanced=True,
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
        response_modalities: str,
        temperature: float,
        top_p: float,
        system_prompt: str,
    ) -> IO.NodeOutput:
        try:
            model_label = model["model"]
            model_id = MODEL_IDS[model_label]

            settings = {
                "model_label": model_label,
                "model_id": model_id,
                "aspect_ratio": model["aspect_ratio"],
                "resolution": model["resolution"],
                "response_modalities": response_modalities,
                "thinking_level": model.get("thinking_level"),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "system_prompt": system_prompt,
            }

            return IO.NodeOutput(_validate_settings(settings))
        except Exception as error:
            _log_soft_error("GeminiImageSettings.execute", error)
            return IO.NodeOutput(_default_settings())


class GeminiImage2(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NanobananaPro_KASKI",
            display_name="GeminiImage2 IO-unlocked",
            category=CATEGORY,
            description=(
                "Generate or edit images via Google Vertex API. Shared model "
                "configuration is supplied by the Nanobanana Settings node."
            ),
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip=(
                        "Text prompt describing the image to generate or the "
                        "edits to apply."
                    ),
                ),
                IO.Custom(SETTINGS_TYPE).Input(
                    "settings",
                    tooltip=(
                        "Connect the output of one central Nanobanana Settings "
                        "node. The same output can feed multiple generators."
                    ),
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip=(
                        "ComfyUI cache-buster and control-after-generate value. "
                        "It forces a new request when changed; it is not sent to "
                        "Gemini by the current image request schema."
                    ),
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip=(
                        "Optional reference image input. This socket accepts a "
                        "single IMAGE or a batched IMAGE tensor. If a batch is "
                        "connected, each image in that batch is sent to Gemini "
                        "as its own reference image."
                    ),
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip=(
                        "Optional Gemini input files from a compatible file node."
                    ),
                ),
            ],
            outputs=[
                IO.Image.Output(display_name="image"),
                IO.String.Output(display_name="text"),
                IO.Image.Output(
                    display_name="thought_image",
                    tooltip=(
                        "Thinking-process image when available. Usually requires "
                        "HIGH thinking and IMAGE+TEXT."
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
        files: list[GeminiPart] | None = None,
    ) -> IO.NodeOutput:
        del seed  # Deliberately used only as a ComfyUI cache-buster.

        try:
            validate_string(prompt, strip_whitespace=True, min_length=1)
            settings = _validate_settings(settings)

            total_images = 0
            if images is not None:
                total_images = get_number_of_images(images)

            if total_images > MAX_REFERENCE_IMAGES:
                raise ValueError(
                    "The current maximum number of supported reference images "
                    f"is {MAX_REFERENCE_IMAGES}; received {total_images}."
                )

            parts: list[GeminiPart] = [GeminiPart(text=prompt)]

            if images is not None and total_images > 0:
                parts.extend(
                    await create_image_parts(
                        cls,
                        images,
                        image_limit=MAX_REFERENCE_IMAGES,
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
                        GeminiTextPart(
                            text=settings["system_prompt"]
                        )
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
                price_extractor=calculate_tokens_price,
            )

            return IO.NodeOutput(
                await get_image_from_response(response),
                get_text_from_response(response),
                await get_image_from_response(
                    response,
                    thought=True,
                ),
            )
        except Exception as error:
            _log_soft_error("GeminiImage2.execute", error)
            black = _black_image()
            return IO.NodeOutput(
                black,
                "",
                black,
            )


NANOBANANA_REWRITE_NODE_CLASS_MAPPINGS = {
    "NanobananaPro_KASKI": GeminiImage2,
    "NanobananaSettings_KASKI": GeminiImageSettings,
}

NANOBANANA_REWRITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "NanobananaPro_KASKI": "GeminiImage2 IO-unlocked",
    "NanobananaSettings_KASKI": "Nanobanana Settings",
}

NODE_CLASS_MAPPINGS = NANOBANANA_REWRITE_NODE_CLASS_MAPPINGS
NODE_DISPLAY_NAME_MAPPINGS = NANOBANANA_REWRITE_NODE_DISPLAY_NAME_MAPPINGS
