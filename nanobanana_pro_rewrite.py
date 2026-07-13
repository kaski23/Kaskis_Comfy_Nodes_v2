import base64
import os
from enum import Enum
from io import BytesIO
from typing import Literal

import torch
from typing_extensions import override

import folder_paths
from comfy_api.latest import IO, ComfyExtension, Input, Types
from comfy_api_nodes.apis.gemini import (
    GeminiContent,
    GeminiFileData,
    GeminiGenerateContentRequest,
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
    Modality,
)
from comfy_api_nodes.util import (
    ApiEndpoint,
    audio_to_base64_string,
    bytesio_to_image_tensor,
    download_url_to_image_tensor,
    get_number_of_images,
    sync_op,
    tensor_to_base64_string,
    upload_images_to_comfyapi,
    validate_string,
    video_to_base64_string,
)


GEMINI_IMAGE_SYS_PROMPT = (
    "You are an expert image-generation engine. You must ALWAYS produce an image.\n"
    "Interpret all user input—regardless of "
    "format, intent, or abstraction—as literal visual directives for image composition.\n"
    "If a prompt is conversational or lacks specific visual details, "
    "you must creatively invent a concrete visual scenario that depicts the concept.\n"
    "Prioritize generating the visual representation above any text, formatting, or conversational requests."
)


async def create_image_parts(
    cls: type[IO.ComfyNode],
    images: Input.Image,
    image_limit: int = 0,
) -> list[GeminiPart]:
    image_parts: list[GeminiPart] = []
    if image_limit < 0:
        raise ValueError("image_limit must be greater than or equal to 0 when creating Gemini image parts.")
    total_images = get_number_of_images(images)
    if total_images <= 0:
        raise ValueError("No images provided to create_image_parts; at least one image is required.")

    # If image_limit == 0 --> use all images; otherwise clamp to image_limit.
    effective_max = total_images if image_limit == 0 else min(total_images, image_limit)

    # Number of images we'll send as URLs (fileData)
    num_url_images = min(effective_max, 10)  # Vertex API max number of image links
    reference_images_urls = await upload_images_to_comfyapi(
        cls,
        images,
        max_images=num_url_images,
    )
    for reference_image_url in reference_images_urls:
        image_parts.append(
            GeminiPart(
                fileData=GeminiFileData(
                    mimeType=GeminiMimeType.image_png,
                    fileUri=reference_image_url,
                )
            )
        )
    for idx in range(num_url_images, effective_max):
        image_parts.append(
            GeminiPart(
                inlineData=GeminiInlineData(
                    mimeType=GeminiMimeType.image_png,
                    data=tensor_to_base64_string(images[idx]),
                )
            )
        )
    return image_parts
    

def get_parts_by_type(response: GeminiGenerateContentResponse, part_type: Literal["text"] | str) -> list[GeminiPart]:
    """
    Filter response parts by their type.

    Args:
        response: The API response from Gemini.
        part_type: Type of parts to extract ("text" or a MIME type).

    Returns:
        List of response parts matching the requested type.
    """
    if not response.candidates:
        if response.promptFeedback and response.promptFeedback.blockReason:
            feedback = response.promptFeedback
            raise ValueError(
                f"Gemini API blocked the request. Reason: {feedback.blockReason} ({feedback.blockReasonMessage})"
            )
        raise ValueError(
            "Gemini API returned no response candidates. If you are using the `IMAGE` modality, "
            "try changing it to `IMAGE+TEXT` to view the model's reasoning and understand why image generation failed."
        )
    parts = []
    blocked_reasons = []
    for candidate in response.candidates:
        if candidate.finishReason and candidate.finishReason.upper() == "IMAGE_PROHIBITED_CONTENT":
            blocked_reasons.append(candidate.finishReason)
            continue
        if candidate.content is None or candidate.content.parts is None:
            continue
        for part in candidate.content.parts:
            if part_type == "text" and part.text:
                parts.append(part)
            elif part.inlineData and part.inlineData.mimeType == part_type:
                parts.append(part)
            elif part.fileData and part.fileData.mimeType == part_type:
                parts.append(part)

    if not parts and blocked_reasons:
        raise ValueError(f"Gemini API blocked the request. Reasons: {blocked_reasons}")

    return parts
    

def calculate_tokens_price(response: GeminiGenerateContentResponse) -> float | None:
    if not response.modelVersion:
        return None
    # Define prices (Cost per 1,000,000 tokens), see https://cloud.google.com/vertex-ai/generative-ai/pricing
    if response.modelVersion in ("gemini-2.5-pro-preview-05-06", "gemini-2.5-pro"):
        input_tokens_price = 1.25
        output_text_tokens_price = 10.0
        output_image_tokens_price = 0.0
    elif response.modelVersion in (
        "gemini-2.5-flash-preview-04-17",
        "gemini-2.5-flash",
    ):
        input_tokens_price = 0.30
        output_text_tokens_price = 2.50
        output_image_tokens_price = 0.0
    elif response.modelVersion in (
        "gemini-2.5-flash-image-preview",
        "gemini-2.5-flash-image",
    ):
        input_tokens_price = 0.30
        output_text_tokens_price = 2.50
        output_image_tokens_price = 30.0
    elif response.modelVersion == "gemini-3-pro-preview":
        input_tokens_price = 2
        output_text_tokens_price = 12.0
        output_image_tokens_price = 0.0
    elif response.modelVersion == "gemini-3-pro-image-preview":
        input_tokens_price = 2
        output_text_tokens_price = 12.0
        output_image_tokens_price = 120.0
    else:
        return None
    final_price = response.usageMetadata.promptTokenCount * input_tokens_price
    if response.usageMetadata.candidatesTokensDetails:
        for i in response.usageMetadata.candidatesTokensDetails:
            if i.modality == Modality.IMAGE:
                final_price += output_image_tokens_price * i.tokenCount  # for Nano Banana models
            else:
                final_price += output_text_tokens_price * i.tokenCount
    if response.usageMetadata.thoughtsTokenCount:
        final_price += output_text_tokens_price * response.usageMetadata.thoughtsTokenCount
    return final_price / 1_000_000.0    


def get_text_from_response(response: GeminiGenerateContentResponse) -> str:
    """
    Extract and concatenate all text parts from the response.

    Args:
        response: The API response from Gemini.

    Returns:
        Combined text from all text parts in the response.
    """
    parts = get_parts_by_type(response, "text")
    return "\n".join([part.text for part in parts])


async def get_image_from_response(response: GeminiGenerateContentResponse) -> Input.Image:
    image_tensors: list[Input.Image] = []
    parts = get_parts_by_type(response, "image/png")
    for part in parts:
        if part.inlineData:
            image_data = base64.b64decode(part.inlineData.data)
            returned_image = bytesio_to_image_tensor(BytesIO(image_data))
        else:
            returned_image = await download_url_to_image_tensor(part.fileData.fileUri)
        image_tensors.append(returned_image)
    if len(image_tensors) == 0:
        return torch.zeros((1, 1024, 1024, 4))
    return torch.cat(image_tensors, dim=0)    

class GeminiImage2(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="NanobananaPro_KASKI",
            display_name="Nanobanana Pro IO-unlocked",
            category="KASKI/api-adaptions/nanobanana",
            description="Generate or edit images synchronously via Google Vertex API.",
            inputs=[
                IO.String.Input(
                    "prompt",
                    multiline=True,
                    tooltip="Text prompt describing the image to generate or the edits to apply. "
                    "Include any constraints, styles, or details the model should follow.",
                    default="",
                ),
                IO.String.Input(
                    "model",
                    default="gemini-3-pro-image-preview",
                    tooltip="Choose between gemini-3-pro-image-preview and Nano Banana 2 (Gemini 3.1 Flash Image)",
                ),
                IO.Int.Input(
                    "seed",
                    default=42,
                    min=0,
                    max=0xFFFFFFFFFFFFFFFF,
                    control_after_generate=True,
                    tooltip="When the seed is fixed to a specific value, the model makes a best effort to provide "
                    "the same response for repeated requests. Deterministic output isn't guaranteed. "
                    "Also, changing the model or parameter settings, such as the temperature, "
                    "can cause variations in the response even when you use the same seed value. "
                    "By default, a random seed value is used.",
                ),
                IO.String.Input(
                    "aspect_ratio",
                    default="auto",
                    tooltip="If set to 'auto', matches your input image's aspect ratio; "
                    "if no image is provided, a 16:9 square is usually generated.",
                ),
                IO.String.Input(
                    "resolution",
                    tooltip="Target output resolution. For 2K/4K the native Gemini upscaler is used.",
                ),
                IO.String.Input(
                    "response_modalities",
                    tooltip="Choose 'IMAGE' for image-only output, or "
                    "'IMAGE+TEXT' to return both the generated image and a text response.",
                ),
                IO.Image.Input(
                    "images",
                    optional=True,
                    tooltip="Optional reference image(s). "
                    "To include multiple images, use the Batch Images node (up to 14).",
                ),
                IO.Custom("GEMINI_INPUT_FILES").Input(
                    "files",
                    optional=True,
                    tooltip="Optional file(s) to use as context for the model. "
                    "Accepts inputs from the Gemini Generate Content Input Files node.",
                ),
                IO.String.Input(
                    "system_prompt",
                    multiline=True,
                    default=GEMINI_IMAGE_SYS_PROMPT,
                    optional=True,
                    tooltip="Foundational instructions that dictate an AI's behavior.",
                ),
            ],
            outputs=[
                IO.Image.Output(),
                IO.String.Output(),
            ],
            hidden=[
                IO.Hidden.auth_token_comfy_org,
                IO.Hidden.api_key_comfy_org,
                IO.Hidden.unique_id,
            ],
            is_api_node=True,
            price_badge=IO.PriceBadge(
                depends_on=IO.PriceBadgeDepends(widgets=["resolution"]),
                expr="""
                (
                  $r := widgets.resolution;
                  ($contains($r,"1k") or $contains($r,"2k"))
                    ? {"type":"usd","usd":0.134,"format":{"suffix":"/Image","approximate":true}}
                    : $contains($r,"4k")
                      ? {"type":"usd","usd":0.24,"format":{"suffix":"/Image","approximate":true}}
                      : {"type":"text","text":"Token-based"}
                )
                """,
            ),
        )

    @classmethod
    async def execute(
        cls,
        prompt: str,
        model: str,
        seed: int,
        aspect_ratio: str,
        resolution: str,
        response_modalities: str,
        images: Input.Image | None = None,
        files: list[GeminiPart] | None = None,
        system_prompt: str = "",
    ) -> IO.NodeOutput:
        # -----------------------------
        # VALIDATION (hard fail)
        # -----------------------------

        validate_string(prompt, strip_whitespace=True, min_length=1)

        # aspect_ratio
        VALID_ASPECT_RATIOS = {
            "auto", "1:1", "2:3", "3:2", "3:4", "4:3",
            "4:5", "5:4", "9:16", "16:9", "21:9"
        }
        if not isinstance(aspect_ratio, str):
            raise TypeError("aspect_ratio must be a string")
        aspect_ratio = aspect_ratio.strip().lower()
        if not aspect_ratio:
            raise ValueError("aspect_ratio must not be empty")
        if aspect_ratio not in VALID_ASPECT_RATIOS:
            raise ValueError(f"Invalid aspect_ratio '{aspect_ratio}'. Allowed: {VALID_ASPECT_RATIOS}")

        # resolution
        VALID_RESOLUTIONS = {"1K", "2K", "4K"}
        if not isinstance(resolution, str):
            raise TypeError("resolution must be a string")
        resolution = resolution.strip().upper()
        if not resolution:
            raise ValueError("resolution must not be empty")
        if resolution not in VALID_RESOLUTIONS:
            raise ValueError(f"Invalid resolution '{resolution}'. Allowed: {VALID_RESOLUTIONS}")

        # response_modalities
        VALID_MODALITIES = {"IMAGE", "IMAGE+TEXT"}
        if not isinstance(response_modalities, str):
            raise TypeError("response_modalities must be a string")
        response_modalities = response_modalities.strip().upper()
        if not response_modalities:
            raise ValueError("response_modalities must not be empty")
        if response_modalities not in VALID_MODALITIES:
            raise ValueError(f"Invalid response_modalities '{response_modalities}'. Allowed: {VALID_MODALITIES}")


        parts: list[GeminiPart] = [GeminiPart(text=prompt)]
        if images is not None:
            if get_number_of_images(images) > 14:
                raise ValueError("The current maximum number of supported images is 14.")
            parts.extend(await create_image_parts(cls, images))
        if files is not None:
            parts.extend(files)

        image_config = GeminiImageConfig(imageSize=resolution)
        if aspect_ratio != "auto":
            image_config.aspectRatio = aspect_ratio

        gemini_system_prompt = None
        if system_prompt:
            gemini_system_prompt = GeminiSystemInstructionContent(parts=[GeminiTextPart(text=system_prompt)], role=None)

        response = await sync_op(
            cls,
            ApiEndpoint(path=f"/proxy/vertexai/gemini/{model}", method="POST"),
            data=GeminiImageGenerateContentRequest(
                contents=[
                    GeminiContent(role=GeminiRole.user, parts=parts),
                ],
                generationConfig=GeminiImageGenerationConfig(
                    responseModalities=(["IMAGE"] if response_modalities == "IMAGE" else ["TEXT", "IMAGE"]),
                    imageConfig=image_config,
                ),
                systemInstruction=gemini_system_prompt,
            ),
            response_model=GeminiGenerateContentResponse,
            price_extractor=calculate_tokens_price,
        )
        return IO.NodeOutput(await get_image_from_response(response), get_text_from_response(response))
        

class GeminiSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ([
                    "gemini-3-pro-image-preview", 
                    "Nano Banana 2 (Gemini 3.1 Flash Image)"
                ], {
                    "default": "gemini-3-pro-image-preview"
                }),
                "aspect_ratio": ([
                    "auto", "1:1", "2:3", "3:2", "3:4", "4:3",
                    "4:5", "5:4", "9:16", "16:9", "21:9"
                ], {
                    "default": "auto"
                }),
                "resolution": (["1K", "2K", "4K"], {
                    "default": "1K"
                }),
                "response_modalities": (["IMAGE", "IMAGE+TEXT"], {
                    "default": "IMAGE"
                }),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("model","aspect_ratio", "resolution", "response_modalities")
    FUNCTION = "get_settings"
    CATEGORY = "KASKI/api-adaptions/nanobanana"

    def get_settings(self, model, aspect_ratio, resolution, response_modalities):
        # hard validation
        valid_models = {"gemini-3-pro-image-preview", "Nano Banana 2 (Gemini 3.1 Flash Image)"}

        valid_aspect_ratios = {
            "auto", "1:1", "2:3", "3:2", "3:4", "4:3",
            "4:5", "5:4", "9:16", "16:9", "21:9"
        }
        valid_resolutions = {"1K", "2K", "4K"}
        valid_modalities = {"IMAGE", "IMAGE+TEXT"}
        
        if not isinstance(model, str):
            raise TypeError("model must be a string")
        model = model.strip().lower()
        if not model:
            raise ValueError("model must not be empty")
        if model not in valid_models:
            raise ValueError(f"Invalid valid_models '{valid_models}'")

        if not isinstance(aspect_ratio, str):
            raise TypeError("aspect_ratio must be a string")
        aspect_ratio = aspect_ratio.strip().lower()
        if not aspect_ratio:
            raise ValueError("aspect_ratio must not be empty")
        if aspect_ratio not in valid_aspect_ratios:
            raise ValueError(f"Invalid aspect_ratio '{aspect_ratio}'")

        if not isinstance(resolution, str):
            raise TypeError("resolution must be a string")
        resolution = resolution.strip().upper()
        if not resolution:
            raise ValueError("resolution must not be empty")
        if resolution not in valid_resolutions:
            raise ValueError(f"Invalid resolution '{resolution}'")

        if not isinstance(response_modalities, str):
            raise TypeError("response_modalities must be a string")
        response_modalities = response_modalities.strip().upper()
        if not response_modalities:
            raise ValueError("response_modalities must not be empty")
        if response_modalities not in valid_modalities:
            raise ValueError(f"Invalid response_modalities '{response_modalities}'")

        return (model, aspect_ratio, resolution, response_modalities)
        



NANOBANANA_REWRITE_NODE_CLASS_MAPPINGS = {
    "NanobananaPro_KASKI": GeminiImage2,
    "NanobananaSettings_KASKI": GeminiSettings,
}


NANOBANANA_REWRITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "NanobananaPro_KASKI": "Nanobanana Pro IO-unlocked",
    "NanobananaSettings_KASKI": "Nanobanana Settings",
}