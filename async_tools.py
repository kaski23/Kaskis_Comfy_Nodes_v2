import asyncio
from comfy.comfy_types import ComfyNodeABC

class AsyncDelay(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "The images to preview."}),
                "delay": ("INT", )
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "run"
    CATEGORY = "KASKI/async"

    async def run(self, image, delay):
        delay = max(0, delay) / 1000.0
        await asyncio.sleep(delay)
        return (image,)
