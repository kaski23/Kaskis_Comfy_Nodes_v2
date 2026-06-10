import torch
import numpy as np
import math
import torch.nn.functional as F
from comfy.comfy_types import ComfyNodeABC


class VideoSizeLengthConformer(ComfyNodeABC):

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {}),   # (B,H,W,C)
                "min_length": ("INT", {"default": 25, "min": 0, "max": 99999}),
                "max_length": ("INT", {"default": 250, "min": 0, "max": 99999}),

                "min_width":  ("INT", {"default": 720, "min": 0, "max": 8192}),
                "min_height": ("INT", {"default": 720, "min": 0, "max": 8192}),
                "max_width":  ("INT", {"default": 1920, "min": 0, "max": 8192}),
                "max_height": ("INT", {"default": 1920, "min": 0, "max": 8192}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sanitize"
    CATEGORY = "KASKI/Video"


    # -------------------------------------------------------------
    # Main function
    # -------------------------------------------------------------
    def sanitize(
        self,
        video: torch.Tensor,
        min_length: int, max_length: int,
        min_width: int, min_height: int,
        max_width: int, max_height: int
    ):
        # -----------------------------------------------------------
        # n-frames-check
        frames = video
        n, h, w, _ = frames.shape
        
        if min_length + max_length == 0:
            min_length, max_length = n, n
            
        elif min_length > 0 and max_length == 0:
            max_length = 999999999
        
        elif min_length >= max_length:
            raise ValueError ("min_length >= max_length")
            
        # --- Extend ---
        if n < min_length:
            frames = self._ping_pong_extend(frames, target=min_length)
            n = frames.shape[0]

        # --- Shorten ---
        if n > max_length:
            frames = self._adaptive_shorten(frames, target=max_length)
            n = frames.shape[0]
        
        # -----------------------------------------------------------
        #width-check
        if min_width + max_width == 0:
            min_width, max_width = w, w
            
        elif min_width > 0 and max_width == 0:
            max_width = 999999999   
            
        elif min_width >= max_width:
            raise ValueError ("min_width >= max_width")
        
        #height-check
        if min_height + max_height == 0:
            min_height, max_height = h, h
            
        elif min_height > 0 and max_height == 0:
            max_height = 999999999   
            
        elif min_height >= max_height:
            raise ValueError ("min_height >= max_height")

        # --- Resize ---
       
        frames = self._resize_to_constraints(
            frames,
            min_width=min_width,  min_height=min_height,
            max_width=max_width,  max_height=max_height,
        )

        return (frames,)


    # -------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------
    def _ping_pong_extend(self, video: torch.Tensor, target: int) -> torch.Tensor:
        frames = [video]
        forward = True
        length_accum = video.shape[0]

        while length_accum < target:
            block = video.flip(0) if forward else video
            frames.append(block)
            length_accum += block.shape[0]
            forward = not forward

        out = torch.cat(frames, dim=0)
        return out[:target]


    def _adaptive_shorten(self, video: torch.Tensor, target: int) -> torch.Tensor:
        length = video.shape[0]
        if length <= target:
            return video

        # Kick even distributed indiced frames
        indices = torch.linspace(0, length - 1, target).long()
        return video[indices]


    def _resize_to_constraints(
        self,
        video: torch.Tensor,
        min_width: int, min_height: int,
        max_width: int, max_height: int,
    ) -> torch.Tensor:

        B, H, W, C = video.shape

        scale_up = 1.0
        scale_down = 1.0

        # --- Min-Size ---
        if H < min_height:
            scale_up = max(scale_up, min_height / H)
        if W < min_width:
            scale_up = max(scale_up, min_width / W)

        # --- Max-Size ---
        if H > max_height:
            scale_down = min(scale_down, max_height / H)
        if W > max_width:
            scale_down = min(scale_down, max_width / W)

        scale = scale_up * scale_down

        if scale == 1.0:
            return video

        new_h = int(round(H * scale))
        new_w = int(round(W * scale))

        vid = video.permute(0, 3, 1, 2)

        vid = F.interpolate(
            vid,
            size=(new_h, new_w),
            mode="bilinear",
            align_corners=False
        )

        return vid.permute(0, 2, 3, 1).contiguous()

       


class WanVaceInputConform(ComfyNodeABC):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # (B, H, W, C)
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT")
    RETURN_NAMES = ("sequence_lengthened", "optimal_width", "optimal_height")
    FUNCTION = "conform"
    CATEGORY = "KASKI/Video"

    def conform(self, images):
        # --- Ensure tensor ---
        if isinstance(images, list):
            images = torch.stack(images, dim=0)

        B, H, W, C = images.shape

        # --- Frame count: extend to (4n + 1) using ping-pong ---
        remainder = (B - 1) % 4
        if remainder != 0:
            needed = 4 - remainder

            if B > 1:
                # exclude last frame, then flip
                reverse = torch.flip(images[:-1], dims=[0])
            else:
                reverse = images

            # build ping-pong extension
            chunks = []
            total = 0
            while total < needed:
                chunks.append(reverse)
                total += reverse.shape[0]

            extended = torch.cat(chunks, dim=0)[:needed]
            images = torch.cat([images, extended], dim=0)

        # --- Resolution buckets ---
        allowed_resolutions = [
            (480, 832),
            (832, 480),
            (512, 512),
            (768, 768),
            (1024, 1024),
            (1280, 720),
            (720, 1280),
        ]

        # --- Find best fitting resolution (smallest upscale only) ---
        def score():
            aspect_ratio_video = W / H
            scores = []

            for w, h in allowed_resolutions:
                aspect_ratio_bucket = w / h
                aspect_penalty = (aspect_ratio_bucket - aspect_ratio_video) ** 2

                width_diff = w - W
                height_diff = h - H

                scale_penalty = 0
                if width_diff < 0:
                    scale_penalty += (width_diff ** 2) * 2  # Downscale härter bestrafen
                else:
                    scale_penalty += width_diff

                if height_diff < 0:
                    scale_penalty += (height_diff ** 2) * 2
                else:
                    scale_penalty += height_diff

                scores.append(scale_penalty + aspect_penalty * 1000)
            return scores
                
        scores = score()
        best_idx = int(np.argmin(scores))
        width, height = allowed_resolutions[best_idx]
        return (images, width, height)