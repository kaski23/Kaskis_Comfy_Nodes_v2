import math

import torch
import torch.nn.functional as F
from comfy.comfy_types import ComfyNodeABC


import math

import torch
from comfy.comfy_types import ComfyNodeABC


class MinMaxSize(ComfyNodeABC):
    """
    Calculates an optimal output canvas size from minimum and maximum
    resolution constraints without modifying the input image.

    The original aspect ratio is preserved whenever all constraints can
    be satisfied through proportional scaling alone.

    If this is impossible, the image is assumed to be scaled as far as
    possible without exceeding the maximum bounds. The remaining
    minimum constraint is then fulfilled through padding by the
    downstream resize/padding node.

    A constraint value of 0 disables that constraint.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Input image or image sequence used to determine the optimal output size."}),

                "min_width": (
                    "INT",
                    {"default": 720, "min": 0, "max": 8192, "tooltip": "Minimum allowed output width. Set to 0 to disable this constraint."},
                ),
                "min_height": (
                    "INT",
                    {"default": 720, "min": 0, "max": 8192, "tooltip": "Minimum allowed output height. Set to 0 to disable this constraint."},
                ),

                "max_width": (
                    "INT",
                    {"default": 1920, "min": 0, "max": 8192, "tooltip": "Maximum allowed output width. Set to 0 to disable this constraint."},
                ),
                "max_height": (
                    "INT",
                    {"default": 1920, "min": 0, "max": 8192, "tooltip": "Maximum allowed output height. Set to 0 to disable this constraint."},
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("optimal_width", "optimal_height")
    OUTPUT_TOOLTIPS = (
        "Recommended output width within the specified size constraints.",
        "Recommended output height within the specified size constraints.",
    )

    FUNCTION = "calculate"
    CATEGORY = "KASKI/InputConform"


    def calculate(
        self,
        image: torch.Tensor,
        min_width: int,
        min_height: int,
        max_width: int,
        max_height: int,
    ):
        """
        Main node entry point.

        Determines the optimal final canvas resolution based on the
        dimensions of the input IMAGE batch.

        No actual resize or padding operation is performed.
        """

        if image.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), "
                f"got {tuple(image.shape)}"
            )

        if image.shape[0] == 0:
            raise ValueError(
                "IMAGE batch contains no images."
            )

        self._validate_bounds(
            min_width,
            max_width,
            "width",
        )

        self._validate_bounds(
            min_height,
            max_height,
            "height",
        )

        _, height, width, _ = image.shape

        # Check whether the source already violates a minimum constraint.
        needs_upscale = (
            (min_width > 0 and width < min_width)
            or
            (min_height > 0 and height < min_height)
        )

        # Check whether the source already violates a maximum constraint.
        needs_downscale = (
            (max_width > 0 and width > max_width)
            or
            (max_height > 0 and height > max_height)
        )

        # If the source already lies inside all active bounds,
        # there is no reason to change its resolution.
        if not needs_upscale and not needs_downscale:
            return (
                width,
                height,
            )

        # Minimum proportional scale required to satisfy all
        # active minimum dimensions.
        min_scale = self._get_min_scale(
            width,
            height,
            min_width,
            min_height,
        )

        # Maximum proportional scale allowed before exceeding
        # any active maximum dimension.
        max_scale = self._get_max_scale(
            width,
            height,
            max_width,
            max_height,
        )

        if min_scale <= max_scale:
            # There exists a proportional scale that satisfies every
            # active constraint. No padding is required.

            if needs_upscale:
                scale = min_scale

            elif needs_downscale:
                scale = max_scale

            else:
                scale = 1.0

            optimal_width, optimal_height = (
                self._get_scaled_dimensions(
                    width,
                    height,
                    scale,
                )
            )

        else:
            # No proportional scale can satisfy all constraints.
            #
            # Example:
            #
            # Source:      2000 x 500
            # Min height:   720
            # Max width:   1920
            #
            # Scaling to H=720 would produce W=2880 and violate
            # max_width.
            #
            # Therefore:
            # 1. Scale as far as the maximum constraints allow.
            # 2. Use padding to satisfy the remaining minimum.

            scale = max_scale

            scaled_width, scaled_height = (
                self._get_scaled_dimensions(
                    width,
                    height,
                    scale,
                )
            )

            # The final canvas must be at least as large as the
            # minimum constraints. Any missing area is expected
            # to be created as padding downstream.
            optimal_width = scaled_width
            optimal_height = scaled_height

            if min_width > 0:
                optimal_width = max(
                    optimal_width,
                    min_width,
                )

            if min_height > 0:
                optimal_height = max(
                    optimal_height,
                    min_height,
                )

        # Final safety clamp against enabled maximum constraints.
        #
        # This mainly protects against integer rounding around the
        # calculated proportional scale.
        if max_width > 0:
            optimal_width = min(
                optimal_width,
                max_width,
            )

        if max_height > 0:
            optimal_height = min(
                optimal_height,
                max_height,
            )

        return (
            int(optimal_width),
            int(optimal_height),
        )


    @staticmethod
    def _validate_bounds(
        minimum: int,
        maximum: int,
        axis_name: str,
    ) -> None:
        """
        Validates one min/max constraint pair.

        Zero means "disabled", so minimum and maximum are only compared
        when both constraints are active.
        """

        if minimum < 0 or maximum < 0:
            raise ValueError(
                f"{axis_name}: min/max values must be >= 0."
            )

        if (
            minimum > 0
            and maximum > 0
            and minimum > maximum
        ):
            raise ValueError(
                f"{axis_name}: minimum ({minimum}) cannot exceed "
                f"maximum ({maximum})."
            )


    @staticmethod
    def _get_min_scale(
        width: int,
        height: int,
        min_width: int,
        min_height: int,
    ) -> float:
        """
        Calculates the smallest proportional scale factor that would
        satisfy all active minimum dimensions.

        Returns at least 1.0 because minimum constraints can only
        require upscaling.
        """

        scale = 1.0

        if min_width > 0 and width < min_width:
            scale = max(
                scale,
                min_width / width,
            )

        if min_height > 0 and height < min_height:
            scale = max(
                scale,
                min_height / height,
            )

        return scale


    @staticmethod
    def _get_max_scale(
        width: int,
        height: int,
        max_width: int,
        max_height: int,
    ) -> float:
        """
        Calculates the largest proportional scale factor that still
        fits inside all active maximum dimensions.

        Returns infinity if no maximum constraint is enabled.
        """

        scale = float("inf")

        if max_width > 0:
            scale = min(
                scale,
                max_width / width,
            )

        if max_height > 0:
            scale = min(
                scale,
                max_height / height,
            )

        return scale


    @staticmethod
    def _get_scaled_dimensions(
        width: int,
        height: int,
        scale: float,
    ):
        """
        Converts a continuous proportional scale factor into integer
        output dimensions.

        Upscaling rounds upwards so a minimum constraint is not missed
        by a fraction of a pixel.

        Downscaling rounds downwards so a maximum constraint is not
        exceeded by a fraction of a pixel.
        """

        if abs(scale - 1.0) < 1e-8:
            return (
                width,
                height,
            )

        if scale > 1.0:
            scaled_width = max(
                1,
                int(math.ceil(
                    width * scale - 1e-6
                )),
            )

            scaled_height = max(
                1,
                int(math.ceil(
                    height * scale - 1e-6
                )),
            )

        else:
            scaled_width = max(
                1,
                int(math.floor(
                    width * scale + 1e-6
                )),
            )

            scaled_height = max(
                1,
                int(math.floor(
                    height * scale + 1e-6
                )),
            )

        return (
            scaled_width,
            scaled_height,
        )



class AlignFramesToSeconds(ComfyNodeABC):
    """
    Aligns a frame count to the next full second at a given frame rate.

    If the input frame count already fits within an exact whole-second
    duration, that duration is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "n_frames": (
                    "INT",
                    {
                        "default": 25,
                        "min": 1,
                        "max": 999999,
                        "tooltip": "Number of frames in the source sequence.",
                    },
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 1.0,
                        "max": 240.0,
                        "tooltip": "Frame rate used to align the sequence length to a whole number of seconds.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "frames_to_lengthen_to",
        "length_in_seconds",
        "fps",
    )

    OUTPUT_TOOLTIPS = (
        "Smallest frame count that covers the calculated whole-second duration.",
        "Smallest whole-second duration that can contain the source sequence.",
        "Pass-through of the fps used.",
    )

    FUNCTION = "align"
    CATEGORY = "KASKI/InputConform"

    def align(
        self,
        n_frames: int,
        fps: float,
    ):
        """
        Calculates the smallest whole-second duration that can contain
        the given number of frames, then calculates the minimum number
        of frames required to cover that duration.
        """

        length_in_seconds = math.ceil(
            n_frames / fps
        )

        frames_to_lengthen_to = math.ceil(
            length_in_seconds * fps
        )

        return (
            frames_to_lengthen_to,
            length_in_seconds,
            fps,
        )
        

class WanVideoOptimals(ComfyNodeABC):
    """
    Calculates WAN/VACE-compatible target parameters without modifying
    the input video.

    Outputs:
    - preferred target width
    - preferred target height
    - next valid temporal length following the 4n + 1 rule
    """

    # Supported/preferred WAN/VACE resolution buckets.
    RESOLUTION_BUCKETS = (
        (480, 832),
        (832, 480),
        (512, 512),
        (768, 768),
        (1024, 1024),
        (1280, 720),
        (720, 1280),
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Input image sequence used to calculate WAN/VACE-compatible resolution and frame-count targets."}),
            }
        }

    RETURN_TYPES = (
        "INT",
        "INT",
        "INT",
    )

    RETURN_NAMES = (
        "optimal_width",
        "optimal_height",
        "optimal_n_frames",
    )
    OUTPUT_TOOLTIPS = (
        "Recommended WAN/VACE target width.",
        "Recommended WAN/VACE target height.",
        "Smallest frame count greater than or equal to the input length that satisfies the 4n + 1 requirement.",
    )

    FUNCTION = "calculate"
    CATEGORY = "KASKI/InputConform"

    def calculate(
        self,
        video: torch.Tensor,
    ):
        """
        Main node entry point.

        Inspects the source dimensions and frame count, then calculates
        suitable WAN/VACE target parameters without touching the video.
        """

        if video.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), got {tuple(video.shape)}"
            )

        frame_count, height, width, _ = video.shape

        if frame_count == 0:
            raise ValueError(
                "Video contains no frames."
            )

        optimal_width, optimal_height = (
            self._find_resolution_bucket(
                width,
                height,
            )
        )

        optimal_n_frames = (
            self._next_4n_plus_1(
                frame_count
            )
        )

        return (
            optimal_width,
            optimal_height,
            optimal_n_frames,
        )

    @staticmethod
    def _next_4n_plus_1(
        frame_count: int,
    ) -> int:
        """
        Returns the smallest frame count >= the source length that
        satisfies WAN/VACE's 4n + 1 temporal requirement.

        Examples:
            1  -> 1
            2  -> 5
            5  -> 5
            6  -> 9
            24 -> 25
            25 -> 25
        """

        remainder = (
            frame_count - 1
        ) % 4

        # Source length is already valid.
        if remainder == 0:
            return frame_count

        return (
            frame_count
            + (4 - remainder)
        )

    @classmethod
    def _find_resolution_bucket(
        cls,
        width: int,
        height: int,
    ):
        """
        Evaluates every supported WAN/VACE resolution bucket and returns
        the one with the lowest heuristic score.
        """

        best_bucket = None
        best_score = float("inf")

        for (
            bucket_width,
            bucket_height,
        ) in cls.RESOLUTION_BUCKETS:

            score = cls._bucket_score(
                width,
                height,
                bucket_width,
                bucket_height,
            )

            if score < best_score:
                best_score = score
                best_bucket = (
                    bucket_width,
                    bucket_height,
                )

        return best_bucket

    @staticmethod
    def _bucket_score(
        width: int,
        height: int,
        bucket_width: int,
        bucket_height: int,
    ) -> float:
        """
        Scores how well a WAN/VACE resolution bucket matches the source.

        The heuristic considers:
        - aspect-ratio difference
        - absolute width/height difference
        - a stronger penalty when the bucket would require downscaling

        Lower scores represent a better fit.

        This intentionally mirrors the behavior of the previous
        WanVaceInputConform implementation.
        """

        video_aspect = (
            width / height
        )

        bucket_aspect = (
            bucket_width / bucket_height
        )

        # Penalize buckets whose shape differs from the source.
        aspect_penalty = (
            bucket_aspect - video_aspect
        ) ** 2

        width_diff = (
            bucket_width - width
        )

        height_diff = (
            bucket_height - height
        )

        scale_penalty = 0.0

        # A negative difference means the source dimension is larger
        # than the bucket and therefore requires downscaling.
        #
        # Downscaling is intentionally penalized quadratically here,
        # while upscaling remains linear.
        if width_diff < 0:
            scale_penalty += (
                width_diff ** 2
            ) * 2
        else:
            scale_penalty += width_diff

        if height_diff < 0:
            scale_penalty += (
                height_diff ** 2
            ) * 2
        else:
            scale_penalty += height_diff

        # Aspect ratio receives an additional weight so similarly sized
        # but badly shaped buckets are less likely to win.
        return (
            scale_penalty
            + aspect_penalty * 1000
        )


IOCONFORMER_NODE_CLASS_MAPPINGS = {
    "MinMaxSize_KASKI": MinMaxSize,
    "WanVideoOptimals_KASKI": WanVideoOptimals,
    "AlignFramesToSeconds_KASKI" : AlignFramesToSeconds,
}


IOCONFORMER_NODE_DISPLAY_NAME_MAPPINGS = {
    "MinMaxSize_KASKI": "Min/Max Size",
    "WanVideoOptimals_KASKI": "WAN Video Optimals",
    "AlignFramesToSeconds_KASKI": "Align Frames to Seconds",
}