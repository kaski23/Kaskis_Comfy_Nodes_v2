import math

import torch
import torch.nn.functional as F

from comfy.comfy_types import ComfyNodeABC

from .external_libraries.RIFE import interpolate_between_two_frames
from comfy.utils import ProgressBar

class ExtendVideo(ComfyNodeABC):
    """
    Extends an IMAGE batch interpreted as a video sequence.

    If the input already contains at least n_frames, it is returned
    unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Input image sequence to extend if it contains fewer than the requested number of frames."}),
                "n_frames": (
                    "INT",
                    {"default": 25, "min": 1, "max": 99999, "tooltip": "Minimum number of frames the output sequence should contain."},
                ),
                "method": (
                    [
                        "ping_pong",
                        "repeat_last_frame",
                        "repeat_from_start",
                    ],
                    {"default": "ping_pong", "tooltip": "How additional frames are generated: reverse the sequence, hold the last frame, or loop from the beginning."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    OUTPUT_TOOLTIPS = (
        "Input sequence extended to at least the requested frame count.",
    )
    FUNCTION = "extend"
    CATEGORY = "KASKI/videoTools"

    def extend(
        self,
        video: torch.Tensor,
        n_frames: int,
        method: str,
    ):
        """
        Main node entry point.

        Extends the sequence to at least n_frames using the selected
        temporal repetition method.
        """

        if video.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), got {tuple(video.shape)}"
            )

        current_frames = video.shape[0]

        if current_frames == 0:
            raise ValueError(
                "Video contains no frames."
            )

        # Extension nodes are intentionally non-destructive:
        # sequences that are already long enough pass through unchanged.
        if current_frames >= n_frames:
            return (video,)

        missing_frames = (
            n_frames - current_frames
        )

        if method == "ping_pong":
            extension = self._extend_ping_pong(
                video,
                missing_frames,
            )

        elif method == "repeat_last_frame":
            extension = self._extend_last_frame(
                video,
                missing_frames,
            )

        elif method == "repeat_from_start":
            extension = self._extend_from_start(
                video,
                missing_frames,
            )

        else:
            raise ValueError(
                f"Unknown extension method: {method}"
            )

        output = torch.cat(
            (video, extension),
            dim=0,
        )

        return (output.contiguous(),)

    @staticmethod
    def _extend_last_frame(
        video: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        """
        Creates a static hold by repeating the final source frame.
        """

        return video[-1:].repeat(
            count,
            1,
            1,
            1,
        )

    @staticmethod
    def _extend_from_start(
        video: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        """
        Loops the complete source sequence from its first frame.

        Example:
        0 1 2 3 | 0 1 2 3 | 0 1 ...
        """

        repetitions = math.ceil(
            count / video.shape[0]
        )

        return video.repeat(
            repetitions,
            1,
            1,
            1,
        )[:count]

    @staticmethod
    def _extend_ping_pong(
        video: torch.Tensor,
        count: int,
    ) -> torch.Tensor:
        """
        Extends the sequence forwards/backwards in ping-pong order.

        Turnaround frames are not duplicated.

        Example source:
            0 1 2 3

        Extension:
            2 1 0 1 2 3 2 1 0 ...

        Result:
            0 1 2 3 2 1 0 1 2 3 ...
        """

        frame_count = video.shape[0]

        # A single-frame sequence has no temporal direction,
        # so ping-pong degenerates into a static hold.
        if frame_count == 1:
            return video.repeat(
                count,
                1,
                1,
                1,
            )

        # Exclude the original final frame from the reverse part,
        # because it already exists immediately before the extension.
        reverse = video[:-1].flip(0)

        # Exclude frame zero from the forward part to avoid duplicating
        # it at the second turnaround.
        forward = video[1:]

        # One complete continuation cycle.
        cycle = torch.cat(
            (reverse, forward),
            dim=0,
        )

        repetitions = math.ceil(
            count / cycle.shape[0]
        )

        return cycle.repeat(
            repetitions,
            1,
            1,
            1,
        )[:count]


class ShortenVideo(ComfyNodeABC):
    """
    Shortens an IMAGE batch interpreted as a video sequence.

    If the input already contains at most n_frames, it is returned
    unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE", {"tooltip": "Input image sequence to shorten if it contains more than the requested number of frames."}),
                "n_frames": (
                    "INT",
                    {"default": 25, "min": 1, "max": 99999, "tooltip": "Maximum number of frames the output sequence should contain."},
                ),
                "method": (
                    [
                        "cut_end",
                        "cut_beginning",
                        "resample",
                    ],
                    {"default": "cut_end", "tooltip": "How frames are removed: trim the end, trim the beginning, or evenly resample the full sequence."},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("video",)
    OUTPUT_TOOLTIPS = (
        "Input sequence shortened to at most the requested frame count.",
    )
    FUNCTION = "shorten"
    CATEGORY = "KASKI/videoTools"

    def shorten(
        self,
        video: torch.Tensor,
        n_frames: int,
        method: str,
    ):
        """
        Main node entry point.

        Reduces the sequence to at most n_frames using either trimming
        or temporal resampling.
        """

        if video.ndim != 4:
            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), got {tuple(video.shape)}"
            )

        current_frames = video.shape[0]

        if current_frames == 0:
            raise ValueError(
                "Video contains no frames."
            )

        # Shortening nodes never extend input sequences.
        if current_frames <= n_frames:
            return (video,)

        if method == "cut_end":
            # Keep the beginning of the sequence and discard the tail.
            output = video[:n_frames]

        elif method == "cut_beginning":
            # Keep the end of the sequence and discard the beginning.
            output = video[-n_frames:]

        elif method == "resample":
            # Preserve the complete temporal span while reducing
            # the number of represented frames.
            output = self._resample(
                video,
                n_frames,
            )

        else:
            raise ValueError(
                f"Unknown shortening method: {method}"
            )

        return (output.contiguous(),)

    @staticmethod
    def _resample(
        video: torch.Tensor,
        target_frames: int,
    ) -> torch.Tensor:
        """
        Uniformly samples target_frames across the complete source
        sequence.

        This behaves like a temporal speed-up:
        the first and last source frames remain represented while
        intermediate frames are discarded as evenly as possible.
        """

        source_frames = video.shape[0]

        # Generate evenly spaced floating-point frame positions across
        # the complete sequence, including both endpoints.
        indices = torch.linspace(
            0,
            source_frames - 1,
            steps=target_frames,
            device=video.device,
        )

        # Convert the continuous positions to actual source-frame indices.
        indices = (
            indices
            .round()
            .long()
            .clamp(0, source_frames - 1)
        )

        return video[indices]



class SeedanceStutterFix(ComfyNodeABC):

    # =========================================================================
    # Settings
    # =========================================================================

    ANALYSIS_SCALE = 0.2
    TEMPORAL_RADIUS = 2
    MOTION_TEMPERATURE = 0.01

    # =========================================================================
    # ComfyUI
    # =========================================================================

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": (
                    "IMAGE",
                    {
                        "tooltip":
                            "Input IMAGE batch interpreted as a video sequence."
                    },
                ),

                "similarity_threshold": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip":
                            "Frames below this continuity threshold are detected as stutters.",
                    },
                ),

                "repair": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip":
                            "If disabled, analysis still runs but the input sequence is passed through unchanged.",
                    },
                ),

                "repair_method": (
                    [
                        "rife",
                        "blend",
                    ],
                    {
                        "default": "rife",
                        "tooltip":
                            "Method used to reconstruct detected frames.",
                    },
                ),
            }
        }

    RETURN_TYPES = (
        "IMAGE",
        "STRING",
    )

    RETURN_NAMES = (
        "images",
        "frame_table",
    )

    FUNCTION = "process"
    CATEGORY = "KASKI/videoTools"

    # =========================================================================
    # 1. ANALYZE IMAGES
    # =========================================================================

    @classmethod
    def _downscale_for_analysis(
        cls,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Comfy IMAGE:
            [B,H,W,C]

        Analysis tensor:
            [B,C,H*0.2,W*0.2]
        """

        x = images.permute(
            0,
            3,
            1,
            2,
        ).float()

        height = x.shape[2]
        width = x.shape[3]

        target_height = max(
            1,
            round(height * cls.ANALYSIS_SCALE),
        )

        target_width = max(
            1,
            round(width * cls.ANALYSIS_SCALE),
        )

        return F.interpolate(
            x,
            size=(
                target_height,
                target_width,
            ),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_frame_differences(
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns one difference value per transition.

        differences[0]:
            frame 0 -> frame 1

        differences[1]:
            frame 1 -> frame 2
        """

        previous = images[:-1]
        current = images[1:]

        return torch.abs(
            current - previous
        ).mean(
            dim=(1, 2, 3)
        )

    # -------------------------------------------------------------------------

    @classmethod
    def _calculate_continuity_scores(
        cls,
        differences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compares every frame transition against its local temporal context.

        1.0:
            Current difference is at least as large as expected.

        < 1.0:
            Difference has collapsed relative to surrounding transitions.

        Lower score = more suspicious.
        """

        count = differences.shape[0]

        scores = torch.ones_like(
            differences
        )

        for i in range(count):

            start = max(
                0,
                i - cls.TEMPORAL_RADIUS,
            )

            end = min(
                count,
                i + cls.TEMPORAL_RADIUS + 1,
            )

            before = differences[
                start:i
            ]

            after = differences[
                i + 1:end
            ]

            if (
                before.numel() > 0
                and after.numel() > 0
            ):

                neighbors = torch.cat(
                    (
                        before,
                        after,
                    )
                )

            elif before.numel() > 0:

                neighbors = before

            elif after.numel() > 0:

                neighbors = after

            else:

                scores[i] = 1.0
                continue

            expected_motion = torch.quantile(
                neighbors,
                0.5,
            )

            current_motion = differences[i]

            motion_delta = (
                current_motion
                - expected_motion
            )

            if motion_delta < 0:

                scores[i] = torch.clamp(
                    torch.exp(
                        motion_delta
                        / cls.MOTION_TEMPERATURE
                    ),
                    min=0.0,
                    max=1.0,
                )

            else:

                scores[i] = 1.0

        return scores

    # -------------------------------------------------------------------------

    @classmethod
    def _analyze_images(
        cls,
        images: torch.Tensor,
    ):
        """
        Pure image analysis.

        Does NOT:
            - apply threshold
            - generate table
            - repair anything

        Returns:
            differences
            continuity_scores
        """

        analysis_images = cls._downscale_for_analysis(
            images
        )

        differences = cls._calculate_frame_differences(
            analysis_images
        )

        continuity_scores = cls._calculate_continuity_scores(
            differences
        )

        return (
            differences,
            continuity_scores,
        )

    # =========================================================================
    # 2. GENERATE TABLE
    # =========================================================================

    @staticmethod
    def _generate_table(
        differences: torch.Tensor,
        continuity_scores: torch.Tensor,
        similarity_threshold: float,
    ) -> list:
        """
        Converts analysis results into structured frame data.

        This is the ONLY place where the threshold decides whether
        a frame is KEEP or DETECTED.

        This method knows nothing about repair.
        """

        differences_cpu = (
            differences
            .detach()
            .float()
            .cpu()
        )

        continuity_cpu = (
            continuity_scores
            .detach()
            .float()
            .cpu()
        )

        table = [
            {
                "frame": 0,
                "raw_diff": float("inf"),
                "continuity": float("inf"),
                "detected": False,
            }
        ]

        for transition_index in range(
            continuity_cpu.shape[0]
        ):

            frame_index = (
                transition_index + 1
            )

            raw_diff = differences_cpu[
                transition_index
            ].item()

            continuity = continuity_cpu[
                transition_index
            ].item()

            detected = (
                continuity
                < similarity_threshold
            )

            table.append(
                {
                    "frame": frame_index,
                    "raw_diff": raw_diff,
                    "continuity": continuity,
                    "detected": detected,
                }
            )

        return table

    # -------------------------------------------------------------------------

    @staticmethod
    def _table_to_string(
        table: list,
    ) -> str:
        """
        Human-readable representation of the analysis table.

        This is display only.
        """

        rows = [
            "Frame | Raw Diff   | Continuity | Action",
            "------+------------+------------+---------",
        ]

        for entry in table:

            frame_index = entry["frame"]

            if frame_index == 0:

                raw_diff = "inf"
                continuity = "inf"

            else:

                raw_diff = (
                    f"{entry['raw_diff']:.6f}"
                )

                continuity = (
                    f"{entry['continuity']:.6f}"
                )

            action = (
                "DETECTED"
                if entry["detected"]
                else "KEEP"
            )

            rows.append(
                f"{frame_index:<5} | "
                f"{raw_diff:<10} | "
                f"{continuity:<10} | "
                f"{action}"
            )

        return "\n".join(
            rows
        )

    # =========================================================================
    # 3. REPAIR METHODS
    # =========================================================================

    @staticmethod
    def _repair_rife(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        image_1:
            [H,W,C]

        image_2:
            [H,W,C]

        returns:
            interpolated midpoint [H,W,C]
        """

        return interpolate_between_two_frames(
            image_1,
            image_2,
            model="4.25",
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _repair_blend(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
    ) -> torch.Tensor:

        return torch.lerp(
            image_1,
            image_2,
            0.5,
        )

    # -------------------------------------------------------------------------

    @classmethod
    def _repair_frame(
        cls,
        repair_method: str,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Common repair interface:

            [H,W,C]
            [H,W,C]

                ↓

            [H,W,C]
        """

        if repair_method == "rife":

            return cls._repair_rife(
                image_1,
                image_2,
            )

        if repair_method == "blend":

            return cls._repair_blend(
                image_1,
                image_2,
            )

        raise ValueError(
            f"Unknown repair method: {repair_method}"
        )

    # =========================================================================
    # 4. FIX STUTTERS
    # =========================================================================

    @classmethod
    def _fix_stutters(
        cls,
        images: torch.Tensor,
        table: list,
        repair_method: str,
    ) -> torch.Tensor:
        """
        Consumes the analysis table and repairs all detected frames.

        Seedance failure:

            A B B C
                ^
                detected

        Repair:

            A B X C

        X is generated between:
            images[i - 1]
            images[i + 1]

        The original sequence length is preserved.
        """

        output = images.clone()

        frame_count = images.shape[0]

        detected_entries = [
            entry
            for entry in table
            if entry["detected"]
        ]

        progress_bar = ProgressBar(
            len(detected_entries)
        )

        for entry in detected_entries:

            frame_index = entry["frame"]

            # Cannot generate an intermediate frame if there is
            # no frame after the detected one.
            if frame_index >= frame_count - 1:
                progress_bar.update(1)
                continue

            image_1 = images[
                frame_index - 1
            ]

            image_2 = images[
                frame_index + 1
            ]

            output[
                frame_index
            ] = cls._repair_frame(
                repair_method,
                image_1,
                image_2,
            )

            progress_bar.update(1)

        return output

    # =========================================================================
    # 5. NODE ORCHESTRATION
    # =========================================================================

    def process(
        self,
        images: torch.Tensor,
        similarity_threshold: float,
        repair: bool,
        repair_method: str,
    ):
        """
        Pipeline:

            Analyze Images
                ↓
            Generate Table
                ↓
            Generate Table String
                ↓
            repair?
                ├─ False -> pass original IMAGE through
                └─ True  -> Fix Stutters
        """

        if images.ndim != 4:

            raise ValueError(
                "Expected IMAGE tensor with shape "
                f"(B,H,W,C), got {tuple(images.shape)}"
            )

        frame_count = images.shape[0]

        # ---------------------------------------------------------------------
        # No frames
        # ---------------------------------------------------------------------

        if frame_count == 0:

            return (
                images,
                "Frame | Raw Diff   | Continuity | Action\n"
                "------+------------+------------+---------",
            )

        # ---------------------------------------------------------------------
        # Single frame
        # ---------------------------------------------------------------------

        if frame_count == 1:

            table = [
                {
                    "frame": 0,
                    "raw_diff": float("inf"),
                    "continuity": float("inf"),
                    "detected": False,
                }
            ]

            return (
                images,
                self._table_to_string(
                    table
                ),
            )

        # ---------------------------------------------------------------------
        # 1. Analyze
        # ---------------------------------------------------------------------

        (
            differences,
            continuity_scores,
        ) = self._analyze_images(
            images
        )

        # ---------------------------------------------------------------------
        # 2. Generate table
        # ---------------------------------------------------------------------

        table = self._generate_table(
            differences,
            continuity_scores,
            similarity_threshold,
        )

        # ---------------------------------------------------------------------
        # 3. Generate string output
        # ---------------------------------------------------------------------

        frame_table = self._table_to_string(
            table
        )

        # ---------------------------------------------------------------------
        # 4. Repair — and ONLY this call is bypassed by repair=False
        # ---------------------------------------------------------------------

        if repair:

            output_images = self._fix_stutters(
                images,
                table,
                repair_method,
            )

        else:

            output_images = images

        # ---------------------------------------------------------------------
        # Output
        # ---------------------------------------------------------------------

        return (
            output_images.contiguous(),
            frame_table,
        )



VIDEO_TOOLS_NODE_CLASS_MAPPINGS = {
    "ExtendVideo_KASKI": ExtendVideo,
    "ShortenVideo_KASKI": ShortenVideo,
    "SeedanceStutterFix_KASKI": SeedanceStutterFix,
}



VIDEO_TOOLS_NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtendVideo_KASKI": "Extend Video",
    "ShortenVideo_KASKI": "Shorten Video",
    "SeedanceStutterFix_KASKI": "Seedance Stutter Fix",
}