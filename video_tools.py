import math

import torch
import torch.nn.functional as F

from comfy.comfy_types import ComfyNodeABC

from .external_libraries.RIFE import calculate_optical_flow, interpolate_between_two_frames
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


class TemporalSmoother(ComfyNodeABC):

    # =========================================================================
    # Settings
    # =========================================================================

    ANALYSIS_SCALE = 0.2
    TEMPORAL_RADIUS = 2

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
                            "Input IMAGE batch interpreted as a video sequence.",
                    },
                ),
                "resample": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip":
                            "Apply temporal resampling. Analysis always runs.",
                    },
                ),
                "resample_method": (
                    [
                        "rife",
                        "blend",
                    ],
                    {
                        "default": "rife",
                        "tooltip":
                            "Method used to generate additional temporal samples.",
                    },
                ),
                "sensitivity": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 2.0,
                        "step": 0.05,
                        "tooltip": (
                            "Controls how strongly motion deviations affect "
                            "temporal resampling. 0 disables correction, "
                            "1 uses measured motion directly, values above "
                            "1 increase correction strength."
                        ),
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
        "motion_table",
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

        Analysis IMAGE:
            [B,H*scale,W*scale,C]
        """

        x = images.permute(0, 3, 1, 2).float()

        height, width = x.shape[2:4]

        target_height = max(
            1,
            round(height * cls.ANALYSIS_SCALE),
        )

        target_width = max(
            1,
            round(width * cls.ANALYSIS_SCALE),
        )

        x = F.interpolate(
            x,
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        return x.permute(0, 2, 3, 1).contiguous()

    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_frame_motion(
        images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate one RIFE optical-flow motion score per transition.

        motion[0]:
            frame 0 -> frame 1

        motion[1]:
            frame 1 -> frame 2
        """

        motion_scores = []

        for i in range(images.shape[0] - 1):

            flow_1, flow_2 = calculate_optical_flow(
                images[i],
                images[i + 1],
                model="4.25",
            )

            magnitude_1 = torch.linalg.vector_norm(
                flow_1,
                dim=-1,
            ).mean()

            magnitude_2 = torch.linalg.vector_norm(
                flow_2,
                dim=-1,
            ).mean()

            motion_scores.append(
                (magnitude_1 + magnitude_2) * 0.5
            )

        return torch.stack(
            motion_scores
        )

    # -------------------------------------------------------------------------

    @classmethod
    def _calculate_motion_baseline(
        cls,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Estimate locally expected motion for each transition.

        The current transition is excluded from its own baseline.
        Median is used so local motion outliers have little influence.
        """

        count = motion.shape[0]
        baseline = torch.empty_like(motion)

        for i in range(count):

            start = max(
                0,
                i - cls.TEMPORAL_RADIUS,
            )

            end = min(
                count,
                i + cls.TEMPORAL_RADIUS + 1,
            )

            before = motion[start:i]
            after = motion[i + 1:end]

            if (
                before.numel() > 0
                and after.numel() > 0
            ):

                neighbors = torch.cat(
                    (before, after)
                )

            elif before.numel() > 0:

                neighbors = before

            elif after.numel() > 0:

                neighbors = after

            else:

                baseline[i] = motion[i]
                continue

            baseline[i] = torch.quantile(
                neighbors,
                0.5,
            )

        return baseline

    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_motion_ratio(
        motion: torch.Tensor,
        baseline: torch.Tensor,
    ) -> torch.Tensor:
        """
        Normalize motion against the local baseline.

        ratio = 1:
            approximately one normal temporal interval

        ratio > 1:
            more motion than expected

        ratio < 1:
            less motion than expected
        """

        ratio = torch.ones_like(
            motion
        )

        valid = baseline > 1e-8

        ratio[valid] = (
            motion[valid]
            / baseline[valid]
        )

        return ratio.clamp_min(
            0.0
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _apply_sensitivity(
        motion_ratio: torch.Tensor,
        sensitivity: float,
    ) -> torch.Tensor:
        """
        Scale deviations from neutral motion in logarithmic space.

        sensitivity = 0:
            all ratios become 1 -> no temporal correction

        sensitivity = 1:
            measured ratios remain unchanged

        sensitivity > 1:
            motion deviations are amplified
        """

        return (
            motion_ratio
            .clamp_min(1e-8)
            .pow(sensitivity)
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _calculate_target_intervals(
        adjusted_ratio: torch.Tensor,
    ) -> torch.Tensor:
        """
        Quantize each transition independently.

        adjusted ratio:
            < 0.5       -> 0 intervals
            0.5 - 1.5   -> 1 interval
            1.5 - 2.5   -> 2 intervals
            2.5 - 3.5   -> 3 intervals
            ...

        No cumulative state is carried between transitions.
        """

        return torch.floor(
            adjusted_ratio + 0.5
        ).to(
            dtype=torch.int64
        ).clamp_min(
            0
        )

    # -------------------------------------------------------------------------

    @classmethod
    def _analyze_images(
        cls,
        images: torch.Tensor,
        sensitivity: float,
    ):
        """
        Build the local temporal motion model.

        Returns:
            motion
            baseline
            motion_ratio
            adjusted_ratio
            target_intervals
        """

        analysis_images = cls._downscale_for_analysis(
            images
        )

        motion = cls._calculate_frame_motion(
            analysis_images
        )

        baseline = cls._calculate_motion_baseline(
            motion
        )

        motion_ratio = cls._calculate_motion_ratio(
            motion,
            baseline,
        )

        adjusted_ratio = cls._apply_sensitivity(
            motion_ratio,
            sensitivity,
        )

        target_intervals = cls._calculate_target_intervals(
            adjusted_ratio
        )

        return (
            motion,
            baseline,
            motion_ratio,
            adjusted_ratio,
            target_intervals,
        )

    # =========================================================================
    # 2. GENERATE TABLE
    # =========================================================================

    @staticmethod
    def _generate_table(
        motion: torch.Tensor,
        baseline: torch.Tensor,
        motion_ratio: torch.Tensor,
        adjusted_ratio: torch.Tensor,
        target_intervals: torch.Tensor,
    ) -> list:
        """
        Convert temporal analysis into structured transition data.
        """

        motion_cpu = (
            motion
            .detach()
            .float()
            .cpu()
        )

        baseline_cpu = (
            baseline
            .detach()
            .float()
            .cpu()
        )

        ratio_cpu = (
            motion_ratio
            .detach()
            .float()
            .cpu()
        )

        adjusted_cpu = (
            adjusted_ratio
            .detach()
            .float()
            .cpu()
        )

        intervals_cpu = (
            target_intervals
            .detach()
            .cpu()
        )

        table = [{
            "frame": 0,
            "motion": float("inf"),
            "baseline": float("inf"),
            "motion_ratio": 1.0,
            "adjusted_ratio": 1.0,
            "target_intervals": 0,
            "action": "START",
        }]

        for transition_index in range(
            motion_cpu.shape[0]
        ):

            frame_index = (
                transition_index + 1
            )

            intervals = int(
                intervals_cpu[
                    transition_index
                ].item()
            )

            if intervals == 0:

                action = "COLLAPSE"

            elif intervals == 1:

                action = "KEEP"

            else:

                action = (
                    f"INSERT {intervals - 1}"
                )

            table.append({
                "frame": frame_index,
                "motion": motion_cpu[
                    transition_index
                ].item(),
                "baseline": baseline_cpu[
                    transition_index
                ].item(),
                "motion_ratio": ratio_cpu[
                    transition_index
                ].item(),
                "adjusted_ratio": adjusted_cpu[
                    transition_index
                ].item(),
                "target_intervals": intervals,
                "action": action,
            })

        return table

    # -------------------------------------------------------------------------

    @staticmethod
    def _table_to_string(
        table: list,
    ) -> str:

        rows = [
            "Frame | Motion     | Baseline   | Ratio    | Adjusted | Intervals | Action",
            "------+------------+------------+----------+----------+-----------+---------",
        ]

        for entry in table:

            frame_index = entry["frame"]

            if frame_index == 0:

                motion = "inf"
                baseline = "inf"
                ratio = "1.000000"
                adjusted = "1.000000"
                intervals = "-"

            else:

                motion = (
                    f"{entry['motion']:.6f}"
                )

                baseline = (
                    f"{entry['baseline']:.6f}"
                )

                ratio = (
                    f"{entry['motion_ratio']:.6f}"
                )

                adjusted = (
                    f"{entry['adjusted_ratio']:.6f}"
                )

                intervals = str(
                    entry["target_intervals"]
                )

            rows.append(
                f"{frame_index:<5} | "
                f"{motion:<10} | "
                f"{baseline:<10} | "
                f"{ratio:<8} | "
                f"{adjusted:<8} | "
                f"{intervals:<9} | "
                f"{entry['action']}"
            )

        return "\n".join(
            rows
        )

    # =========================================================================
    # 3. RESAMPLING METHODS
    # =========================================================================

    @staticmethod
    def _resample_rife(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:

        return interpolate_between_two_frames(
            image_1,
            image_2,
            timestep=timestep,
            model="4.25",
        )

    # -------------------------------------------------------------------------

    @staticmethod
    def _resample_blend(
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:

        return torch.lerp(
            image_1,
            image_2,
            timestep,
        )

    # -------------------------------------------------------------------------

    @classmethod
    def _resample_frame(
        cls,
        resample_method: str,
        image_1: torch.Tensor,
        image_2: torch.Tensor,
        timestep: float,
    ) -> torch.Tensor:

        if resample_method == "rife":

            return cls._resample_rife(
                image_1,
                image_2,
                timestep,
            )

        if resample_method == "blend":

            return cls._resample_blend(
                image_1,
                image_2,
                timestep,
            )

        raise ValueError(
            f"Unknown resample method: {resample_method}"
        )

    # =========================================================================
    # 4. TEMPORAL RESAMPLING
    # =========================================================================

    @classmethod
    def _resample_motion(
        cls,
        images: torch.Tensor,
        table: list,
        resample_method: str,
    ) -> torch.Tensor:
        """
        Build a new sequence whose frame density follows locally
        measured motion.

        target_intervals == 0:

            A -> B contains effectively no temporal progress.

            A and B occupy the same temporal slot.
            The later frame B is kept.

        target_intervals == 1:

            A -> B already represents approximately one normal
            temporal interval.

            Keep the original transition.

        target_intervals == 2:

            A -------- B

            becomes:

            A ---- X ---- B

        target_intervals == 3:

            A ------------ B

            becomes:

            A ---- X ---- Y ---- B

        There is no upper interval limit.
        """

        output_frames = [
            images[0]
        ]

        generated_frames = sum(
            max(
                entry["target_intervals"] - 1,
                0,
            )
            for entry in table[1:]
        )

        progress_bar = ProgressBar(
            generated_frames
        )

        for transition_index, entry in enumerate(
            table[1:]
        ):

            image_1 = images[
                transition_index
            ]

            image_2 = images[
                transition_index + 1
            ]

            interval_count = entry[
                "target_intervals"
            ]

            # -----------------------------------------------------------------
            # 0 intervals
            #
            # Both originals occupy the same temporal slot.
            # Keep the later frame.
            # -----------------------------------------------------------------

            if interval_count == 0:

                output_frames[-1] = image_2
                continue

            # -----------------------------------------------------------------
            # Insert missing temporal samples.
            # -----------------------------------------------------------------

            for step in range(
                1,
                interval_count,
            ):

                timestep = (
                    step
                    / interval_count
                )

                output_frames.append(
                    cls._resample_frame(
                        resample_method,
                        image_1,
                        image_2,
                        timestep,
                    )
                )

                progress_bar.update(1)

            # Original second frame closes the temporal segment.
            output_frames.append(
                image_2
            )

        return torch.stack(
            output_frames,
            dim=0,
        )

    # =========================================================================
    # 5. NODE ORCHESTRATION
    # =========================================================================

    def process(
        self,
        images: torch.Tensor,
        resample: bool,
        resample_method: str,
        sensitivity: float,
    ):
        """
        Pipeline:

            RIFE Optical Flow
                ↓
            Local Motion Baseline
                ↓
            Motion Ratio
                ↓
            Sensitivity
                ↓
            Local Interval Quantization
                ↓
            Motion Table
                ↓
            resample?
                ├─ False -> original sequence
                └─ True  -> temporally smoothed sequence
        """

        if images.ndim != 4:

            raise ValueError(
                f"Expected IMAGE tensor with shape (B,H,W,C), "
                f"got {tuple(images.shape)}"
            )

        frame_count = images.shape[0]

        # ---------------------------------------------------------------------
        # No frames
        # ---------------------------------------------------------------------

        if frame_count == 0:

            return (
                images,
                "Frame | Motion     | Baseline   | Ratio    | Adjusted | Intervals | Action\n"
                "------+------------+------------+----------+----------+-----------+---------",
            )

        # ---------------------------------------------------------------------
        # Single frame
        # ---------------------------------------------------------------------

        if frame_count == 1:

            table = [{
                "frame": 0,
                "motion": float("inf"),
                "baseline": float("inf"),
                "motion_ratio": 1.0,
                "adjusted_ratio": 1.0,
                "target_intervals": 0,
                "action": "START",
            }]

            return (
                images,
                self._table_to_string(
                    table
                ),
            )

        # ---------------------------------------------------------------------
        # Analyze
        # ---------------------------------------------------------------------

        (
            motion,
            baseline,
            motion_ratio,
            adjusted_ratio,
            target_intervals,
        ) = self._analyze_images(
            images,
            sensitivity,
        )

        # ---------------------------------------------------------------------
        # Generate table
        # ---------------------------------------------------------------------

        table = self._generate_table(
            motion,
            baseline,
            motion_ratio,
            adjusted_ratio,
            target_intervals,
        )

        motion_table = self._table_to_string(
            table
        )

        # ---------------------------------------------------------------------
        # Resample
        # ---------------------------------------------------------------------

        if resample:

            output_images = self._resample_motion(
                images,
                table,
                resample_method,
            )

        else:

            output_images = images

        return (
            output_images.contiguous(),
            motion_table,
        )





VIDEO_TOOLS_NODE_CLASS_MAPPINGS = {
    "ExtendVideo_KASKI": ExtendVideo,
    "ShortenVideo_KASKI": ShortenVideo,
    "TemporalSmoother_KASKI": TemporalSmoother,
}



VIDEO_TOOLS_NODE_DISPLAY_NAME_MAPPINGS = {
    "ExtendVideo_KASKI": "Extend Video",
    "ShortenVideo_KASKI": "Shorten Video",
    "TemporalSmoother_KASKI": "Temporal Smoother",
}