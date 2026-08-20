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



class SeedanceStutterFix(ComfyNodeABC):

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
                    {"tooltip": "Input IMAGE batch interpreted as a video sequence."},
                ),
                "similarity_threshold": (
                    "FLOAT",
                    {
                        "default": 0.2,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Frames below this continuity threshold are detected as stutters.",
                    },
                ),
                "repair": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "Enable or bypass frame repair. Analysis and frame table are generated either way.",
                    },
                ),
                "repair_method": (
                    ["rife", "blend"],
                    {
                        "default": "rife",
                        "tooltip": "Interpolation method used to reconstruct frames.",
                    },
                ),
                "repair_target": (
                    ["auto", "AXBC", "ABXC", "hardcore"],
                    {
                        "default": "auto",
                        "tooltip": (
                            "Which frame to replace for a detected A-B-B-C stutter. "
                            "auto chooses based on surrounding frame differences; "
                            "AXBC replaces the first B; ABXC replaces the second B; "
                            "hardcore ignores detection and reconstructs every second frame."
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

        Analysis IMAGE:
            [B,H*scale,W*scale,C]
        """

        x = images.permute(0, 3, 1, 2).float()

        height, width = x.shape[2:4]

        target_height = max(1, round(height * cls.ANALYSIS_SCALE))
        target_width = max(1, round(width * cls.ANALYSIS_SCALE))

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
        Calculates one RIFE motion score per frame transition.

        motion[0]:
            frame 0 -> frame 1

        motion[1]:
            frame 1 -> frame 2

        The score is the mean magnitude of both midpoint-directed
        RIFE optical-flow fields.
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

        return torch.stack(motion_scores)


    # -------------------------------------------------------------------------

    @classmethod
    def _calculate_continuity_scores(
        cls,
        motion: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compares each transition against its local temporal context.

        1.0:
            Motion is at least as large as locally expected.

        0.5:
            Motion is roughly half of locally expected motion.

        0.0:
            Motion has collapsed completely.

        Lower score = more suspicious.
        """

        count = motion.shape[0]
        scores = torch.ones_like(motion)

        for i in range(count):

            start = max(0, i - cls.TEMPORAL_RADIUS)
            end = min(count, i + cls.TEMPORAL_RADIUS + 1)

            before = motion[start:i]
            after = motion[i + 1:end]

            if before.numel() > 0 and after.numel() > 0:
                neighbors = torch.cat((before, after))
            elif before.numel() > 0:
                neighbors = before
            elif after.numel() > 0:
                neighbors = after
            else:
                continue

            expected_motion = torch.quantile(
                neighbors,
                0.5,
            )

            if expected_motion <= 1e-8:
                scores[i] = 1.0
                continue

            scores[i] = torch.clamp(
                motion[i] / expected_motion,
                min=0.0,
                max=1.0,
            )

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
            motion
            continuity_scores
        """

        analysis_images = cls._downscale_for_analysis(images)

        motion = cls._calculate_frame_motion(
            analysis_images
        )

        continuity_scores = cls._calculate_continuity_scores(
            motion
        )

        return motion, continuity_scores


    # =========================================================================
    # 2. GENERATE TABLE
    # =========================================================================

    @staticmethod
    def _generate_table(
        motion: torch.Tensor,
        continuity_scores: torch.Tensor,
        similarity_threshold: float,
    ) -> list:
        """
        Converts analysis results into structured frame data.

        This is the ONLY place where the threshold decides whether
        a frame is KEEP or DETECTED.

        This method knows nothing about repair.
        """

        motion_cpu = motion.detach().float().cpu()
        continuity_cpu = continuity_scores.detach().float().cpu()

        table = [{
            "frame": 0,
            "motion": float("inf"),
            "continuity": float("inf"),
            "detected": False,
        }]

        for transition_index in range(continuity_cpu.shape[0]):

            frame_index = transition_index + 1

            motion_score = motion_cpu[
                transition_index
            ].item()

            continuity = continuity_cpu[
                transition_index
            ].item()

            table.append({
                "frame": frame_index,
                "motion": motion_score,
                "continuity": continuity,
                "detected": continuity < similarity_threshold,
            })

        return table


    # -------------------------------------------------------------------------

    @staticmethod
    def _table_to_string(
        table: list,
    ) -> str:
        """
        Human-readable representation of the analysis table.
        """

        rows = [
            "Frame | Motion     | Continuity | Action",
            "------+------------+------------+---------",
        ]

        for entry in table:

            frame_index = entry["frame"]

            if frame_index == 0:
                motion = "inf"
                continuity = "inf"
            else:
                motion = f"{entry['motion']:.6f}"
                continuity = f"{entry['continuity']:.6f}"

            action = (
                "DETECTED"
                if entry["detected"]
                else "KEEP"
            )

            rows.append(
                f"{frame_index:<5} | "
                f"{motion:<10} | "
                f"{continuity:<10} | "
                f"{action}"
            )

        return "\n".join(rows)

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
        repair_target: str = "auto",
    ) -> torch.Tensor:

        if repair_target not in (
            "auto",
            "AXBC",
            "ABXC",
            "hardcore",
        ):
            raise ValueError(
                f"Unknown repair target: {repair_target}"
            )

        output = images.clone()

        working_table = [
            entry.copy()
            for entry in table
        ]

        # =========================================================================
        # Pass 0: Collapse triple / quad / ... duplicates to exactly two frames
        #
        # A B B B C
        #     ^ ^
        #
        # -> A B B C
        #
        # A B B B B C
        #     ^ ^ ^
        #
        # -> A B B C
        #
        # First and last B are preserved.
        # =========================================================================

        i = 1

        while i < len(working_table):

            if not working_table[i]["detected"]:
                i += 1
                continue

            run_start = i
            run_end = i

            while (
                run_end + 1 < len(working_table)
                and working_table[run_end + 1]["detected"]
            ):
                run_end += 1

            # A single detected entry is the normal A B B C case.
            # Two or more consecutive detections mean 3+ duplicate frames.
            if run_end > run_start:

                # Keep:
                #   first B = frame run_start - 1
                #   last  B = frame run_end
                #
                # Drop everything in between:
                #   run_start ... run_end - 1

                keep_mask = torch.ones(
                    output.shape[0],
                    dtype=torch.bool,
                    device=output.device,
                )

                keep_mask[run_start:run_end] = False

                output = output[keep_mask]

                del working_table[
                    run_start:run_end
                ]

                # The original last B is now at run_start and remains detected.
                # Continue after it.
                i = run_start + 1

            else:
                i += 1

        # Table indices must match the filtered sequence.
        for frame_index, entry in enumerate(working_table):
            entry["frame"] = frame_index

        frame_count = output.shape[0]

        # =========================================================================
        # Hardcore
        # =========================================================================

        if repair_target == "hardcore":

            # Work from the already filtered sequence.
            source = output.clone()

            repair_indices = list(
                range(1, frame_count - 1, 2)
            )

            progress_bar = ProgressBar(
                len(repair_indices)
            )

            for frame_index in repair_indices:

                output[frame_index] = cls._repair_frame(
                    repair_method,
                    source[frame_index - 1],
                    source[frame_index + 1],
                )

                progress_bar.update(1)

            return output

        # =========================================================================
        # Detection-based repair
        # =========================================================================

        detected_entries = [
            entry
            for entry in working_table
            if entry["detected"]
        ]

        progress_bar = ProgressBar(
            len(detected_entries)
        )

        for entry in detected_entries:

            # A B B C
            #     ^
            #
            # frame_index points to the second B.

            frame_index = entry["frame"]
            selected_target = repair_target

            # ---------------------------------------------------------------------
            # Auto
            # ---------------------------------------------------------------------

            if repair_target == "auto":

                can_replace_first = frame_index >= 2
                can_replace_second = frame_index < frame_count - 1

                if can_replace_first and can_replace_second:

                    left_motion = working_table[
                        frame_index - 1
                    ]["motion"]

                    right_motion = working_table[
                        frame_index + 1
                    ]["motion"]

                    if left_motion > right_motion:
                        selected_target = "AXBC"
                    else:
                        selected_target = "ABXC"

                elif can_replace_first:
                    selected_target = "AXBC"

                elif can_replace_second:
                    selected_target = "ABXC"

                else:
                    progress_bar.update(1)
                    continue

            # ---------------------------------------------------------------------
            # AXBC
            # ---------------------------------------------------------------------

            if selected_target == "AXBC":

                if frame_index < 2:
                    progress_bar.update(1)
                    continue

                repair_index = frame_index - 1

                image_1 = output[
                    frame_index - 2
                ]

                image_2 = output[
                    frame_index
                ]

            # ---------------------------------------------------------------------
            # ABXC
            # ---------------------------------------------------------------------

            elif selected_target == "ABXC":

                if frame_index >= frame_count - 1:
                    progress_bar.update(1)
                    continue

                repair_index = frame_index

                image_1 = output[
                    frame_index - 1
                ]

                image_2 = output[
                    frame_index + 1
                ]

            # Immediately replace the frame in the current sequence.
            output[repair_index] = cls._repair_frame(
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
        repair_target: str,
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
                f"Expected IMAGE tensor with shape (B,H,W,C), got {tuple(images.shape)}"
            )

        frame_count = images.shape[0]

        # -------------------------------------------------------------------------
        # No frames
        # -------------------------------------------------------------------------

        if frame_count == 0:
            return (
                images,
                "Frame | Raw Diff   | Continuity | Action\n"
                "------+------------+------------+---------",
            )

        # -------------------------------------------------------------------------
        # Single frame
        # -------------------------------------------------------------------------

        if frame_count == 1:
            table = [{
                "frame": 0,
                "raw_diff": float("inf"),
                "continuity": float("inf"),
                "detected": False,
            }]

            return (
                images,
                self._table_to_string(table),
            )

        # -------------------------------------------------------------------------
        # 1. Analyze
        # -------------------------------------------------------------------------

        motion, continuity_scores = self._analyze_images(images)

        # -------------------------------------------------------------------------
        # 2. Generate table
        # -------------------------------------------------------------------------

        table = self._generate_table(
            motion,
            continuity_scores,
            similarity_threshold,
        )

        # -------------------------------------------------------------------------
        # 3. Generate string output
        # -------------------------------------------------------------------------

        frame_table = self._table_to_string(table)

        # -------------------------------------------------------------------------
        # 4. Repair — and ONLY this call is bypassed by repair=False
        # -------------------------------------------------------------------------

        if repair:
            output_images = self._fix_stutters(
                images,
                table,
                repair_method,
                repair_target,
            )
        else:
            output_images = images

        # -------------------------------------------------------------------------
        # Output
        # -------------------------------------------------------------------------

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