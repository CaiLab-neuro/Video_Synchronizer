#!/usr/bin/env python3
"""
Author: Yanbin Xu
Date: 1.19.2026

This script creates cut-video-aligned gaze/world CSV files for downstream
`gazed_object` analysis.

Example Use: 
to process a single subject with specific camera types:
python gaze_frame_alignment_in_cut_3.py <input_cut_video_dir> <input_gaze_world_dir> <output_dir> <input_pickle_file_path> [--subject-ids <subject_id>] [--camera-ids <camera_id>] [--log-path <log_file_path>]

Inputs:

    1) `input_video_dir`
        Directory containing cut egocentric videos. 
        - `{subject}_{camera}_cut_merged.mp4`
    2) `input_gaze_world_dir`
        Directory containing raw gaze and world timestamp CSV files.
        - `{subject}_{camera}_gaze.csv`
        - `{subject}_{camera}_world_timestamps.csv`
    3) `output_dir`
        Directory where cut-aligned CSV files and log file will be written.
    4) `pickle_file`
        Path to pickle containing cut-frame metadata per subject/camera.
        Expected structure:
            `{subject_key: {camera: [start_frame, ...]}}`

Outputs:

    For each processed `(subject, camera)`, the script writes:
    1) `{output_dir}/{subject}_{camera}_world_timestamps.csv`
    - rows restricted to cut-video duration
    - includes `frame_idx` (0-based in cut video)
    - includes `frame_timestamp`
    - includes `source_frame_idx` (index in original world video)
    2) `{output_dir}/{subject}_{camera}_gaze.csv`
    - gaze rows aligned to cut-world timestamps
    - preserves original gaze columns
    - includes `frame_idx`, `frame_timestamp`, `source_frame_idx`

    It also writes a log file:
    - `{output_dir}/gaze_object.log` (or `--log-path` if provided)
"""

import argparse
import logging
import os
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import pandas as pd


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, mode="w"),
        ],
        force=True,
    )


class GazeFrameAligner:
    def __init__(self, video_dir: str, gaze_world_dir: str, output_dir: str, cut_info: dict, logger=None):
        self.video_dir = Path(video_dir)
        self.gaze_world_path = Path(gaze_world_dir)
        self.output_dir = Path(output_dir)
        self.cut_info = cut_info
        self.logger = logger or logging.getLogger(__name__)

    @staticmethod
    def load_csv(file_path: Path) -> pd.DataFrame:
        return pd.read_csv(file_path)

    @staticmethod
    def discover_pairs(video_dir: Path) -> list[tuple[str, str]]:
        pairs: set[tuple[str, str]] = set()
        for video_path in video_dir.glob("*_cut_merged.mp4"):
            parts = video_path.stem.split("_")
            if len(parts) < 4:
                continue
            pairs.add((parts[0], parts[1]))
        return sorted(pairs)

    @staticmethod
    def _count_frames_ffprobe(video_path: Path) -> int | None:
        logger = logging.getLogger(__name__)
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=nb_read_frames",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except FileNotFoundError as exc:
            raise RuntimeError("ffprobe is required but was not found in PATH.") from exc
        except subprocess.CalledProcessError as exc:
            logger.warning(
                "ffprobe failed for %s (exit=%s). Falling back to OpenCV decode.",
                video_path,
                exc.returncode,
            )
            return None

        frame_counts = [int(line.strip()) for line in result.stdout.splitlines() if line.strip().isdigit()]
        if not frame_counts:
            logger.warning(
                "ffprobe returned no numeric frame count for %s. Falling back to OpenCV decode.",
                video_path,
            )
            return None

        ffprobe_count = max(frame_counts)
        if ffprobe_count <= 0:
            logger.warning(
                "ffprobe returned non-positive frame count for %s: %d. Falling back to OpenCV decode.",
                video_path,
                ffprobe_count,
            )
            return None
        return ffprobe_count

    @staticmethod
    def _get_cap_prop_frame_count(video_path: Path) -> int | None:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        cap_prop_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        if cap_prop_count <= 0:
            return None
        return cap_prop_count

    @staticmethod
    def _count_frames_opencv_decode(video_path: Path) -> int:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        frame_count = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            frame_count += 1

        cap.release()
        return frame_count

    @staticmethod
    def count_video_frames(video_path: Path) -> int:
        logger = logging.getLogger(__name__)
        ffprobe_count = GazeFrameAligner._count_frames_ffprobe(video_path)
        cap_prop_count = GazeFrameAligner._get_cap_prop_frame_count(video_path)

        if ffprobe_count is not None and cap_prop_count is not None and ffprobe_count == cap_prop_count:
            logger.info(
                "Frame count path=ffprobe_fast_path video=%s frames=%d",
                video_path,
                ffprobe_count,
            )
            return ffprobe_count

        if ffprobe_count is not None and cap_prop_count is not None and ffprobe_count != cap_prop_count:
            logger.warning(
                "Frame count mismatch for %s: ffprobe=%d cap_prop=%d. Falling back to OpenCV decode.",
                video_path,
                ffprobe_count,
                cap_prop_count,
            )
        elif ffprobe_count is not None and cap_prop_count is None:
            logger.warning(
                "CAP_PROP_FRAME_COUNT unavailable for %s. Falling back to OpenCV decode.",
                video_path,
            )
        elif ffprobe_count is None and cap_prop_count is not None:
            logger.warning(
                "ffprobe frame count unavailable for %s. Falling back to OpenCV decode.",
                video_path,
            )

        decoded_count = GazeFrameAligner._count_frames_opencv_decode(video_path)
        logger.info(
            "Frame count path=opencv_fallback video=%s ffprobe_frames=%s cap_prop_frames=%s decoded_frames=%d",
            video_path,
            ffprobe_count if ffprobe_count is not None else "n/a",
            cap_prop_count if cap_prop_count is not None else "n/a",
            decoded_count,
        )
        return decoded_count

    def get_cut_info_entry(self, subject_id: str, camera: str):
        if subject_id in self.cut_info and camera in self.cut_info[subject_id]:
            return self.cut_info[subject_id][camera]

        try:
            subject_int = int(subject_id)
        except ValueError:
            return None

        if subject_int in self.cut_info and camera in self.cut_info[subject_int]:
            return self.cut_info[subject_int][camera]

        return None

    def build_cut_world(self, world_df: pd.DataFrame, start_frame: int, num_cut_frames: int) -> pd.DataFrame:
        if "timestamp [ns]" not in world_df.columns:
            raise ValueError("Missing required world column: timestamp [ns]")

        source_start = max(int(start_frame), 0)
        source_end = min(source_start + int(num_cut_frames) - 1, len(world_df) - 1)
        if source_end < source_start:
            return world_df.iloc[0:0].copy()

        world_cut_df = world_df.iloc[source_start : source_end + 1].copy().reset_index(drop=True)
        world_cut_df["source_frame_idx"] = range(source_start, source_start + len(world_cut_df))
        world_cut_df["frame_idx"] = range(len(world_cut_df))
        world_cut_df["frame_timestamp"] = world_cut_df["timestamp [ns]"]
        return world_cut_df

    def build_cut_gaze(self, gaze_df: pd.DataFrame, world_cut_df: pd.DataFrame) -> pd.DataFrame:
        required_gaze_cols = {"timestamp [ns]", "gaze x [px]", "gaze y [px]"}
        if not required_gaze_cols.issubset(gaze_df.columns):
            missing = sorted(required_gaze_cols.difference(gaze_df.columns))
            raise ValueError(f"Missing required gaze columns: {missing}")

        gaze_sorted = gaze_df.sort_values("timestamp [ns]").copy()
        world_sorted = world_cut_df.sort_values("timestamp [ns]").copy()

        aligned = pd.merge_asof(
            gaze_sorted,
            world_sorted[["timestamp [ns]", "frame_idx", "frame_timestamp", "source_frame_idx"]],
            left_on="timestamp [ns]",
            right_on="timestamp [ns]",
            direction="backward",
            allow_exact_matches=True,
        )

        return aligned.dropna(subset=["frame_idx"]).reset_index(drop=True)

    def process_pair(self, subject_id: str, camera: str) -> None:
        cut_video_path = self.video_dir / f"{subject_id}_{camera}_cut_merged.mp4"
        world_path = self.gaze_world_path / f"{subject_id}_{camera}_world_timestamps.csv"
        gaze_path = self.gaze_world_path / f"{subject_id}_{camera}_gaze.csv"

        if not cut_video_path.exists():
            self.logger.warning("Missing cut video: %s. Skipping.", cut_video_path)
            return
        if not world_path.exists():
            self.logger.warning("Missing world CSV: %s. Skipping.", world_path)
            return
        if not gaze_path.exists():
            self.logger.warning("Missing gaze CSV: %s. Skipping.", gaze_path)
            return

        cut_entry = self.get_cut_info_entry(subject_id, camera)
        if cut_entry is None:
            self.logger.warning(
                "No cut information in pickle for subject=%s camera=%s. Skipping.",
                subject_id,
                camera,
            )
            return

        start_frame = int(cut_entry[0])
        num_cut_frames = self.count_video_frames(cut_video_path)
        if num_cut_frames <= 0:
            self.logger.warning("Cut video has no readable frames: %s. Skipping.", cut_video_path)
            return

        world_df = self.load_csv(world_path)
        gaze_df = self.load_csv(gaze_path)

        world_cut_df = self.build_cut_world(world_df, start_frame, num_cut_frames)
        if len(world_cut_df) != num_cut_frames:
            self.logger.warning(
                "World cut length mismatch for subject=%s camera=%s: video_frames=%d world_frames=%d",
                subject_id,
                camera,
                num_cut_frames,
                len(world_cut_df),
            )

        cut_gaze_df = self.build_cut_gaze(gaze_df, world_cut_df)

        out_world = self.output_dir / f"{subject_id}_{camera}_world_timestamps_cut.csv"
        out_gaze = self.output_dir / f"{subject_id}_{camera}_gaze_cut.csv"
        world_cut_df.to_csv(out_world, index=False)
        cut_gaze_df.to_csv(out_gaze, index=False)

        self.logger.info(
            "Saved subject=%s camera=%s -> world_rows=%d gaze_rows=%d",
            subject_id,
            camera,
            len(world_cut_df),
            len(cut_gaze_df),
        )
        self.logger.info("World CSV: %s", out_world)
        self.logger.info("Gaze CSV: %s", out_gaze)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create cut-video gaze/world CSV files for gazed_object pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input_video_dir", help="Directory containing *_cut_merged.mp4 files")
    parser.add_argument("input_gaze_world_dir", help="Directory containing raw gaze/world CSV files")
    parser.add_argument("output_dir", help="Directory to save cut gaze/world CSV files")
    parser.add_argument("pickle_file", help="Pickle file with cut start frame per subject/camera")
    parser.add_argument("--subject-id", help="Optional comma-separated subject IDs (e.g., 27,Mingbo)")
    parser.add_argument("--camera-id", help="Optional comma-separated cameras (e.g., child,parent)")
    parser.add_argument("--log-path", help="Optional log file path")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, "gaze_object.log")
    log_path = Path(args.log_path)
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    logger.info("Logging to: %s", log_path.resolve())

    with open(args.pickle_file, "rb") as f:
        to_cut_frames = pickle.load(f)

    aligner = GazeFrameAligner(
        args.input_video_dir,
        args.input_gaze_world_dir,
        args.output_dir,
        to_cut_frames,
        logger=logger,
    )

    pairs = GazeFrameAligner.discover_pairs(Path(args.input_video_dir))
    if args.subject_id:
        subjects = {x.strip() for x in args.subject_id.split(",") if x.strip()}
        pairs = [(subj, cam) for subj, cam in pairs if subj in subjects]
    if args.camera_id:
        cameras = {x.strip() for x in args.camera_id.split(",") if x.strip()}
        pairs = [(subj, cam) for subj, cam in pairs if cam in cameras]

    if not pairs:
        logger.warning("No subject/camera pairs found in %s", args.input_video_dir)
        return

    for subj, cam in pairs:
        logger.info("Processing subject=%s camera=%s", subj, cam)
        aligner.process_pair(subj, cam)


if __name__ == "__main__":
    main()
