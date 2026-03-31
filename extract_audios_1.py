from pathlib import Path
import subprocess
import argparse

"""
Author: Yanbin Xu
Date: 1/18/2026
Description: Extract audio from either all videos in a folder or one specific video file.

Example:
python extract_audios_1.py <input_path> <output_dir>

Inputs:
    1) input_path:
        - a video directory, in which case all supported video files are processed
        - or a single video file, in which case only that file is processed
    
Outputs:
    1) Extracted audio files: *{video_stem}.wav* in the output directory.
       
"""

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() not in VIDEO_EXTENSIONS:
            raise ValueError(f"Unsupported video file extension: {input_path.suffix}")
        return [input_path]

    if input_path.is_dir():
        videos = sorted(
            p for p in input_path.iterdir()
            if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS
        )
        return videos

    raise FileNotFoundError(f"Input path not found: {input_path}")


def extract_audio_from_video(video_path: Path, output_dir: Path, sample_rate: int) -> None:
    output_path = output_dir / f"{video_path.stem}.wav"
    print(f"Extracting audio from {video_path} to {output_path} at {sample_rate} Hz")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(video_path),
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        str(output_path)
    ]
    subprocess.run(cmd, check=True)


def extract_audios(video_paths: list[Path], output_dir: Path, sample_rate: int) -> None:
    for video_path in video_paths:
        extract_audio_from_video(video_path, output_dir, sample_rate)


def main():
    parser = argparse.ArgumentParser(
        description='Extract audio from all videos in a folder or from one specific video file.'
    )
    parser.add_argument('input_path', help='Video folder or single video file path')
    parser.add_argument('output_dir', help='Folder to save extracted audio files')
    parser.add_argument(
        '--sample-rate',
        type=int,
        default=48000,
        help='Output audio sample rate in Hz (default: 48000)'
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"No output directory found. Created: {output_dir}")

    try:
        video_paths = discover_videos(input_path)
    except (FileNotFoundError, ValueError) as exc:
        print(exc)
        return

    if not video_paths:
        print(f"No video files found in {input_path}")
        return

    if args.sample_rate <= 0:
        print(f"Invalid sample rate: {args.sample_rate}. It must be a positive integer.")
        return

    print(f"Found {len(video_paths)} video file(s) to process.")
    extract_audios(video_paths, output_dir, args.sample_rate)

if __name__ == "__main__":
    main()

