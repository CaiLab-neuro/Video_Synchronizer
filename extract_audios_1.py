from pathlib import Path
import subprocess
import argparse

"""
Author: Yanbin Xu
Date: 1/18/2026
Description: Extract audio from multiple videos for multiple subjects and camera types.
If you want to process all subjects, do not provide --subject-ids argument.
Else, provide a comma-separated list of subject IDs to process only those subjects.

Example:
python extract_audios.py <input_dir> <output_dir> [--subject-ids <subject_id>] [--log-path <log_file_path>]

Inputs:
    1) Subject ID(s) and Camera ID(s).
    Subject ID: e.g. 27 or 27,28 etc.
    Camera ID: e.g. child or parent or child,parent

    2) Video directory: /path/to/video_data
    Within this directory, the code expects
        a) videos: *{subject_id}_{camera}.mp4*
    
Outputs:
    1) Extracted audio files: *{subject_id}_{camera}.wav* in the output directory.
       
"""

def extract_audios(subj_ids, cameras, input_dir, output_dir):
    for subj in subj_ids:
        for cam in cameras:
            video_path = Path(f"{input_dir}/{subj}_{cam}.mp4")
            output_path = Path(f"{output_dir}/{subj}_{cam}.wav")
            if not video_path.exists():
                print(f"[skip] Subject {subj}, camera {cam}: video not found: {video_path}")
                continue
            # Build ffmpeg command
            print (f"Extracting audio from {video_path} to {output_path}")

            # FFmpeg command
            cmd = [
                "ffmpeg",
                "-y",                # overwrite output file if exists
                "-i", str(video_path),    # input file
                "-vn",               # no video
                "-c:a", "pcm_s16le", # uncompressed 16-bit PCM
                "-ar", "48000",      # set audio sample rate to 48000 Hz
                "-ac", "1",          # set number of audio channels to 1 (mono)
                str(output_path)
            ]
            # Run the command
            subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description='Extract audio from videos for multiple subjects and camera types.')
    parser.add_argument('input_dir', help='Root folder containing videos')
    parser.add_argument('output_dir', help='Folder to save cut videos and audios')
    parser.add_argument('--subject-ids', help='Process only specific subject IDs e.g. 1,2 or 1)')
    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"Input Directory Not Found: {args.input_dir}")
        return
    if not Path(args.output_dir).exists():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        print(f"No Output Directory Found, Created output directory: {args.output_dir}")

    if args.subject_ids:
        subject_ids = [int(sid) for sid in args.subject_ids.split(',')]
        print(f"Processing only specific subject IDs: {subject_ids}")
    else:
        video_dir = Path(args.input_dir)
        subject_ids = [
            int(p.stem.split('_')[0]) for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ]
        subject_ids = sorted(set(subject_ids))

    if not subject_ids:
        print("No subject IDs found in the input directory.")
        return  

    extract_audios(
        subj_ids=subject_ids,
        cameras=['child', 'parent', 'side'],
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )

if __name__ == "__main__":
    main()

