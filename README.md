# Video_Synchronizer

`Video_Synchronizer` aligns multimodal recordings across cameras using audio, then maps the synchronized cut boundaries back to gaze/world timestamp files.

This folder contains the synchronization stage of the GBAT workflow. The alignment scripts assume filenames follow a `{subject}_{camera}...` naming pattern, where `{camera}` is a camera ID string such as `child`, `parent`, or `side`. Those names are examples, not the only camera IDs accepted by the code.

## Main Scripts

- `extract_audios_1.py`: extracts mono WAV audio from input videos with `ffmpeg`
- `video_aligner_publish_2.py`: estimates camera offsets from audio spectrograms, cuts synchronized video/audio segments, and exports merged cut videos
- `gaze_frame_alignment_in_cut_3.py`: creates cut-aligned gaze and world timestamp CSV files for downstream gaze-object analysis

## Pipeline

1. Extract audio from the original videos.
2. Compute cross-camera temporal offsets from log-mel spectrograms.
3. Trim each recording to the shared overlapping segment.
4. Cut the corresponding video and audio with `ffmpeg`.
5. Merge cut video and cut audio into synchronized outputs.
6. Use the saved cut-frame mapping to generate cut-aligned gaze/world CSV files.

## Expected Inputs

### Videos
`video_aligner_publish_2.py` expects filenames like:

- `{subject}_{camera_id}.mp4`

Important:

- In `video_aligner_publish_2.py`, camera IDs are handled as strings taken from the second underscore-separated token in the filename.
- The code does not enforce that camera IDs must be exactly `child`, `parent`, or `side`; those are example names used in this README.
- The key requirement is consistent naming across related files, for example `{subject}_room.mp4`, `{subject}_room.wav`, `{subject}_room_world_timestamps.csv`, and `{subject}_room_gaze.csv`.

`extract_audios_1.py` accepts any video filename ending in one of:

- `.mp4`
- `.avi`
- `.mov`
- `.mkv`
- `.webm`

### Gaze/world CSVs
`video_aligner_publish_2.py` and `gaze_frame_alignment_in_cut_3.py` expect:

- `{subject}_{camera}_world_timestamps.csv`
- `{subject}_{camera}_gaze.csv`

Here, `{camera}` is also matched as a string from filenames rather than from a hard-coded fixed set.

The world timestamp CSV should include the timestamp of each frame of the video. The gaze CSV should include the timestamp of each gaze. They do not need to have the same temporal resolution.

The world timestamp CSV must contain `timestamp [ns]`. The gaze CSV must contain at least:

- `timestamp [ns]`
- `gaze x [px]`
- `gaze y [px]`

## Installation

Use a dedicated Python environment, then install the libraries imported by the scripts.

### Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`

### Python packages

```bash
pip install numpy pandas matplotlib librosa torch opencv-python
```

## Usage

### 1. Extract audio

```bash
python extract_audios_1.py <input_path> <output_audio_dir>
```

Optional:

```bash
python extract_audios_1.py <input_path> <output_audio_dir> --sample-rate 16000
```

Examples:

```bash
python extract_audios_1.py C:\path\to\video_folder C:\path\to\audio_out
python extract_audios_1.py C:\path\to\one_video.mp4 C:\path\to\audio_out
python extract_audios_1.py C:\path\to\video_folder C:\path\to\audio_out --sample-rate 44100
```

How `extract_audios_1.py` works:

- `input_path` can be either a directory or a single video file.
- If `input_path` is a directory, the script scans that folder and processes all supported video files it finds.
- If `input_path` is a single video file, the script extracts audio only from that file.
- `--sample-rate` is optional. If omitted, the output audio sample rate defaults to `48000` Hz.
- The output audio filename is `{video_stem}.wav`, so for example `session_A.mp4` becomes `session_A.wav`.

Output:

- `{video_stem}.wav`

### 2. Synchronize and cut videos

```bash
python video_aligner_publish_2.py <input_video_dir> <input_audio_dir> <input_gaze_world_dir> <output_dir> --merged-output-dir <merged_output_dir>
```

Common examples of optional arguments:

- `--subject-id 27,28`
- `--camera-id child,parent,side`
- `--ref-cam child`
- `--device auto|cpu|cuda`
- `--max-lag-sec 600`
- `--cut-frames-pkl <path>`
- `--log-path <path>`

- If `--subject-id` is omitted, the script scans `input_video_dir`, takes the first underscore-separated token from each video filename, converts it to an integer, removes duplicates with `np.unique`, and processes all discovered subjects.
- If `--camera-id` is omitted, the script scans `input_video_dir`, takes the second underscore-separated token from each video filename, removes duplicates with `np.unique`, and processes all discovered camera IDs.
- `--camera-id` accepts camera ID strings. `child,parent,side` are examples, not the only values supported by the code.

Outputs include:

- cut videos in `output_dir`
- cut WAV files in `output_dir`
- merged synchronized videos in `merged_output_dir`
- `audio_offset.pkl`
- `to_cut_frames.pkl`
- spectrogram and cross-correlation figures
- log file

### 3. Align gaze/world CSVs to the cut videos

```bash
python gaze_frame_alignment_in_cut_3.py <input_cut_video_dir> <input_gaze_world_dir> <output_dir> <pickle_file>
```

Common examples of optional arguments:

- `--subject-id 27,28`
- `--camera-id child,parent`
- `--log-path <path>`

- If both are omitted, the script discovers all `(subject, camera)` pairs by scanning `input_cut_video_dir` for files matching `*_cut_merged.mp4`.
- Each filename is parsed as `{subject}_{camera}_cut_merged.mp4`, the discovered pairs are deduplicated, and `sorted(pairs)` is used, so the default processing order is the sorted list of all available subject/camera pairs.
- If only one filter is provided, the script first discovers all pairs and then keeps only the pairs matching the given subjects or cameras.
- `--camera-id` accepts camera ID strings. `child,parent` are examples, not hard-coded required values.

Outputs include:

- `{subject}_{camera}_world_timestamps_cut.csv`
- `{subject}_{camera}_gaze_cut.csv`

## Notes

- `extract_audios_1.py` accepts either a folder of videos or one specific video file, supports `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`, and defaults to a `48000` Hz output sample rate.
- `video_aligner_publish_2.py` defaults to all discovered subjects and all discovered camera IDs from the input video filenames.
- Across `video_aligner_publish_2.py` and `gaze_frame_alignment_in_cut_3.py`, camera IDs are generally treated as strings parsed from filenames rather than as a hard-coded fixed set. The main requirement is consistent camera naming across matching video, audio, gaze, and world files.
- `video_aligner_publish_2.py` chooses `camera_list[0]` as the reference if `--ref-cam` is not provided. If that reference camera is missing for a subject after waveform loading, it falls back to the first camera actually present in the loaded waveform dictionary.
- `gaze_frame_alignment_in_cut_3.py` defaults to all discovered `(subject, camera)` pairs from `*_cut_merged.mp4` files in the cut-video directory.
- The synchronizer saves cut-frame metadata to a pickle file; `gaze_frame_alignment_in_cut_3.py` uses that file to remap timestamps to the cut videos.
- `gaze_frame_alignment_in_cut_3.py` can count frames with `ffprobe` and falls back to OpenCV decoding if needed.

## Citation

If you use the tool, please cite:

```bibtex
@misc{baig2026gazebehaviorannotationtoolkitgbat,
      title={GazeBehavior Annotation Toolkit (GBAT): AI-powered toolkit for automatic annotation of egocentric eye-tracking and video data of child-caregiver interaction}, 
      author={Iba Baig and Kevin Li and Yanbin Xu and Seiji Cattelain and Marie Hallo and Hayato Ono and Sho Tsuji and Ming Bo Cai},
      year={2026},
      eprint={2605.22962},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.22962}, 
}
```
