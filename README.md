# Video_Synchronizer

`Video_Synchronizer` aligns multimodal recordings across cameras using audio, then maps the synchronized cut boundaries back to gaze/world timestamp files.

This folder contains the synchronization stage of the GBAT workflow. It is designed for datasets where videos follow a `{subject}_{camera}.mp4` naming pattern and where gaze/world CSV files are available for the egocentric cameras.

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
The scripts expect filenames like:

- `{subject}_child.mp4`
- `{subject}_parent.mp4`
- `{subject}_side.mp4`

### Gaze/world CSVs
`video_aligner_publish_2.py` and `gaze_frame_alignment_in_cut_3.py` expect:

- `{subject}_{camera}_world_timestamps.csv`
- `{subject}_{camera}_gaze.csv`

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
python extract_audios_1.py <input_video_dir> <output_audio_dir>
```

Optional:

```bash
python extract_audios_1.py <input_video_dir> <output_audio_dir> --subject-ids 27,28
```

Output:

- `{subject}_{camera}.wav`

### 2. Synchronize and cut videos

```bash
python video_aligner_publish_2.py <input_video_dir> <input_audio_dir> <input_gaze_world_dir> <output_dir> --merged-output-dir <merged_output_dir>
```

Common optional arguments:

- `--subject-id 27,28`
- `--camera-id child,parent,side`
- `--ref-cam child`
- `--device auto|cpu|cuda`
- `--max-lag-sec 600`
- `--cut-frames-pkl <path>`
- `--log-path <path>`

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

Common optional arguments:

- `--subject-id 27,28`
- `--camera-id child,parent`
- `--log-path <path>`

Outputs include:

- `{subject}_{camera}_world_timestamps_cut.csv`
- `{subject}_{camera}_gaze_cut.csv`

## Notes

- `video_aligner_publish_2.py` chooses the first available camera as the reference if `--ref-cam` is not provided.
- The synchronizer saves cut-frame metadata to a pickle file; `gaze_frame_alignment_in_cut_3.py` uses that file to remap timestamps to the cut videos.
- `gaze_frame_alignment_in_cut_3.py` can count frames with `ffprobe` and falls back to OpenCV decoding if needed.
