# Video_Synchronizer

`Video_Synchronizer` aligns video recordings across cameras using audio spectrogram, then cut gaze/world timestamp files based on synchronized cut boundaries. This post-hoc alignment tool is useful when hardware synchronization is not feasible.

This folder contains the synchronization stage of the GBAT workflow. The alignment scripts assume filenames follow a `{subject}_{camera}...` naming pattern, where `{subject}` is a subject ID, and `{camera}` is a camera ID.

## Main Scripts

- `extract_audios_1.py`: extracts mono WAV audio from input videos with `ffmpeg`
- `video_aligner_publish_2.py`: estimates camera offsets from audio spectrograms, cuts synchronized video/audio segments, and exports merged cut videos
- `gaze_frame_alignment_in_cut_3.py`: creates cut-aligned gaze and world timestamp CSV files for downstream gaze-object analysis

## Installation

### Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`

### Python packages

```bash
pip install numpy pandas matplotlib librosa torch opencv-python
```

## Usage

### 1. Extract audio (extract_audios_1.py)

```bash
python extract_audios_1.py <input_path> <output_audio_dir> --sample-rate <sample_rate>
```

Examples:

```bash
python extract_audios_1.py C:\path\to\video_folder C:\path\to\audio_out
python extract_audios_1.py C:\path\to\one_video.mp4 C:\path\to\audio_out
python extract_audios_1.py C:\path\to\video_folder C:\path\to\audio_out --sample-rate 44100
```

- If `input_path` is a directory, the script scans that folder and processes all supported video files it finds.
- If `input_path` is a single video file, the script extracts audio only from that file.
- `--sample-rate` is optional. If omitted, the output audio sample rate defaults to `48000` Hz.

Output:

- `{video_stem}.wav`

### 2. Synchronize and cut videos (video_aligner_publish_2.py)

```bash
python video_aligner_publish_2.py <input_video_dir> <input_audio_dir> <input_gaze_world_dir> <output_dir> 
```
optional arguments:
- `--merged-output-dir <path>`
- `--subject-id <e.g. 27,28>`
- `--camera-id <e.g. child,parent,side>`
- `--ref-cam <e.g. child>``
- `--device <auto|cpu|cuda>`
- `--max-lag-sec 600`
- `--cut-frames-pkl <path>`
- `--log-path <path>`

-`video_aligner_publish_2.py` expects filenames with consistent naming: {subject}_{camera_id}.mp4, {subject}_{camera_id}.wav, {subject}_{camera_id}_world_timestamps.csv, and {subject}_{camera_id}_gaze.csv
- If `--subject-id` is omitted, the script scans `input_video_dir`, takes the first underscore-separated token from each video filename, converts it to an integer, removes duplicates with `np.unique`, and processes all discovered subjects.
- If `--camera-id` is omitted, the script scans `input_video_dir`, takes the second underscore-separated token from each video filename, removes duplicates with `np.unique`, and processes all discovered camera IDs.
- The world timestamp CSV should include the timestamp of each frame of the video. The gaze CSV should include the timestamp of each gaze. They do not need to have the same temporal resolution.
- The world timestamp CSV must contain column `timestamp [ns]`. The gaze CSV must contain at least: `timestamp [ns]', `gaze x [px]`, `gaze y [px]`
- `video_aligner_publish_2.py` chooses `camera_list[0]` as the reference if `--ref-cam` is not provided. If that reference camera is missing for a subject after waveform loading, it falls back to the first camera actually present in the loaded waveform dictionary.


Outputs:

- cut videos in `output_dir`
- cut WAV files in `output_dir`
- merged synchronized videos in `merged_output_dir`
- `audio_offset.pkl`
- `to_cut_frames.pkl`
- spectrogram and cross-correlation figures
- log file

### 3. Align gaze/world CSVs to the cut videos (gaze_frame_alignment_in_cut_3.py)

```bash
python gaze_frame_alignment_in_cut_3.py <input_cut_video_dir> <input_gaze_world_dir> <output_dir> <pickle_file>
```
Optional arguments:

- `--subject-id <e.g. 27,28>`
- `--camera-id <e.g. child,parent>`
- `--log-path <path>`

- The code expects filenames with consistent naming: {subject}_{camera_id}_cut_merged.mp4, {subject}_{camera_id}_world_timestamps.csv, and {subject}_{camera_id}_gaze.csv
- If both <subject_id> and <camera_id> are omitted, the script discovers all `(subject, camera)` pairs by scanning `input_cut_video_dir` for files matching `*_cut_merged.mp4`.
- `gaze_frame_alignment_in_cut_3.py` can count frames with `ffprobe` and falls back to OpenCV decoding if needed.

Outputs:

- `{subject}_{camera}_world_timestamps_cut.csv`
- `{subject}_{camera}_gaze_cut.csv`

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
