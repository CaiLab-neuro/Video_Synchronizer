# Video-Synchronizer

`Video_Synchronizer` aligns video recordings across cameras using audio spectrogram, then cuts synchronized video/audio segments and maps cut boundaries using world timestamp files. This post-hoc alignment tool is useful when hardware synchronization is not feasible.

This folder contains the synchronization stage of the GBAT workflow. The alignment scripts assume filenames follow a `{subject}_{camera}...` naming pattern, where `{subject}` is a subject ID, and `{camera}` is a camera ID.

## Main Files

- `extract_audios_1.py`: extracts mono WAV audio from input videos with `ffmpeg`
- `video_aligner_2.py`: estimates camera offsets from audio spectrograms, cuts synchronized video/audio segments, exports merged cut videos, and saves cut-frame metadata
- `video_aligner_compression.ipynb`: notebook workflow for recordings that still drift after first alignment; it measures front/middle/back residual offsets, plans a second frame-index cut from the original videos, applies audio tempo correction, and verifies merged outputs
- `gaze_frame_timestamps_cut_3.py`: creates cut-aligned gaze and world timestamp CSV files for downstream gaze-object analysis

Note: `video_aligner_compression.ipynb` reuses the `FrameAligner` implementation. Keep the notebook import cell synchronized with the current aligner module name in this folder (`video_aligner_2.py`).

## Installation

### Requirements

- Python 3.10+
- `ffmpeg`
- `ffprobe`

### Python packages

```bash
pip install numpy pandas scipy matplotlib librosa torch opencv-python
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
- Supported video extensions are `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`.
- The output directory is created if it does not already exist.
- `--sample-rate` is optional. If omitted, the output audio sample rate defaults to `48000` Hz.

Output:

- `{video_stem}.wav`

### 2. Synchronize and cut videos (video_aligner_2.py)

```bash
python video_aligner_2.py <input_video_dir> <input_audio_dir> <output_dir>
```
optional arguments:
- `--input-world-timestamp-dir <path>`
- `--merged-output-dir <path>`
- `--subject-id <e.g. 27,S01>`
- `--camera-id <e.g. child,parent,1>`
- `--ref-cam <e.g. child>`
- `--device <auto|cpu|cuda>`
- `--max-lag-sec 600`
- `--audio-sr <sample_rate>`
- `--spectrogram-sr 400`
- `--n-mels 64`
- `--cut-frames-pkl <path>`
- `--log-path <path>`

- `video_aligner_2.py` expects filenames with consistent naming: `{subject}_{camera_id}.<video_ext>` and `{subject}_{camera_id}.wav`. Supported video extensions are `.mp4`, `.avi`, `.mov`, `.mkv`, and `.webm`. If world timestamp CSVs are provided, they should be named `{subject}_{camera_id}_world_timestamps.csv`.
- The world timestamp CSV is optional for this script (`--input-world-timestamp-dir`). If it is missing, frame boundaries are estimated from video FPS/duration using `ffprobe`.
- If provided, the world timestamp CSV should include each video frame timestamp and must contain column `timestamp [ns]`.
- Subject and camera IDs may be numeric-looking or text. The scripts treat both as filename tokens, so `27`, `S01`, `1`, and `child` are all valid IDs.
- If `--subject-id` is omitted, the script scans `input_video_dir`, takes the token before the first underscore from each video filename, removes duplicates, and processes all discovered subjects.
- If `--camera-id` is omitted, the script scans `input_video_dir`, takes the rest of the filename stem after the first underscore, removes duplicates, and processes all discovered camera IDs.
- Spectrogram synchronization parameters are configurable with `--spectrogram-sr` and `--n-mels`. By default, the script uses each loaded WAV file's sample rate; pass `--audio-sr` to resample all WAV files before synchronization.
- `--device auto` uses CUDA when available and CPU otherwise. CPU synchronization uses SciPy FFT correlation; CUDA synchronization uses PyTorch `conv1d`.
- `video_aligner_2.py` chooses `camera_list[0]` as the reference if `--ref-cam` is not provided. If that reference camera is missing for a subject after waveform loading, it falls back to the first camera actually present in the loaded waveform dictionary.


Outputs:

- `{subject}_{camera}_cut.mp4` in `output_dir`
- `{subject}_{camera}_cut.wav` in `output_dir`
- `{subject}_{camera}_cut_merged.mp4` in `merged_output_dir`
- `audio_offset.pkl` in `output_dir`
- `to_cut_frames.pkl` in `output_dir`, or the path provided with `--cut-frames-pkl`
- `spectrogram_subj{subject}_cam{camera}.png` and `cross_correlation_subj{subject}_cam{camera}_ref{ref_cam}.png` in `output_dir`
- `video_aligner.log` in `output_dir`, or the path provided with `--log-path`


### 3. Align gaze/world CSVs to the cut videos (gaze_frame_timestamps_cut_3.py)

```bash
python 'gaze_frame_timestamps_cut_3.py' <input_cut_video_dir> <input_gaze_world_dir> <output_dir> <pickle_file>
```
Optional arguments:

- `--subject-id <e.g. 27,S01>`
- `--camera-id <e.g. child,parent,1>`
- `--log-path <path>`

- The code expects filenames with consistent naming: `{subject}_{camera_id}_cut_merged.mp4`, `{subject}_{camera_id}_world_timestamps.csv`, and `{subject}_{camera_id}_gaze.csv`.
- `pickle_file` should usually be the `to_cut_frames.pkl` file produced by `video_aligner_2.py`.
- The world timestamp CSV must contain column `timestamp [ns]`.
- The gaze CSV must contain columns `timestamp [ns]`, `gaze x [px]`, and `gaze y [px]`.
- If both <subject_id> and <camera_id> are omitted, the script discovers all `(subject, camera)` pairs by scanning `input_cut_video_dir` for files matching `*_cut_merged.mp4`.


Outputs:

- `{subject}_{camera}_world_timestamps_cut.csv`
- `{subject}_{camera}_gaze_cut.csv`

### 4. Videos go out of sync over time

Some long recordings can exhibit cumulative synchronization drift after initial alignment, for example the child camera appears ahead at the start but lags at the end. Use [video_aligner_compression.ipynb](video_aligner_compression.ipynb) for this case. 

High-level workflow:

- Run or reuse first alignment outputs in `01_first_alignment` and `01_first_alignment_merged`.
- Measure residual front/middle/back audio offsets from the first-cut WAV files.
- Build a second-cut plan from original-video time points using the measured front and back offsets.
- Convert the second-cut time points to original frame indices and extract video frames from the original videos.
- Cut original audio over the same time span and apply `atempo=(span - drift) / span`.
- Merge the second-cut video and tempo-adjusted audio into `{subject}_{camera}_cut2_merged.mp4`.
- Verify merged outputs by measuring front/middle/back residual offsets again.

The notebook reads these environment variables if they are set:

```bash
export GBAT_VIDEO_DIR=/path/to/input_videos
export GBAT_AUDIO_DIR=/path/to/input_audios
export GBAT_SYNC_OUTPUT_DIR=/path/to/output_root
```

Important output folders and files:

- `01_first_alignment/`: first-pass cut videos, cut WAV files, `audio_offset.pkl`, and `to_cut_frames.pkl`
- `01_first_alignment_merged/`: first-pass merged videos
- `diagnostics/residual_front_back_offsets.csv`: residual offsets before second correction
- `02_second_cut_stretch_corrected/second_cut_compression_plan.csv`: second-pass frame/audio plan
- `02_second_cut_stretch_corrected/{subject}_{camera}_cut2_video.mp4`: second-pass video-only frame cut
- `02_second_cut_stretch_corrected/{subject}_{camera}_cut2_audio.wav`: second-pass tempo-adjusted WAV
- `02_second_cut_stretch_corrected/{subject}_{camera}_cut2_merged.mp4`: final merged second-pass output
- `diagnostics/merged_front_back_offsets.csv`: residual offsets after second-pass merge and tempo correction

For short simulated clips with known small offsets, reduce `CONFIG.max_lag_sec` and set `CONFIG.first_ref_cam` explicitly before first alignment. For example:

```python
CONFIG.first_ref_cam = "parent"
CONFIG.audio_sr = 48000
CONFIG.max_lag_sec = 5.0
```

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
