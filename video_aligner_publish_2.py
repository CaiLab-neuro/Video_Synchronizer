import subprocess
import os
import pickle
import math
import sys
import logging
import time
from pathlib import Path
import os.path as osp
import numpy as np
import librosa
import torch
import matplotlib.pyplot as plt
import argparse
import pandas as pd
	        

"""
Align multiple camera videos based on audio synchronization and cut them based on frame indices 

Example Use: 
to process a single subject with specific camera types:
python video_aligner_publish.py /path/to/input_videos /path/to/input_audios /path/to/gaze_world_data /path/to/output_dir --merged-output-dir /path/to/merged_output_dir --subject-id 1 --camera-id child,parent

Inputs:
    1) Subject ID(s) and Camera ID(s).
    Subject ID: e.g. 27 or 27,28 etc.
    Camera ID: e.g. child or parent or child,parent

    2) Video directory: /path/to/video_data
    Within this directory, the code expects
        a) videos: *{subject_id}_{camera}.mp4*

    3) Gaze world data directory: /path/to/gaze_world_data
    Within this directory, the code expects
            a) world_timestamps of videos named as '{subj_ID}_{camera_type}_world_timestamps.csv'
            b) gaze data named as '{subj_ID}_{camera_type}_gaze.csv'
            
    4) output_dir: Directory to save cut videos and audios.
    5) merged_output_dir: Directory to save merged videos with audio (finalized version).
    6) ref_cam: Reference camera type for synchronization (optional). The default is the first camera in the list.
    7) subject-id: Specific subject ID to process (optional). The default is to process all subjects found in the input directory. We expect IDs to be integers.
    8) camera-id: Specific camera types to process (optional). The default is to process all camera types found in the input directory. We expect camera types to be strings.

Outputs:
- Cut videos and audios saved in output_dir.
- Merged videos with audio saved in merged_output_dir.

"""


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


def _resolve_torch_device(requested: str, logger: logging.Logger) -> tuple[torch.device, str]:
    requested_norm = requested.strip().lower()
    if requested_norm not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"Unsupported device option: {requested}. Use one of: auto, cpu, cuda.")

    if requested_norm == "cpu":
        return torch.device("cpu"), requested_norm

    if requested_norm == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda"), requested_norm
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu"), requested_norm

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda"), requested_norm
    return torch.device("cpu"), requested_norm


class FrameAligner:
    def __init__(
        self,
        input_video_dir,
        input_audio_dir,
        input_gaze_world_dir,
        output_dir,
        merged_output_dir,
        logger=None,
        device: torch.device | None = None,
        requested_device: str = "auto",
        max_lag_sec: float = 60.0,
    ):
        """
        Parameters
        ----------
        input_video_dir : str, Original video directory.
        input_audio_dir : str, Directory containing audio files.
        input_gaze_world_dir : str, Directory containing gaze world data.
        output_dir : str, Directory to save cut videos.
        merged_output_dir : str, Directory to save merged videos.
        ref_cam : str, Reference camera type for synchronization.
        """
        self.input_video_dir = input_video_dir
        self.input_audio_dir = input_audio_dir
        self.input_gaze_world_dir = input_gaze_world_dir
        self.output_dir = output_dir
        self.merged_output_dir = merged_output_dir
        self.logger = logger or logging.getLogger(__name__)
        self.device = device or torch.device("cpu")
        self.requested_device = requested_device
        self.max_lag_sec = float(max_lag_sec)
        self.to_cut_frames_all: dict[int, dict[str, list[float | int]]] = {}
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.merged_output_dir, exist_ok=True)

    def load_and_resample_np(self, file_path, target_sr=None):
        """ Load and resample audio to a target sample rate. """
        y_orig, sr_orig = librosa.load(file_path, sr=None)
        if target_sr is not None and sr_orig != target_sr:
            y = librosa.resample(y_orig, orig_sr=sr_orig, target_sr=target_sr)
            sr_used = target_sr
        else:
            y = y_orig
            sr_used = sr_orig
        self.logger.info(
            "Loaded audio=%s sr_orig=%s sr_target=%s sr_used=%s len_resampled=%s len_original=%s",
            Path(file_path).name,
            sr_orig,
            target_sr,
            sr_used,
            np.shape(y),
            np.shape(y_orig),
        )
        return y, sr_used

    def _load_subject_waveforms(self, subj_ID, camera_list, audio_sr=48000):
        files = {cam: f"{self.input_audio_dir}/{subj_ID}_{cam}.wav" for cam in camera_list}
        waveforms = {}
        sr_ref = None

        t0 = time.perf_counter()
        for cam in camera_list:
            if osp.isfile(files[cam]):
                wav, sr = self.load_and_resample_np(files[cam], target_sr=audio_sr)
                waveforms[cam] = wav
                if sr_ref is None:
                    sr_ref = sr
                else:
                    assert sr_ref == sr, "Sample rates must match after resampling!"
        elapsed = time.perf_counter() - t0

        self.logger.info(
            "Loaded waveforms subject=%s cameras=%s elapsed_sec=%.3f",
            subj_ID,
            list(waveforms.keys()),
            elapsed,
        )
        if not waveforms:
            raise FileNotFoundError(f"No existing .wav found for subject {subj_ID} in camera_list.")
        return waveforms, sr_ref, elapsed
    
    def time_to_frame(self, subj_ID, camera_type, cut_start_sec, cut_end_sec, gaze_world_dir):
        """ 
        This function outputs the frame index of the nearest frame corresponding to the given time in seconds at the front and beginning of the cut segment.

        Args:
            subj_ID (int): Subject ID number, e.g., 27, 28, etc.
            camera_type (str): Type of camera, ('child' or 'parent')
            time_in_seconds (float): Time in seconds from to_cut info.

        Returns:
            start_frame (int): Frame index corresponding to the given time in seconds.
            end_frame (int): Frame index corresponding to the given time in seconds.
        """
        saved_timestamps= pd.read_csv(f'{gaze_world_dir}/{subj_ID}_{camera_type}_world_timestamps.csv')
        saved_timestamps['timestamp_s'] = saved_timestamps['timestamp [ns]'] / 1e9
        saved_timestamps['duration_s'] = saved_timestamps['timestamp_s'].shift(-1) - saved_timestamps['timestamp_s']
        saved_timestamps['relative_time'] = saved_timestamps['timestamp_s'] - saved_timestamps['timestamp_s'].iloc[0]

        start_frame_idx = (saved_timestamps['relative_time'] - cut_start_sec).abs().idxmin()
        end_frame_idx = (saved_timestamps['relative_time'] - (saved_timestamps.iloc[-1]['relative_time'] - cut_end_sec)).abs().idxmin()
        start_frame_relative_time = saved_timestamps.loc[start_frame_idx, 'relative_time']
        end_frame_relative_time = saved_timestamps.loc[end_frame_idx, 'relative_time']
        self.logger.info(
            "subject=%s camera=%s start_cut_sec=%.6f -> start_frame=%d frame_time=%.6f",
            subj_ID, camera_type, cut_start_sec, start_frame_idx, start_frame_relative_time
        )
        self.logger.info(
            "subject=%s camera=%s end_cut_sec=%.6f -> end_frame=%d frame_time=%.6f",
            subj_ID, camera_type, cut_end_sec, end_frame_idx, end_frame_relative_time
        )
        start_frame = start_frame_idx 
        end_frame = end_frame_idx
        return int(start_frame), int(end_frame), start_frame_relative_time, end_frame_relative_time
    
    def compute_log_mel_spectrogram(self, audio, sr, n_mels=64, hop_length=512):
        """
        Compute the log-mel spectrogram of an audio signal.

        Args:
            audio (np.ndarray): Audio time series.
            sr (int): Sample rate of the audio.
            n_mels (int): Number of mel bands to generate.
            hop_length (int): Hop length for STFT.

        Returns:
            np.ndarray: Log-mel spectrogram.
        """
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        log_S = librosa.power_to_db(S, ref=np.max)
        return log_S

    def compute_time_shift(self, x, y, pad_size=120000, max_lag_sec=None, spectrograph_sr=400):
        """
        Compute the time shift based on two spectrograms using cross-correlation.

        Args:
            x (torch.Tensor): Spectrogram of the first audio signal (shape: [1, n_mels, time_frames]).
            y (torch.Tensor): Spectrogram of the second audio signal (shape: [1, n_mels, time_frames]).
            pad_size (int): Number of samples to pad on each side for cross-correlation.

        Returns:
            int: Time shift in samples (positive means y lags behind x, negative means y leads x).
            torch.Tensor: Cross-correlation values for each lag.
        """
        def _to_3d_spec(tensor, name):
            if tensor.ndim == 2:
                return tensor.unsqueeze(0)
            if tensor.ndim == 3:
                return tensor
            raise ValueError(
                f"{name} must have shape [n_mels, t] or [1, n_mels, t], got {tuple(tensor.shape)}"
            )

        x = _to_3d_spec(x, "x")
        y = _to_3d_spec(y, "y")
        if x.shape[1] != y.shape[1]:
            raise ValueError(
                f"Channel mismatch: x has {x.shape[1]} mel bins, y has {y.shape[1]} mel bins"
            )
        if x.shape[-1] <= 0 or y.shape[-1] <= 0:
            raise ValueError(
                f"Invalid spectrogram length: x_t={x.shape[-1]} y_t={y.shape[-1]}"
            )

        if max_lag_sec is not None:
            pad_size = int(float(max_lag_sec) * float(spectrograph_sr))
        if pad_size < 0:
            raise ValueError(f"pad_size must be non-negative, got {pad_size}")

        pad_size_end = np.max([y.shape[-1] - x.shape[-1], 0]) + pad_size
        x_padded = torch.nn.functional.pad(x, (pad_size, pad_size_end), mode="constant", value=0)
        one_x = torch.ones_like(x)
        one_y = torch.ones_like(y)
        one_x_padded = torch.nn.functional.pad(one_x, (pad_size, pad_size_end), mode="constant", value=0)
        
        corr = torch.nn.functional.conv1d(x_padded, y, padding=0)
        
        corr_default = torch.nn.functional.conv1d(one_x_padded, one_y, padding=0)
        corr = corr / corr_default
        # Find lag (best alignment)
        lag_samples = torch.argmax(corr) - pad_size
        return lag_samples.item(), corr
    
    def spectrogram_synchronizer(
        self,
        subj_ID,
        camera_list,
        spectrograph_sr=400,
        audio_sr=48000,
        ref_cam=None,
        waveforms=None,
        sr_ref=None,
        max_lag_sec=None,
    ):
        if max_lag_sec is None:
            max_lag_sec = self.max_lag_sec

        if waveforms is None or sr_ref is None:
            waveforms, sr_ref, _ = self._load_subject_waveforms(subj_ID, camera_list, audio_sr=audio_sr)

        if ref_cam is None:
            ref_cam = camera_list[0]
        if ref_cam not in waveforms:
            ref_cam = next(iter(waveforms.keys()))
            self.logger.warning(
                "Requested ref_cam missing for subject=%s. Using fallback ref_cam=%s",
                subj_ID,
                ref_cam,
            )

        hop_length = int(sr_ref / spectrograph_sr)
        pad_size = int(float(max_lag_sec) * float(spectrograph_sr))
        self.logger.info(
            "subject=%s ref_cam=%s spectrogram_hop=%d max_lag_sec=%.3f pad_size=%d",
            subj_ID,
            ref_cam,
            hop_length,
            float(max_lag_sec),
            pad_size,
        )

        timing = {"mel_ref": 0.0, "mel_compare": 0.0, "correlation": 0.0}
        t_ref = time.perf_counter()
        log_mel_ref_np = self.compute_log_mel_spectrogram(
            waveforms[ref_cam], sr_ref, n_mels=64, hop_length=hop_length
        )
        log_mel_ref_base = torch.tensor(log_mel_ref_np, dtype=torch.float32, device=self.device)
        log_mel_ref_base = (log_mel_ref_base - log_mel_ref_base.mean(dim=-1, keepdim=True))
        timing["mel_ref"] = time.perf_counter() - t_ref

        time_lags = {subj_ID: {}}
        for key in camera_list:
            if key == ref_cam or key not in waveforms:
                continue

            t_cmp = time.perf_counter()
            log_mel_compare_np = self.compute_log_mel_spectrogram(
                waveforms[key], sr_ref, n_mels=64, hop_length=hop_length
            )
            log_mel_compare = torch.tensor(log_mel_compare_np, dtype=torch.float32, device=self.device)
            log_mel_compare = (log_mel_compare - log_mel_compare.mean(dim=-1, keepdim=True))
            timing["mel_compare"] += time.perf_counter() - t_cmp

            log_mel_ref = log_mel_ref_base
            self.logger.info(
                "subject=%s ref_cam=%s cam=%s; spectrogram_shapes: ref=%s cmp=%s",
                subj_ID, ref_cam, key, tuple(log_mel_ref.shape), tuple(log_mel_compare.shape)
            )

            t_corr = time.perf_counter()
            try:
                with torch.inference_mode():
                    sample_shift, corr = self.compute_time_shift(
                        log_mel_ref,
                        log_mel_compare,
                        max_lag_sec=max_lag_sec,
                        spectrograph_sr=spectrograph_sr,
                    )
            except RuntimeError as e:
                err_msg = str(e)
                if self.device.type == "cuda" and self.requested_device in {"auto", "cuda"}:
                    self.logger.warning(
                        "CUDA conv1d failed for subject=%s cam=%s (%s). Falling back to CPU.",
                        subj_ID,
                        key,
                        err_msg.splitlines()[0] if err_msg else "runtime error",
                    )
                    log_mel_ref = log_mel_ref.to("cpu")
                    log_mel_compare = log_mel_compare.to("cpu")
                    self.device = torch.device("cpu")
                    with torch.inference_mode():
                        sample_shift, corr = self.compute_time_shift(
                            log_mel_ref,
                            log_mel_compare,
                            max_lag_sec=max_lag_sec,
                            spectrograph_sr=spectrograph_sr,
                        )
                else:
                    raise
            timing["correlation"] += time.perf_counter() - t_corr

            time_shift_s = sample_shift / spectrograph_sr
            time_lags[subj_ID][key] = time_shift_s
            self.logger.info(
                "subject=%s ref_cam=%s cam=%s; delay_sec=%.6f",
                subj_ID, ref_cam, key, time_shift_s
            )

            # Plot and save both spectrograms (reference and compared camera).
            mel_vmin = min(float(np.min(log_mel_ref_np)), float(np.min(log_mel_compare_np)))
            mel_vmax = max(float(np.max(log_mel_ref_np)), float(np.max(log_mel_compare_np)))
            ref_duration_sec = log_mel_ref_np.shape[1] / float(spectrograph_sr)
            cmp_duration_sec = log_mel_compare_np.shape[1] / float(spectrograph_sr)

            plt.figure(figsize=(10, 6), dpi=300)
            ref_im = plt.imshow(
                log_mel_ref_np,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[0, ref_duration_sec, 0, log_mel_ref_np.shape[0]],
                vmin=mel_vmin,
                vmax=mel_vmax,
            )
            # plt.title(f"Reference Spectrogram ({ref_cam})")
            # plt.xlabel("Time (seconds)")
            # plt.ylabel("Mel Bin")
            plt.colorbar(ref_im, label="Power (dB)")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/spectrogram_subj{subj_ID}_cam{ref_cam}.png")
            plt.close()

            plt.figure(figsize=(10, 6), dpi=300)
            cmp_im = plt.imshow(
                log_mel_compare_np,
                aspect="auto",
                origin="lower",
                interpolation="nearest",
                extent=[0, cmp_duration_sec, 0, log_mel_compare_np.shape[0]],
                vmin=mel_vmin,
                vmax=mel_vmax,
            )
            # plt.title(f"Compared Spectrogram ({key})")
            # plt.xlabel("Time (seconds)")
            # plt.ylabel("Mel Bin")
            plt.colorbar(cmp_im, label="Power (dB)")
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/spectrogram_subj{subj_ID}_cam{key}.png")
            plt.close()

            # Plot and save cross-correlation with axis labels.
            corr_np = corr.detach().cpu().numpy().flatten()
            lag_axis_sec = (np.arange(corr_np.shape[0]) - pad_size) / float(spectrograph_sr)
            plt.figure(figsize=(10, 6), dpi=300)
            plt.plot(lag_axis_sec, corr_np)
            plt.xlabel("Lag (seconds)")
            plt.ylabel("Normalized Correlation")
            plt.axvline(x=time_shift_s, color='r', linestyle='--', label=f"Best Lag: {time_shift_s:.3f} sec")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/cross_correlation_subj{subj_ID}_cam{key}_ref{ref_cam}.png")
            plt.close()

        with open(f'{self.output_dir}/audio_offset.pkl', 'wb') as f:
            pickle.dump({'time_lags': time_lags, 'unit': 'sec', 'note': 'negative indicates the sound of the recording precedes that of the child'}, f)
        self.logger.info(
            "Saved audio offsets: %s | timing_sec={mel_ref:%.3f, mel_compare:%.3f, correlation:%.3f}",
            f"{self.output_dir}/audio_offset.pkl",
            timing["mel_ref"],
            timing["mel_compare"],
            timing["correlation"],
        )
        return time_lags[subj_ID]

    def time_lag_to_cut_frames(
        self,
        subj_ID,
        camera_list,
        time_lags,
        audio_sr=48000,
        ref_cam=None,
        waveforms=None,
        sr_ref=None,
    ):
        cutted_all_soundwaves = {}
        to_cut = {}

        if waveforms is None or sr_ref is None:
            waveforms, sr_ref, _ = self._load_subject_waveforms(subj_ID, camera_list, audio_sr=audio_sr)
        # Choose reference camera
        if ref_cam is None:
            ref_cam = next(iter(waveforms.keys()))  # first loaded camera
        if ref_cam not in waveforms:
            raise ValueError(f"ref_cam='{ref_cam}' not found among loaded cameras: {list(waveforms.keys())}")
        for cam in waveforms.keys():
            self.logger.info(
                "subject=%s camera=%s waveform_length=%.2f sr=%d",
                subj_ID, cam, len(waveforms[cam]) / sr_ref, sr_ref
            )
        start_time_sec = {ref_cam: 0.0}
        for cam in waveforms.keys():
            if cam == ref_cam:
                continue
            if cam in time_lags:
                start_time_sec[cam] = float(time_lags[cam])

            else:
                # Alternative: skip this camera or raise an error.
                self.logger.warning(
                    "No time lag found for subject=%s camera=%s. Skipping this camera.",
                    subj_ID, cam
                )
                continue

        # Common start is the latest start across cameras (>= 0 because ref_cam is 0)
        common_start_sec = max(start_time_sec.values())

        # Start trims in samples
        start_trim_samples = {}
        for cam in waveforms.keys():
            trim_sec = common_start_sec - start_time_sec.get(cam)
            trim_samp = int(np.round(trim_sec * sr_ref))
            start_trim_samples[cam] = trim_samp

        # Apply start trims, then equalize end by trimming to shortest remaining length
        trimmed = {}
        remaining_lengths = []
        for cam, wav in waveforms.items():
            s0 = start_trim_samples[cam]
            w = wav[s0:]
            trimmed[cam] = w
            remaining_lengths.append(len(w))

        min_len = min(remaining_lengths)
        logging.info("subject=%s min_trimmed_length=%d", subj_ID, min_len)
        # Initialize to_cut in samples: [start_cut, end_cut]
        to_cut[subj_ID] = {cam: [0, 0] for cam in waveforms.keys()}

        for cam in trimmed.keys():
            # end cut = extra samples beyond min_len
            end_cut = len(trimmed[cam]) - min_len
            to_cut[subj_ID][cam][0] = start_trim_samples[cam]
            to_cut[subj_ID][cam][1] = end_cut
            # apply end trim
            trimmed[cam] = trimmed[cam][:min_len]

        for cam in to_cut[subj_ID]:
            to_cut[subj_ID][cam][0] = to_cut[subj_ID][cam][0] / sr_ref
            to_cut[subj_ID][cam][1] = to_cut[subj_ID][cam][1] / sr_ref

        cutted_all_soundwaves[subj_ID] = trimmed

        lengths = {cam: len(wav) for cam, wav in trimmed.items()}
        if len(set(lengths.values())) != 1:
            raise AssertionError(f"Durations not equal after trimming: {lengths}")
        
        # Convert cut times in seconds to frame indices and relative times
        to_cut_frames = {}
        to_cut_frames[subj_ID] = {}
        frame_map_elapsed = 0.0
        for camera in camera_list:
            if camera in to_cut[subj_ID]:
                cut_start_time = to_cut[subj_ID][camera][0]
                cut_end_time = to_cut[subj_ID][camera][1]
                t_map = time.perf_counter()
                start_frame, end_frame, start_rel_time, end_rel_time = self.time_to_frame(subj_ID, camera, cut_start_time, cut_end_time, self.input_gaze_world_dir)
                frame_map_elapsed += time.perf_counter() - t_map
                to_cut_frames[subj_ID][camera] = [start_frame, end_frame, start_rel_time, end_rel_time]

        return cutted_all_soundwaves, to_cut_frames[subj_ID]

    def _run_ffmpeg(self, cmd, context):
        self.logger.debug("Running command (%s): %s", context, " ".join(cmd))
        try:
            return subprocess.run(cmd, check=True, text=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            self.logger.error("FFmpeg failed during %s", context)
            if e.stderr:
                self.logger.error("FFmpeg stderr (%s): %s", context, e.stderr.strip())
            raise

    def sec_to_ts(self, x):
        """Convert seconds (float/np.float64) to ffmpeg timestamp 'HH:MM:SS.mmm'."""
        x = float(x)
        if not math.isfinite(x) or x < 0:
            raise ValueError(f"Bad timestamp: {x}")
        h = int(x // 3600)
        m = int((x % 3600) // 60)
        s = x - (h*3600 + m*60)
        return f"{h:02d}:{m:02d}:{s:06.3f}"  # 3 ms precision is usually enough

    def cut_frames(self, original_file_path, output_file_path, start_frame, end_frame):
        """
        Cut specific frames using FFmpeg by directly selecting frames from the video.
        
        Parameters
        ----------
        original_file_path : str, Path to the original video file for a specific subject and camera type.
        start_frame : int, Starting frame index (inclusive).
        end_frame : int, Ending frame index (inclusive).
        
        Returns
        None (savwes the cut video to output directory)
        """
        # Build FFmpeg command
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite
            "-i", original_file_path,
            "-vf", f"select='between(n,{start_frame},{end_frame})',setpts=PTS-STARTPTS",
            "-an",  # remove audio
            output_file_path,
        ]

        self._run_ffmpeg(cmd, context="cut_frames")
        self.logger.info("Saved cut video: %s", output_file_path)
        return output_file_path

    def cut_audio(self, start_frame_relative_time, end_frame_relative_time, original_file_path, output_file_path):

        """
        Cut audio segement from original file merge with cut video

        parameters:
        start_frame_relative_time : float, start time in seconds (corresponding time to the cut frame from worldstamp.csv)
        end_frame_relative_time : float, end time in seconds (corresponding time to the cut frame from worldstamp.csv)
        subj : int, Subject ID.
        camera : str, ['child', 'parent'].
        original_file_path : str, Path to the original video file for a specific subject and camera
        output_file_path : str, Path to the output audio file

        Returns
        None (saves the cut audio to output directory)
        """
        # Example: ensure these are plain floats (not np.float64) and valid
        start_s = float(start_frame_relative_time)
        end_s   = float(end_frame_relative_time)
        if end_s <= start_s:
            raise ValueError(f"end time ({end_s}) must be > start time ({start_s})")

        # Prefer duration to avoid ambiguity with -to
        dur_s = end_s - start_s


        # Build strings for ffmpeg
        start_ts = self.sec_to_ts(start_s)
        end_ts = self.sec_to_ts(end_s)
        dur_ts = self.sec_to_ts(dur_s)

        self.logger.info(
            "Extracting audio segment start=%s end=%s duration=%s input=%s",
            start_ts, end_ts, dur_ts, original_file_path
        )


        # Make sure output directory exists

        cmd = [
            "ffmpeg",
            "-y",                     # overwrite if exists
            "-ss", start_ts,          # seek to start (input seeking for speed)
            "-i", original_file_path,
            "-t", dur_ts,             # duration (safer than -to here)
            "-vn",
            "-acodec", "pcm_s16le",   # uncompressed 16-bit PCM WAV
            output_file_path
        ]

        self._run_ffmpeg(cmd, context="cut_audio")
        self.logger.info("Saved cut audio: %s", output_file_path)

    def merge_audio_video(self, video_file_path, audio_file_path, merged_file_path):

        cmd = [
            "ffmpeg", "-y",                      # overwrite output if exists
            "-i", video_file_path,   # video with no audio
            "-i", audio_file_path,
            "-map", "0:v:0",       # your extracted video segment
            "-map", "1:a:0",       # your extracted audio segment
            "-c:v", "copy",      # do NOT re-encode video → no quality loss
            "-c:a", "aac",      # do NOT re-encode audio lossyly (AAC is widely supported in MP4 containers)
            "-shortest",         # trim whichever stream is longer
            merged_file_path
        ]

        self._run_ffmpeg(cmd, context="merge_audio_video")
        self.logger.info("Saved merged video: %s", merged_file_path)

    def process_subject(self, subj, camera_list, ref_cam):

        """
        Process a single subject for both camera types.
        
        Parameters
        ----------
        subj : int, Subject ID.
        """

        subject_t0 = time.perf_counter()
        timing = {
            "waveform_load": 0.0,
            "sync": 0.0,
            "frame_mapping": 0.0,
            "ffmpeg_cut_video": 0.0,
            "ffmpeg_cut_audio": 0.0,
            "ffmpeg_merge": 0.0,
        }

        waveforms, sr_ref, timing["waveform_load"] = self._load_subject_waveforms(
            subj, camera_list, audio_sr=48000
        )
        self.logger.info("------------------------------------")
        self.logger.info("Start Spectrogram Synchronization")
        self.logger.info("------------------------------------")
        t_sync = time.perf_counter()
        time_lags = self.spectrogram_synchronizer(
            subj,
            camera_list,
            spectrograph_sr=400,
            audio_sr=48000,
            ref_cam=ref_cam,
            waveforms=waveforms,
            sr_ref=sr_ref,
            max_lag_sec=self.max_lag_sec,
        )
        timing["sync"] = time.perf_counter() - t_sync

        t_map = time.perf_counter()
        _, to_cut_frames = self.time_lag_to_cut_frames(
            subj,
            camera_list,
            time_lags,
            audio_sr=48000,
            ref_cam=ref_cam,
            waveforms=waveforms,
            sr_ref=sr_ref,
        )
        timing["frame_mapping"] = time.perf_counter() - t_map
        self.to_cut_frames_all[int(subj)] = to_cut_frames
        self.logger.info(
            "Collected cut-frame entries subject=%s cameras=%d camera_ids=%s",
            subj,
            len(to_cut_frames),
            sorted(to_cut_frames.keys()),
        )

        self.logger.info("------------------------------------")
        self.logger.info("Start Cutting Videos and Audios with FFmpeg")
        self.logger.info("------------------------------------")
        for camera in camera_list:
                if camera not in to_cut_frames:
                    self.logger.warning(
                        "No cut-frame info for subject=%s camera=%s. Skipping camera.",
                        subj,
                        camera,
                    )
                    continue
                original_file_path = os.path.join(
                    self.input_video_dir,
                    f"{subj}_{camera}.mp4"
                )

                start_frame, end_frame = to_cut_frames[camera][0], to_cut_frames[camera][1]
                start_frame_relative_time, end_frame_relative_time = to_cut_frames[camera][2], to_cut_frames[camera][3]

                # Build output path
                output_video_path = (
                    f"{self.output_dir}/"
                    f"{subj}_{camera}_cut.mp4"
                )
            
                self.logger.info(
                    "Processing subject=%s camera=%s start_frame=%s end_frame=%s",
                    subj, camera, start_frame, end_frame
                )

                t_cut_video = time.perf_counter()
                self.cut_frames(original_file_path, output_video_path, start_frame, end_frame) 
                timing["ffmpeg_cut_video"] += time.perf_counter() - t_cut_video

                
                output_audio_path = (
                    f"{self.output_dir}/"
                    f"{subj}_{camera}_cut.wav"
                )
                t_cut_audio = time.perf_counter()
                self.cut_audio(start_frame_relative_time, end_frame_relative_time, original_file_path, output_audio_path)
                timing["ffmpeg_cut_audio"] += time.perf_counter() - t_cut_audio

                merged_output_path = (
                    f"{self.merged_output_dir}/"
                    f"{subj}_{camera}_cut_merged.mp4"
                )
                t_merge = time.perf_counter()
                self.merge_audio_video(output_video_path, output_audio_path, merged_output_path)
                timing["ffmpeg_merge"] += time.perf_counter() - t_merge
                self.logger.info(
                    "Completed subject=%s camera=%s with merged output=%s",
                    subj, camera, merged_output_path
                )

        total_elapsed = time.perf_counter() - subject_t0
        self.logger.info(
            "subject=%s timing_summary_sec={waveform_load:%.3f, sync:%.3f, frame_mapping:%.3f, cut_video:%.3f, cut_audio:%.3f, merge:%.3f, total:%.3f}",
            subj,
            timing["waveform_load"],
            timing["sync"],
            timing["frame_mapping"],
            timing["ffmpeg_cut_video"],
            timing["ffmpeg_cut_audio"],
            timing["ffmpeg_merge"],
            total_elapsed,
        )

    def save_cut_frames_pickle(self, pickle_path: str | Path) -> Path:
        path = Path(pickle_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.to_cut_frames_all, f)

        total_cameras = sum(len(camera_map) for camera_map in self.to_cut_frames_all.values())
        self.logger.info(
            "Saved cut-frame pickle: %s subjects=%d cameras=%d",
            path,
            len(self.to_cut_frames_all),
            total_cameras,
        )
        return path

def main():

    """Main function to run the video synchronization."""
        
    parser = argparse.ArgumentParser(description='Video Cutter based on frame indices from pickle file')
    parser.add_argument('input_video_dir', help='Root folder containing videos')
    parser.add_argument('input_audio_dir', help='Root folder containing audios')
    parser.add_argument('input_gaze_world_dir', help='Folder contains worldstamps of videos and gaze data')
    parser.add_argument('output_dir', help='Folder to save cut videos and audios')
    parser.add_argument('merged_output_dir_pos', nargs='?', default=None, help='(Optional legacy positional) folder to save merged videos')
    parser.add_argument('--merged-output-dir', default=None, help='Folder to save merged videos')
    parser.add_argument('--ref-cam', default=None, help='reference camera type for synchronization. e.g., child')
    parser.add_argument('--subject-id', help='Process only specific subject ID. e.g., 1,2,3')
    parser.add_argument('--camera-id', help='Process only specific camera type. e.g., child, parent, room, side')
    parser.add_argument('--cut-frames-pkl', default=None, help='Path to save cut-frame mapping pickle (default: output_dir/to_cut_frames.pkl)')
    parser.add_argument('--log-path', default=None, help='Path to log file (default: output_dir/video_aligner.log)')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto', help='Device for conv1d sync compute')
    parser.add_argument('--max-lag-sec', type=float, default=600.0, help='Maximum lag (seconds) searched by cross-correlation')

    args = parser.parse_args()
        
        
    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, 'video_aligner.log')
    setup_logging(Path(args.log_path))
    logger = logging.getLogger(__name__)
    logger.info("Logging to: %s", Path(args.log_path).resolve())
    resolved_device, requested_device = _resolve_torch_device(args.device, logger)
    logger.info("Requested device=%s resolved device=%s", requested_device, resolved_device)
    logger.info("Configured max_lag_sec=%.3f", args.max_lag_sec)

    # Check if ffmpeg is available
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FFmpeg not found. Please install FFmpeg to use this script.")
        return
    
    if args.subject_id:
        subj_ids = [int(id.strip()) for id in args.subject_id.split(',')]
    else:
        video_dir = Path(args.input_video_dir)
        subj_ids = np.unique([
            int(p.stem.split('_')[0])
            for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ])

    if args.camera_id:
        camera_list = [i.strip() for i in args.camera_id.split(',')]
    else:
        video_dir = Path(args.input_video_dir)
        camera_list = np.unique([
            p.stem.split('_')[1] for p in video_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".webm"}
        ])
    
    # Backward compatible merged-output-dir resolution:
    # priority: explicit flag > legacy positional > output_dir
    if args.merged_output_dir is not None:
        merged_output_dir = args.merged_output_dir
    elif args.merged_output_dir_pos is not None:
        merged_output_dir = args.merged_output_dir_pos
    else:
        merged_output_dir = args.output_dir

    cut_frames_pkl_path = Path(args.cut_frames_pkl) if args.cut_frames_pkl else Path(args.output_dir) / "to_cut_frames.pkl"

    # Initialize synchronizer
    logger.info("Subjects=%s Cameras=%s", list(subj_ids), list(camera_list))
    logger.info("Input video dir: %s", args.input_video_dir)
    logger.info("Input audio dir: %s", args.input_audio_dir)
    logger.info("Input gaze/world dir: %s", args.input_gaze_world_dir)
    logger.info("Output dir: %s", args.output_dir)
    logger.info("Merged output dir: %s", merged_output_dir)
    logger.info("Cut-frame pickle path: %s", cut_frames_pkl_path)

    synchronizer = FrameAligner(
        args.input_video_dir,
        args.input_audio_dir,
        args.input_gaze_world_dir,
        args.output_dir,
        merged_output_dir,
        logger=logger,
        device=resolved_device,
        requested_device=requested_device,
        max_lag_sec=args.max_lag_sec,
    )
    for subj in subj_ids:
        logger.info("Starting subject: %s", subj)
        synchronizer.process_subject(subj, camera_list, args.ref_cam)
        logger.info("Completed subject: %s", subj)
    saved_pickle_path = synchronizer.save_cut_frames_pickle(cut_frames_pkl_path)
    logger.info("Cut-frame pickle ready: %s", saved_pickle_path)
    logger.info("Video synchronization complete.")

if __name__ == "__main__":
    main()
