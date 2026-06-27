# agentic_preprocessing/tools/audio_transform_tool.py


"""
Audio transformation utilities for cleaned dataset building.

Purpose:
- Read accepted raw audio.
- Convert to 16 kHz mono WAV.
- Preserve raw files unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), always_2d=True)

    # shape: [samples, channels]
    audio = audio.astype(np.float32)

    return audio, int(sr)


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.reshape(-1, 1)

    if audio.shape[1] == 1:
        return audio

    mono = np.mean(audio, axis=1, keepdims=True)
    return mono.astype(np.float32)


def resample_audio(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr:
        return audio.astype(np.float32)

    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(source_sr, target_sr)
        up = target_sr // gcd
        down = source_sr // gcd
        resampled = resample_poly(audio, up, down, axis=0)
        return resampled.astype(np.float32)

    except Exception:
        # Fallback interpolation if scipy is unavailable.
        old_len = audio.shape[0]
        new_len = int(round(old_len * target_sr / source_sr))

        old_x = np.linspace(0.0, 1.0, old_len, endpoint=False)
        new_x = np.linspace(0.0, 1.0, new_len, endpoint=False)

        channels = []
        for ch in range(audio.shape[1]):
            channels.append(np.interp(new_x, old_x, audio[:, ch]))

        return np.stack(channels, axis=1).astype(np.float32)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    audio = np.clip(audio, -1.0, 1.0)

    sf.write(
        str(path),
        audio,
        samplerate=sample_rate,
        subtype="PCM_16",
        format="WAV",
    )


def transform_to_clean_wav(
    source_path: Path,
    target_path: Path,
    target_sample_rate: int = 16000,
    target_channels: int = 1,
) -> dict:
    audio, source_sr = read_audio(source_path)

    source_channels = audio.shape[1] if audio.ndim == 2 else 1

    if target_channels == 1:
        audio = downmix_to_mono(audio)

    audio = resample_audio(audio, source_sr=source_sr, target_sr=target_sample_rate)

    write_wav(target_path, audio, sample_rate=target_sample_rate)

    return {
        "source_sample_rate": source_sr,
        "source_channels": source_channels,
        "target_sample_rate": target_sample_rate,
        "target_channels": target_channels,
        "target_num_samples": int(audio.shape[0]),
        "target_duration_sec": float(audio.shape[0] / target_sample_rate),
    }