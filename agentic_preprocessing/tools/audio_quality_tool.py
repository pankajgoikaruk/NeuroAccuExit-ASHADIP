# agentic_preprocessing\tools\audio_quality_tool.py

"""
Audio quality utilities for Agentic AI preprocessing.

Version 0.1:
- Non-destructive
- WAV-focused
- No file deletion
- No file movement
- Produces interpretable quality indicators and decision reasons
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


AGENT_VERSION = "v0.1_non_destructive_dataset_auditor"


DEFAULT_POLICY: Dict[str, Any] = {
    "expected_sample_rate": 16000,
    "expected_duration_sec": 5.0,
    "duration_tolerance_sec": 1.0,
    "min_duration_sec": 1.0,
    "high_silence_ratio": 0.70,
    "very_low_rms_db": -45.0,
    "clipping_ratio_threshold": 0.01,
    "min_speech_activity_ratio": 0.25,
}


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except Exception:
        return default


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """
    Exact duplicate detector helper.

    This does not detect near-duplicates yet.
    It is intentionally safe and deterministic for V1.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _decode_pcm_samples(raw: bytes, sample_width: int) -> Tuple[List[int], float]:
    """
    Decode integer PCM samples from WAV bytes.

    Supports:
    - 8-bit unsigned PCM
    - 16-bit signed PCM
    - 24-bit signed PCM
    - 32-bit signed PCM

    Note:
    Some 32-bit WAV files may store float PCM. This V1 tool treats 32-bit as signed int.
    Those edge cases can be improved later using soundfile/librosa if needed.
    """
    if sample_width == 1:
        samples = [b - 128 for b in raw]
        max_abs_possible = 128.0
        return samples, max_abs_possible

    if sample_width == 2:
        n = len(raw) // 2
        samples = list(struct.unpack("<" + "h" * n, raw[: n * 2]))
        max_abs_possible = 32768.0
        return samples, max_abs_possible

    if sample_width == 3:
        samples: List[int] = []
        for i in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[i : i + 3], byteorder="little", signed=False)
            if value & 0x800000:
                value -= 0x1000000
            samples.append(value)
        max_abs_possible = 8388608.0
        return samples, max_abs_possible

    if sample_width == 4:
        n = len(raw) // 4
        samples = list(struct.unpack("<" + "i" * n, raw[: n * 4]))
        max_abs_possible = 2147483648.0
        return samples, max_abs_possible

    raise ValueError(f"unsupported_sample_width_{sample_width}")


def read_wav_basic_stats(path: Path) -> Dict[str, Any]:
    """
    Read basic WAV quality statistics.

    Non-destructive:
    - does not modify audio
    - does not move files
    - does not delete files
    """
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        duration_sec = n_frames / float(sample_rate) if sample_rate else 0.0
        raw = wf.readframes(n_frames)

    samples, max_abs_possible = _decode_pcm_samples(raw, sample_width)

    if not samples:
        return {
            "readable": False,
            "error": "empty_audio",
            "source_file": str(path),
        }

    abs_samples = [abs(x) for x in samples]
    peak = max(abs_samples)
    rms = math.sqrt(sum(float(x) * float(x) for x in samples) / len(samples))

    peak_db = 20 * math.log10(max(peak / max_abs_possible, 1e-12))
    rms_db = 20 * math.log10(max(rms / max_abs_possible, 1e-12))

    # Simple near-silence estimate.
    # Samples below -45 dBFS are counted as near-silent.
    silence_threshold = max_abs_possible * (10 ** (-45 / 20))
    silence_ratio = sum(1 for x in abs_samples if x < silence_threshold) / len(abs_samples)

    # Simple clipping estimate.
    clipping_threshold = max_abs_possible * 0.98
    clipping_ratio = sum(1 for x in abs_samples if x >= clipping_threshold) / len(abs_samples)

    speech_activity_ratio = max(0.0, min(1.0, 1.0 - silence_ratio))

    return {
        "readable": True,
        "error": "",
        "source_file": str(path),
        "sample_rate": sample_rate,
        "channels": channels,
        "sample_width": sample_width,
        "n_frames": n_frames,
        "duration_sec": duration_sec,
        "rms_db": rms_db,
        "peak_db": peak_db,
        "silence_ratio": silence_ratio,
        "clipping_ratio": clipping_ratio,
        "speech_activity_ratio": speech_activity_ratio,
    }


def build_reason_codes(
    stats: Dict[str, Any],
    policy: Optional[Dict[str, Any]] = None,
    extra_reasons: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Build interpretable reason codes for agentic routing.
    """
    p = dict(DEFAULT_POLICY)
    if policy:
        p.update(policy)

    if not stats.get("readable", False):
        error = stats.get("error", "unreadable_or_unsupported_audio")
        return [str(error)]

    reasons: List[str] = []

    duration = safe_float(stats.get("duration_sec"))
    sample_rate = int(stats.get("sample_rate", 0) or 0)
    rms_db = safe_float(stats.get("rms_db"), -120.0)
    silence_ratio = safe_float(stats.get("silence_ratio"))
    clipping_ratio = safe_float(stats.get("clipping_ratio"))
    speech_activity_ratio = safe_float(stats.get("speech_activity_ratio"))

    expected_sample_rate = int(p.get("expected_sample_rate", 16000))
    expected_duration_sec = safe_float(p.get("expected_duration_sec"), 5.0)
    duration_tolerance_sec = safe_float(p.get("duration_tolerance_sec"), 1.0)
    min_duration_sec = safe_float(p.get("min_duration_sec"), 1.0)
    high_silence_ratio = safe_float(p.get("high_silence_ratio"), 0.70)
    very_low_rms_db = safe_float(p.get("very_low_rms_db"), -45.0)
    clipping_ratio_threshold = safe_float(p.get("clipping_ratio_threshold"), 0.01)
    min_speech_activity_ratio = safe_float(p.get("min_speech_activity_ratio"), 0.25)

    if sample_rate != expected_sample_rate:
        reasons.append("sample_rate_mismatch")

    if duration < min_duration_sec:
        reasons.append("too_short")

    if abs(duration - expected_duration_sec) > duration_tolerance_sec:
        reasons.append("duration_mismatch")

    if silence_ratio > high_silence_ratio:
        reasons.append("high_silence_ratio")

    if rms_db < very_low_rms_db:
        reasons.append("very_low_signal")

    if clipping_ratio > clipping_ratio_threshold:
        reasons.append("possible_clipping_distortion")

    if speech_activity_ratio < min_speech_activity_ratio:
        reasons.append("low_speech_activity")

    if extra_reasons:
        for reason in extra_reasons:
            if reason and reason not in reasons:
                reasons.append(str(reason))

    if not reasons:
        reasons.append("no_major_issue_detected")

    return reasons


def decide_from_reasons(reasons: List[str]) -> Tuple[str, bool]:
    """
    Convert reason codes into a conservative agentic decision.

    Decision meanings:
    - accepted: safe enough for training
    - needs_review: suspicious, human review recommended
    - rejected: clearly low-quality, but still preserved
    - blocked: unreadable/corrupted/unsupported
    """
    blocked_reasons = {
        "class_folder_missing",
        "no_audio_files_found",
        "empty_audio",
    }

    blocked_prefixes = (
        "unsupported_sample_width",
        "file_read_error",
        "wave_error",
    )

    for reason in reasons:
        if reason in blocked_reasons:
            return "blocked", False
        if any(reason.startswith(prefix) for prefix in blocked_prefixes):
            return "blocked", False

    rejected_reasons = {
        "too_short",
        "high_silence_ratio",
        "very_low_signal",
        "low_speech_activity",
    }

    if any(reason in rejected_reasons for reason in reasons):
        return "rejected", False

    review_reasons = {
        "sample_rate_mismatch",
        "duration_mismatch",
        "possible_clipping_distortion",
        "exact_duplicate_candidate",
    }

    if any(reason in review_reasons for reason in reasons):
        return "needs_review", False

    return "accepted", True


def analyse_audio_file(
    path: Path,
    policy: Optional[Dict[str, Any]] = None,
    extra_reasons: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Full single-file audio analysis.
    """
    try:
        stats = read_wav_basic_stats(path)
    except wave.Error as exc:
        stats = {
            "readable": False,
            "error": f"wave_error_{str(exc)}",
            "source_file": str(path),
        }
    except Exception as exc:
        stats = {
            "readable": False,
            "error": f"file_read_error_{str(exc)}",
            "source_file": str(path),
        }

    reasons = build_reason_codes(stats, policy=policy, extra_reasons=extra_reasons)
    decision, safe_to_train = decide_from_reasons(reasons)

    stats["decision"] = decision
    stats["safe_to_train"] = safe_to_train
    stats["reason_codes"] = reasons
    stats["agent_version"] = AGENT_VERSION

    return stats