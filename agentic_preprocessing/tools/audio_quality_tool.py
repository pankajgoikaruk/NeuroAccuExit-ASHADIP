# agentic_preprocessing\tools\audio_quality_tool.py



"""
Audio quality utilities for Agentic AI preprocessing.

Version 0.3:
- Non-destructive
- WAV-focused
- Separates preprocessing reasons from quality reasons
- Adds borderline warning codes
- Uses safe_after_preprocessing instead of safe_to_train
- Does not delete, move, or overwrite raw files
"""

from __future__ import annotations

import hashlib
import math
import struct
import wave
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


AGENT_VERSION = "v0.3_non_destructive_dataset_auditor"


DEFAULT_POLICY: Dict[str, Any] = {
    "expected_sample_rate": 16000,
    "expected_channels": 1,
    "expected_duration_sec": 5.0,
    "duration_tolerance_sec": 1.0,
    "min_duration_sec": 1.0,

    # Hard quality thresholds
    "high_silence_ratio": 0.70,
    "very_low_rms_db": -45.0,
    "clipping_ratio_threshold": 0.01,
    "min_speech_activity_ratio": 0.25,

    # Borderline warning thresholds
    "borderline_silence_ratio": 0.60,
    "borderline_low_rms_db": -35.0,
    "borderline_clipping_ratio": 0.005,
    "borderline_speech_activity_ratio": 0.35,
}


PREPROCESSING_REASON_CODES = {
    "sample_rate_mismatch",
    "channel_mismatch",
    "duration_mismatch",
}


QUALITY_REASON_CODES = {
    "too_short",
    "high_silence_ratio",
    "very_low_signal",
    "possible_clipping_distortion",
    "low_speech_activity",
    "exact_duplicate_candidate",
    "empty_audio",
}


BLOCKED_REASONS = {
    "class_folder_missing",
    "no_audio_files_found",
    "empty_audio",
}


BLOCKED_PREFIXES = (
    "unsupported_sample_width",
    "file_read_error",
    "wave_error",
)


REJECTED_REASONS = {
    "too_short",
    "high_silence_ratio",
    "very_low_signal",
    "low_speech_activity",
}


NEEDS_REVIEW_REASONS = {
    "possible_clipping_distortion",
    "exact_duplicate_candidate",
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
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)

    return h.hexdigest()


def _decode_pcm_samples(raw: bytes, sample_width: int) -> Tuple[List[int], float]:
    if sample_width == 1:
        samples = [b - 128 for b in raw]
        return samples, 128.0

    if sample_width == 2:
        n = len(raw) // 2
        samples = list(struct.unpack("<" + "h" * n, raw[: n * 2]))
        return samples, 32768.0

    if sample_width == 3:
        samples: List[int] = []
        for i in range(0, len(raw) - 2, 3):
            value = int.from_bytes(raw[i : i + 3], byteorder="little", signed=False)
            if value & 0x800000:
                value -= 0x1000000
            samples.append(value)
        return samples, 8388608.0

    if sample_width == 4:
        n = len(raw) // 4
        samples = list(struct.unpack("<" + "i" * n, raw[: n * 4]))
        return samples, 2147483648.0

    raise ValueError(f"unsupported_sample_width_{sample_width}")


def read_wav_basic_stats(path: Path) -> Dict[str, Any]:
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

    silence_threshold = max_abs_possible * (10 ** (-45 / 20))
    silence_ratio = sum(1 for x in abs_samples if x < silence_threshold) / len(abs_samples)

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
    p = dict(DEFAULT_POLICY)
    if policy:
        p.update(policy)

    if not stats.get("readable", False):
        error = stats.get("error", "unreadable_or_unsupported_audio")
        return [str(error)]

    reasons: List[str] = []

    duration = safe_float(stats.get("duration_sec"))
    sample_rate = int(stats.get("sample_rate", 0) or 0)
    channels = int(stats.get("channels", 0) or 0)
    rms_db = safe_float(stats.get("rms_db"), -120.0)
    silence_ratio = safe_float(stats.get("silence_ratio"))
    clipping_ratio = safe_float(stats.get("clipping_ratio"))
    speech_activity_ratio = safe_float(stats.get("speech_activity_ratio"))

    expected_sample_rate = int(p.get("expected_sample_rate", 16000))
    expected_channels = int(p.get("expected_channels", 1))
    expected_duration_sec = safe_float(p.get("expected_duration_sec"), 5.0)
    duration_tolerance_sec = safe_float(p.get("duration_tolerance_sec"), 1.0)
    min_duration_sec = safe_float(p.get("min_duration_sec"), 1.0)
    high_silence_ratio = safe_float(p.get("high_silence_ratio"), 0.70)
    very_low_rms_db = safe_float(p.get("very_low_rms_db"), -45.0)
    clipping_ratio_threshold = safe_float(p.get("clipping_ratio_threshold"), 0.01)
    min_speech_activity_ratio = safe_float(p.get("min_speech_activity_ratio"), 0.25)

    if sample_rate != expected_sample_rate:
        reasons.append("sample_rate_mismatch")

    if channels != expected_channels:
        reasons.append("channel_mismatch")

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


def split_reason_codes(reason_codes: List[str]) -> Tuple[List[str], List[str], List[str]]:
    preprocessing_reasons: List[str] = []
    quality_reasons: List[str] = []
    other_reasons: List[str] = []

    for reason in reason_codes:
        if reason in PREPROCESSING_REASON_CODES:
            preprocessing_reasons.append(reason)
        elif reason in QUALITY_REASON_CODES:
            quality_reasons.append(reason)
        elif reason != "no_major_issue_detected":
            other_reasons.append(reason)

    return preprocessing_reasons, quality_reasons, other_reasons


def build_warning_codes(
    stats: Dict[str, Any],
    reason_codes: List[str],
    policy: Optional[Dict[str, Any]] = None,
) -> List[str]:
    p = dict(DEFAULT_POLICY)
    if policy:
        p.update(policy)

    if not stats.get("readable", False):
        return []

    warnings: List[str] = []

    rms_db = safe_float(stats.get("rms_db"), -120.0)
    silence_ratio = safe_float(stats.get("silence_ratio"))
    clipping_ratio = safe_float(stats.get("clipping_ratio"))
    speech_activity_ratio = safe_float(stats.get("speech_activity_ratio"))

    borderline_silence_ratio = safe_float(p.get("borderline_silence_ratio"), 0.60)
    high_silence_ratio = safe_float(p.get("high_silence_ratio"), 0.70)

    borderline_low_rms_db = safe_float(p.get("borderline_low_rms_db"), -35.0)
    very_low_rms_db = safe_float(p.get("very_low_rms_db"), -45.0)

    borderline_clipping_ratio = safe_float(p.get("borderline_clipping_ratio"), 0.005)
    clipping_ratio_threshold = safe_float(p.get("clipping_ratio_threshold"), 0.01)

    borderline_speech_activity_ratio = safe_float(p.get("borderline_speech_activity_ratio"), 0.35)
    min_speech_activity_ratio = safe_float(p.get("min_speech_activity_ratio"), 0.25)

    if (
        "high_silence_ratio" not in reason_codes
        and borderline_silence_ratio <= silence_ratio <= high_silence_ratio
    ):
        warnings.append("borderline_high_silence")

    if (
        "very_low_signal" not in reason_codes
        and very_low_rms_db <= rms_db <= borderline_low_rms_db
    ):
        warnings.append("borderline_low_signal")

    if (
        "possible_clipping_distortion" not in reason_codes
        and borderline_clipping_ratio <= clipping_ratio <= clipping_ratio_threshold
    ):
        warnings.append("borderline_clipping_risk")

    if (
        "low_speech_activity" not in reason_codes
        and min_speech_activity_ratio <= speech_activity_ratio <= borderline_speech_activity_ratio
    ):
        warnings.append("borderline_low_speech_activity")

    return warnings


def build_preprocessing_actions(preprocessing_reason_codes: List[str]) -> List[str]:
    actions: List[str] = []

    if "sample_rate_mismatch" in preprocessing_reason_codes:
        actions.append("resample_to_expected_sample_rate")

    if "channel_mismatch" in preprocessing_reason_codes:
        actions.append("downmix_to_mono")

    if "duration_mismatch" in preprocessing_reason_codes:
        actions.append("verify_or_normalize_duration")

    return actions


def decide_from_reason_codes(
    raw_reason_codes: List[str],
    preprocessing_reason_codes: List[str],
    quality_reason_codes: List[str],
) -> Tuple[str, bool, bool, List[str], List[str]]:
    """
    Returns:
    - decision
    - safe_after_preprocessing
    - requires_preprocessing
    - preprocessing_actions
    - decision_reason_codes
    """
    preprocessing_actions = build_preprocessing_actions(preprocessing_reason_codes)
    requires_preprocessing = len(preprocessing_actions) > 0

    decision_reason_codes: List[str] = []

    for reason in raw_reason_codes:
        if reason in BLOCKED_REASONS or any(reason.startswith(prefix) for prefix in BLOCKED_PREFIXES):
            decision_reason_codes.append(reason)
            return "blocked", False, requires_preprocessing, preprocessing_actions, decision_reason_codes

    rejected_hits = [r for r in quality_reason_codes if r in REJECTED_REASONS]
    if rejected_hits:
        decision_reason_codes.extend(rejected_hits)
        return "rejected", False, requires_preprocessing, preprocessing_actions, decision_reason_codes

    needs_review_hits = [r for r in quality_reason_codes if r in NEEDS_REVIEW_REASONS]
    if needs_review_hits:
        decision_reason_codes.extend(needs_review_hits)
        return "needs_review", False, requires_preprocessing, preprocessing_actions, decision_reason_codes

    if raw_reason_codes == ["no_major_issue_detected"]:
        decision_reason_codes.append("no_major_issue_detected")
    elif preprocessing_reason_codes and not quality_reason_codes:
        decision_reason_codes.append("preprocessing_required_only")
    else:
        decision_reason_codes.append("accepted_no_blocking_quality_issue")

    return "accepted", True, requires_preprocessing, preprocessing_actions, decision_reason_codes


def analyse_audio_file(
    path: Path,
    policy: Optional[Dict[str, Any]] = None,
    extra_reasons: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
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

    raw_reason_codes = build_reason_codes(stats, policy=policy, extra_reasons=extra_reasons)
    preprocessing_reason_codes, quality_reason_codes, other_reason_codes = split_reason_codes(raw_reason_codes)
    warning_codes = build_warning_codes(stats, raw_reason_codes, policy=policy)

    decision, safe_after_preprocessing, requires_preprocessing, preprocessing_actions, decision_reason_codes = (
        decide_from_reason_codes(
            raw_reason_codes=raw_reason_codes,
            preprocessing_reason_codes=preprocessing_reason_codes,
            quality_reason_codes=quality_reason_codes,
        )
    )

    stats["decision"] = decision
    stats["safe_after_preprocessing"] = safe_after_preprocessing
    stats["requires_preprocessing"] = requires_preprocessing
    stats["preprocessing_actions"] = preprocessing_actions
    stats["raw_reason_codes"] = raw_reason_codes
    stats["preprocessing_reason_codes"] = preprocessing_reason_codes
    stats["quality_reason_codes"] = quality_reason_codes
    stats["warning_codes"] = warning_codes
    stats["decision_reason_codes"] = decision_reason_codes
    stats["other_reason_codes"] = other_reason_codes
    stats["agent_version"] = AGENT_VERSION

    return stats