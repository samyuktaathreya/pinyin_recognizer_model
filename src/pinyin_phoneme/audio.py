"""Loading and cleaning audio. Used by data preparation only."""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def load_audio(path: Path, sample_rate: int) -> np.ndarray:
    """Read any supported file as mono float32 at the given sample rate."""
    wave, source_rate = sf.read(str(path), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    wave = wave - wave.mean()      # remove any constant offset from the recording
    if source_rate != sample_rate:
        divisor = gcd(source_rate, sample_rate)
        wave = resample_poly(wave, sample_rate // divisor, source_rate // divisor)
    return wave.astype(np.float32)


def trim_silence(
    wave: np.ndarray, sample_rate: int, threshold_db: float = 30.0, margin_s: float = 0.05
) -> np.ndarray:
    """Cut leading and trailing audio that is much quieter than the loudest part.

    Loudness is measured in 20 ms frames. Everything from the first to the last
    frame within `threshold_db` of the loudest frame is kept, plus a small margin.
    """
    frame = int(0.02 * sample_rate)
    n_frames = len(wave) // frame
    if n_frames < 2:
        return wave
    rms = np.sqrt((wave[: n_frames * frame].reshape(n_frames, frame) ** 2).mean(axis=1))
    if rms.max() <= 0:
        return wave
    loud = np.nonzero(rms >= rms.max() * 10 ** (-threshold_db / 20))[0]
    margin = int(margin_s * sample_rate)
    start = max(0, loud[0] * frame - margin)
    end = min(len(wave), (loud[-1] + 1) * frame + margin)
    return wave[start:end]
