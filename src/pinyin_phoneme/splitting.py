"""Cut a recording that contains all four tones into one clip per tone.

How it works: measure loudness every 10 ms, mark everything clearly above the
background as sound, and join marks that are less than max_gap apart (so a
third tone that dips to near-silence in the middle stays in one piece). What is
left should be exactly four stretches of sound, taken as tones 1 to 4.

Some speakers first spell the syllable out (initial, then final) before the
four tones. That intro shows up as a fifth stretch at the start and is dropped.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter1d

FRAME_S = 0.01


def find_sounds(wave: np.ndarray, rate: int, max_gap_s: float) -> list[tuple[float, float]]:
    """Return (start, end) in seconds for each stretch of sound in the recording."""
    frame = int(FRAME_S * rate)
    n = len(wave) // frame
    rms = np.sqrt((wave[: n * frame].reshape(n, frame) ** 2).mean(axis=1))
    rms = uniform_filter1d(rms, 5)                      # smooth over 50 ms
    peak = rms.max()
    if peak <= 0:
        return []
    noise = np.percentile(rms, 20)
    # Sound = 10 dB above the background noise, and never quieter than 35 dB below the peak.
    threshold = max(noise * 10 ** (10 / 20), peak * 10 ** (-35 / 20))
    loud = np.append(rms > threshold, False)

    runs, start = [], None
    for i, is_loud in enumerate(loud):
        if is_loud and start is None:
            start = i
        elif not is_loud and start is not None:
            runs.append([start, i])
            start = None

    merged: list[list[int]] = []
    max_gap = int(max_gap_s / FRAME_S)
    for run in runs:
        if merged and run[0] - merged[-1][1] < max_gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)

    # Drop clicks and blips: too short, or far quieter than the loudest part.
    sounds = [
        (s, e) for s, e in merged
        if (e - s) * FRAME_S >= 0.08 and rms[s:e].max() >= peak * 10 ** (-20 / 20)
    ]
    return [(s * FRAME_S, e * FRAME_S) for s, e in sounds]


def cut_points(sounds: list[tuple[float, float]], total_s: float, margin_s: float):
    """Widen each stretch by a margin, without reaching into its neighbours."""
    cuts = []
    for i, (start, end) in enumerate(sounds):
        earliest = 0.0 if i == 0 else (sounds[i - 1][1] + start) / 2
        latest = total_s if i == len(sounds) - 1 else (end + sounds[i + 1][0]) / 2
        cuts.append((max(earliest, start - margin_s), min(latest, end + margin_s)))
    return cuts


def split_four_tones(wave: np.ndarray, rate: int, allowed_counts=(4,),
                     max_gap_s: float = 0.35, margin_s: float = 0.1):
    """Return (clips, sounds). clips is a list of four waveforms, tones 1 to 4,
    or None if the recording did not contain an allowed number of sounds.

    allowed_counts: 4 means just the four tones; 5 means a spelled-out intro
    followed by the four tones. The last four sounds are always the tones.
    """
    wave = wave - wave.mean()      # some sources sit off-centre, which looks like noise
    sounds = find_sounds(wave, rate, max_gap_s)
    if len(sounds) not in allowed_counts:
        return None, sounds
    cuts = cut_points(sounds, len(wave) / rate, margin_s)[-4:]
    return [wave[int(s * rate): int(e * rate)] for s, e in cuts], sounds