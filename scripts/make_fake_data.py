"""Generate synthetic "speech" so the pipeline can be tested without real recordings.

Each fake syllable is a noise burst (whose frequency band depends on the initial)
followed by a buzz (whose resonances depend on the final). It sounds nothing like
Mandarin, but it gives the model something learnable, which is enough to prove
that prepare -> train -> evaluate -> export all work together.

    python scripts/make_fake_data.py --out /tmp/fake/standard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import butter, lfilter

from pinyin_phoneme.labels import FINALS, INITIALS

SYLLABLES = ["ba", "pa", "ma", "bi", "pi", "mi", "bu", "pu", "mu",
             "zhang", "zhe", "zhu", "shang", "she", "shu", "a", "e", "ang"]


def band(noise: np.ndarray, centre: float, rate: int) -> np.ndarray:
    low, high = centre * 0.8 / (rate / 2), min(centre * 1.25 / (rate / 2), 0.99)
    b, a = butter(2, [low, high], btype="band")
    return lfilter(b, a, noise)


def synth(syllable: str, speaker_pitch: float, rng: np.random.Generator, rate: int) -> np.ndarray:
    from pinyin_phoneme.labels import split_syllable
    initial, final = split_syllable(syllable)
    parts = [np.zeros(int(rng.uniform(0.05, 0.3) * rate))]

    if initial != "none":
        centre = 600 + 280 * INITIALS.index(initial)
        burst = band(rng.normal(size=int(0.09 * rate)), centre, rate)
        parts.append(burst / np.abs(burst).max() * 0.4)

    n = int(rng.uniform(0.25, 0.4) * rate)
    t = np.arange(n) / rate
    pitch = speaker_pitch * rng.uniform(0.95, 1.05)
    buzz = sum(np.sin(2 * np.pi * pitch * h * t) / h for h in range(1, 30))
    index = FINALS.index(final)
    vowel = band(buzz, 350 + 90 * index, rate) + band(buzz, 1200 + 110 * ((index * 7) % 20), rate)
    parts.append(vowel / np.abs(vowel).max() * 0.5 * np.hanning(n) ** 0.3)

    parts.append(np.zeros(int(rng.uniform(0.05, 0.3) * rate)))
    wave = np.concatenate(parts)
    return (wave + rng.normal(0, rng.uniform(0.002, 0.02), len(wave))).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--speakers", type=int, default=6)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    rate = 22050   # deliberately not 16 kHz, to exercise resampling
    for s in range(args.speakers):
        pitch = rng.uniform(100, 240)
        folder = Path(args.out) / f"speaker{s + 1}"
        folder.mkdir(parents=True, exist_ok=True)
        for syllable in SYLLABLES:
            for tone in (1, 2, 3, 4):
                sf.write(str(folder / f"{syllable}{tone}.wav"), synth(syllable, pitch, rng, rate), rate)
        (folder / "notes.wav").write_bytes(b"not audio")   # a file that should be skipped
    print(f"Wrote {args.speakers * len(SYLLABLES) * 4} clips to {args.out}")


if __name__ == "__main__":
    main()
