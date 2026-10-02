"""Reads a split csv and hands out (waveform, initial id, final id)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .labels import FINAL_TO_ID, INITIAL_TO_ID


def fit_to_length(wave: np.ndarray, num_samples: int, offset: float = 0.5) -> np.ndarray:
    """Pad with silence or crop so the clip is exactly num_samples long.

    offset says where the spare room goes: 0 = clip at the start, 0.5 = centred,
    1 = clip at the end.
    """
    if len(wave) >= num_samples:
        start = int((len(wave) - num_samples) * offset)
        return wave[start:start + num_samples]
    out = np.zeros(num_samples, dtype=np.float32)
    start = int((num_samples - len(wave)) * offset)
    out[start:start + len(wave)] = wave
    return out


class ClipDataset(Dataset):
    def __init__(self, csv_path: Path, processed_dir: Path, num_samples: int,
                 augment: dict | None = None):
        with open(csv_path, newline="") as f:
            self.rows = list(csv.DictReader(f))
        self.processed_dir = Path(processed_dir)
        self.num_samples = num_samples
        self.augment = augment  # None for validation and test

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        wave, _ = sf.read(str(self.processed_dir / row["path"]), dtype="float32")

        if self.augment:
            wave = self._augment(wave)
        else:
            wave = fit_to_length(wave, self.num_samples)

        return (
            torch.from_numpy(wave),
            INITIAL_TO_ID[row["initial"]],
            FINAL_TO_ID[row["final"]],
        )

    def _augment(self, wave: np.ndarray) -> np.ndarray:
        """Random variations so the model sees a slightly different clip every epoch."""
        cfg = self.augment
        rng = np.random.default_rng()

        # Speed: play the clip slightly faster or slower (also moves the pitch).
        speed = rng.uniform(*cfg["speed"])
        positions = np.arange(0, len(wave) - 1, speed)
        wave = np.interp(positions, np.arange(len(wave)), wave).astype(np.float32)

        # Position: put the syllable anywhere in the window.
        wave = fit_to_length(wave, self.num_samples, offset=rng.uniform())

        # Loudness.
        wave = wave * 10 ** (rng.uniform(*cfg["gain_db"]) / 20)

        # Background noise at a random signal-to-noise ratio.
        power = float((wave ** 2).mean())
        if power > 0:
            noise_power = power / 10 ** (rng.uniform(*cfg["snr_db"]) / 10)
            wave = wave + rng.normal(0, np.sqrt(noise_power), len(wave))

        return np.clip(wave, -1, 1).astype(np.float32)
