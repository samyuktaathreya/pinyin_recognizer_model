"""The network: raw audio in, initial and final scores out.

The spectrogram step lives inside the model (LogMel), built from plain
convolution and matrix operations. That means the exported ONNX file takes raw
audio, and the app cannot compute the features differently from training.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def mel_filterbank(sample_rate: int, n_fft: int, n_mels: int,
                   f_min: float = 50.0, f_max: float | None = None) -> np.ndarray:
    """Triangular filters that pool FFT bins into mel bands. Shape [n_mels, n_fft//2 + 1]."""
    f_max = f_max or sample_rate / 2 - 400

    def to_mel(hz):
        return 2595 * np.log10(1 + np.asarray(hz) / 700)

    def to_hz(mel):
        return 700 * (10 ** (np.asarray(mel) / 2595) - 1)

    edges = to_hz(np.linspace(to_mel(f_min), to_mel(f_max), n_mels + 2))
    bins = np.linspace(0, sample_rate / 2, n_fft // 2 + 1)
    bank = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for m in range(n_mels):
        low, centre, high = edges[m], edges[m + 1], edges[m + 2]
        rising = (bins - low) / (centre - low)
        falling = (high - bins) / (high - centre)
        bank[m] = np.clip(np.minimum(rising, falling), 0, None)
    return bank


class LogMel(nn.Module):
    """Waveform [batch, samples] -> log-mel spectrogram [batch, 1, n_mels, frames].

    Fixed maths, nothing here is trained. The short-time Fourier transform is
    done as a convolution whose kernels are windowed sines and cosines.
    """

    def __init__(self, sample_rate: int, n_fft: int, hop: int, n_mels: int):
        super().__init__()
        self.hop = hop
        self.n_fft = n_fft
        n = np.arange(n_fft)
        k = np.arange(n_fft // 2 + 1)[:, None]
        window = np.hanning(n_fft + 1)[:-1]
        angle = 2 * np.pi * k * n / n_fft
        kernels = np.concatenate([np.cos(angle) * window, -np.sin(angle) * window])
        self.register_buffer("kernels", torch.tensor(kernels, dtype=torch.float32).unsqueeze(1))
        self.register_buffer("mel", torch.tensor(mel_filterbank(sample_rate, n_fft, n_mels)))

    def forward(self, wave: torch.Tensor) -> torch.Tensor:
        spec = nn.functional.conv1d(wave.unsqueeze(1), self.kernels,
                                    stride=self.hop, padding=self.n_fft // 2)
        real, imag = spec.chunk(2, dim=1)
        power = real ** 2 + imag ** 2                     # [batch, bins, frames]
        logmel = torch.log(torch.matmul(self.mel, power) + 1e-6)
        # Normalise each clip so loudness does not matter.
        mean = logmel.mean(dim=(1, 2), keepdim=True)
        std = logmel.std(dim=(1, 2), keepdim=True)
        return ((logmel - mean) / (std + 1e-5)).unsqueeze(1)


def conv_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2),
    )


class PhonemeNet(nn.Module):
    def __init__(self, n_initials: int, n_finals: int, audio_cfg: dict,
                 channels: list[int]):
        super().__init__()
        self.features = LogMel(audio_cfg["sample_rate"], audio_cfg["n_fft"],
                               audio_cfg["hop"], audio_cfg["n_mels"])
        sizes = [1] + list(channels)
        self.trunk = nn.Sequential(*[conv_block(a, b) for a, b in zip(sizes, sizes[1:])])
        self.dropout = nn.Dropout(0.2)
        self.initial_head = nn.Linear(channels[-1], n_initials)
        self.final_head = nn.Linear(channels[-1], n_finals)

    def forward(self, wave: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns raw scores (logits) for the initial and for the final."""
        x = self.trunk(self.features(wave))
        x = self.dropout(x.mean(dim=(2, 3)))             # global average pool
        return self.initial_head(x), self.final_head(x)
