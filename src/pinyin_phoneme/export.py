"""Turn a trained run into the files the app loads.

    python -m pinyin_phoneme.export --run runs/2026-10-02_baseline

Writes exports/pinyin_phoneme.onnx and exports/labels.json, then runs the same
audio through PyTorch and through ONNX Runtime to confirm they agree.

The ONNX model:
  input   "audio"          float32 [batch, num_samples]  raw 16 kHz mono, values in -1..1
  output  "initial_probs"  float32 [batch, n_initials]   probabilities, sum to 1
  output  "final_probs"    float32 [batch, n_finals]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from .config import EXPORTS_DIR
from .evaluate import load_run


class WithProbabilities(nn.Module):
    """Wraps the model so the exported file returns probabilities instead of raw scores."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, audio: torch.Tensor):
        initial_logits, final_logits = self.model(audio)
        return initial_logits.softmax(dim=1), final_logits.softmax(dim=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--out", default=str(EXPORTS_DIR))
    args = parser.parse_args()

    model, checkpoint = load_run(Path(args.run))
    wrapped = WithProbabilities(model).eval()
    num_samples = checkpoint["num_samples"]
    sample_rate = checkpoint["config"]["audio"]["sample_rate"]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = out_dir / "pinyin_phoneme.onnx"

    example = torch.randn(1, num_samples) * 0.1
    torch.onnx.export(
        wrapped, (example,), str(onnx_path),
        input_names=["audio"], output_names=["initial_probs", "final_probs"],
        dynamic_axes={"audio": {0: "batch"}, "initial_probs": {0: "batch"}, "final_probs": {0: "batch"}},
        opset_version=17, dynamo=False,
    )

    (out_dir / "labels.json").write_text(json.dumps({
        "sample_rate": sample_rate,
        "num_samples": num_samples,
        "initials": checkpoint["initials"],
        "finals": checkpoint["finals"],
    }, indent=2))

    # Check: same audio through both, outputs must match.
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    test_audio = torch.randn(3, num_samples) * 0.1
    with torch.no_grad():
        expected = wrapped(test_audio)
    actual = session.run(None, {"audio": test_audio.numpy()})
    worst = max(float(np.abs(e.numpy() - a).max()) for e, a in zip(expected, actual))
    if worst > 1e-4:
        raise SystemExit(f"ONNX output differs from PyTorch by {worst:.2e}; do not ship this file.")

    size_kb = onnx_path.stat().st_size / 1024
    print(f"Wrote {onnx_path} ({size_kb:.0f} KB) and {out_dir / 'labels.json'}")
    print(f"Input: {num_samples} samples ({num_samples / sample_rate:.1f}s at {sample_rate} Hz)")
    print(f"ONNX matches PyTorch (largest difference {worst:.1e})")


if __name__ == "__main__":
    main()
