"""Measure a trained run on speakers it has never heard.

    python -m pinyin_phoneme.evaluate --run runs/2026-10-02_baseline
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import DATA_DIR
from .dataset import ClipDataset
from .train import build_model, pick_device


def load_run(run_dir: Path, device: str = "cpu"):
    checkpoint = torch.load(run_dir / "best.pt", map_location=device, weights_only=False)
    model = build_model(checkpoint["config"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    run_dir, data_dir = Path(args.run), Path(args.data)
    device = pick_device()
    model, checkpoint = load_run(run_dir, device)
    initials, finals = checkpoint["initials"], checkpoint["finals"]

    dataset = ClipDataset(data_dir / "splits" / f"{args.split}.csv", data_dir / "processed",
                          checkpoint["num_samples"])
    loader = DataLoader(dataset, batch_size=256)

    predicted_i, predicted_f = [], []
    with torch.no_grad():
        for wave, _, _ in loader:
            initial_logits, final_logits = model(wave.to(device))
            predicted_i += initial_logits.argmax(1).tolist()
            predicted_f += final_logits.argmax(1).tolist()

    by_speaker = defaultdict(lambda: [0, 0, 0, 0])   # initial, final, both, total
    confusions = {"initial": Counter(), "final": Counter()}
    for row, pi, pf in zip(dataset.rows, predicted_i, predicted_f):
        ok_i, ok_f = initials[pi] == row["initial"], finals[pf] == row["final"]
        tally = by_speaker[row["speaker"]]
        tally[0] += ok_i
        tally[1] += ok_f
        tally[2] += ok_i and ok_f
        tally[3] += 1
        if not ok_i:
            confusions["initial"][(row["initial"], initials[pi])] += 1
        if not ok_f:
            confusions["final"][(row["final"], finals[pf])] += 1

    totals = [sum(t[k] for t in by_speaker.values()) for k in range(4)]
    n = totals[3]
    print(f"{args.split} split: {n} clips from {len(by_speaker)} speaker(s)")
    print(f"  initial {totals[0] / n:.1%}   final {totals[1] / n:.1%}   both {totals[2] / n:.1%}")
    for speaker, t in sorted(by_speaker.items()):
        print(f"  {speaker}: initial {t[0] / t[3]:.1%}  final {t[1] / t[3]:.1%}  both {t[2] / t[3]:.1%}")
    for kind, counter in confusions.items():
        print(f"Most common {kind} mistakes (said -> heard as):")
        for (truth, guess), count in counter.most_common(8):
            print(f"  {truth} -> {guess}: {count}")

    metrics = {
        "split": args.split, "clips": n,
        "initial": totals[0] / n, "final": totals[1] / n, "both": totals[2] / n,
        "by_speaker": {s: {"initial": t[0] / t[3], "final": t[1] / t[3], "both": t[2] / t[3]}
                       for s, t in by_speaker.items()},
        "confusions": {kind: [[a, b, c] for (a, b), c in counter.most_common(30)]
                       for kind, counter in confusions.items()},
    }
    (run_dir / f"metrics_{args.split}.json").write_text(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
