"""Train the model. Each run gets its own folder under runs/.

    python -m pinyin_phoneme.train --config configs/baseline.yaml --name baseline
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import date
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from .config import DATA_DIR, DEFAULT_CONFIG, RUNS_DIR
from .dataset import ClipDataset
from .labels import FINALS, INITIALS
from .model import PhonemeNet


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_num_samples(cfg: dict, data_dir: Path) -> int:
    """Model input length in samples: from the config, or from what prepare_data measured."""
    seconds = cfg["audio"]["clip_seconds"]
    if seconds is None:
        seconds = json.loads((data_dir / "processed" / "stats.json").read_text())["clip_seconds"]
    return int(round(seconds * cfg["audio"]["sample_rate"]))


def build_model(cfg: dict) -> PhonemeNet:
    return PhonemeNet(len(INITIALS), len(FINALS), cfg["audio"], cfg["model"]["channels"])


@torch.no_grad()
def accuracy(model: nn.Module, loader: DataLoader, device: str) -> dict[str, float]:
    """Share of clips with the right initial, the right final, and both."""
    model.eval()
    hits = torch.zeros(3)
    total = 0
    for wave, initial, final in loader:
        initial_logits, final_logits = model(wave.to(device))
        ok_i = initial_logits.argmax(1).cpu() == initial
        ok_f = final_logits.argmax(1).cpu() == final
        hits += torch.tensor([ok_i.sum(), ok_f.sum(), (ok_i & ok_f).sum()])
        total += len(wave)
    initial_acc, final_acc, both_acc = (hits / max(total, 1)).tolist()
    return {"initial": initial_acc, "final": final_acc, "both": both_acc}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", default=str(DATA_DIR))
    parser.add_argument("--name", default="baseline")
    parser.add_argument("--epochs", type=int, help="override the config (for quick tests)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    train_cfg = cfg["train"]
    epochs = args.epochs or train_cfg["epochs"]
    data_dir = Path(args.data)
    torch.manual_seed(train_cfg["seed"])
    device = pick_device()
    num_samples = resolve_num_samples(cfg, data_dir)

    run_dir = RUNS_DIR / f"{date.today()}_{args.name}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, run_dir / "config.yaml")

    processed = data_dir / "processed"
    train_set = ClipDataset(data_dir / "splits" / "train.csv", processed, num_samples, cfg["augment"])
    val_set = ClipDataset(data_dir / "splits" / "val.csv", processed, num_samples)
    workers = train_cfg["num_workers"]
    train_loader = DataLoader(train_set, batch_size=train_cfg["batch_size"], shuffle=True,
                              num_workers=workers, drop_last=len(train_set) > train_cfg["batch_size"])
    val_loader = DataLoader(val_set, batch_size=256, num_workers=workers)

    model = build_model(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"{len(train_set)} training clips, {len(val_set)} validation clips, "
          f"{n_params:,} parameters, device: {device}, run folder: {run_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"],
                                  weight_decay=train_cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=train_cfg["lr"], total_steps=epochs * len(train_loader))
    loss_fn = nn.CrossEntropyLoss()

    best = -1.0
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.time()
        running, batches = 0.0, 0
        for wave, initial, final in train_loader:
            wave, initial, final = wave.to(device), initial.to(device), final.to(device)
            initial_logits, final_logits = model(wave)
            loss = loss_fn(initial_logits, initial) + loss_fn(final_logits, final)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running += loss.item()
            batches += 1

        val = accuracy(model, val_loader, device)
        entry = {"epoch": epoch, "train_loss": running / batches, **{f"val_{k}": v for k, v in val.items()}}
        log.append(entry)
        (run_dir / "log.json").write_text(json.dumps(log, indent=2))

        score = (val["initial"] + val["final"]) / 2
        marker = ""
        if score > best:
            best = score
            torch.save({"model": model.state_dict(), "config": cfg, "num_samples": num_samples,
                        "initials": INITIALS, "finals": FINALS, "epoch": epoch},
                       run_dir / "best.pt")
            marker = "  (saved)"
        print(f"epoch {epoch:3d}  loss {entry['train_loss']:.3f}  "
              f"val initial {val['initial']:.1%}  final {val['final']:.1%}  both {val['both']:.1%}  "
              f"{time.time() - started:.0f}s{marker}")

    print(f"\nBest checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
