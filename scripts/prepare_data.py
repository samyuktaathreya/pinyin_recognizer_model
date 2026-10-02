"""data/standard -> data/processed + data/splits.

For every audio file under data/standard (made by standardize.py):
  1. work out speaker / syllable / tone from its path (source_format.parse_path)
  2. convert to 16 kHz mono and trim silence
  3. save as wav under data/processed and add a row to manifest.csv

Then measure clip lengths, and split the clips into train / val / test by speaker.

    python scripts/prepare_data.py --config configs/baseline.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

from pinyin_phoneme.audio import AUDIO_EXTENSIONS, load_audio, trim_silence
from pinyin_phoneme.config import DATA_DIR, DEFAULT_CONFIG
from pinyin_phoneme.labels import FINALS, INITIALS, split_syllable
from pinyin_phoneme.source_format import parse_path

FIELDS = ["path", "speaker", "syllable", "initial", "final", "tone", "seconds"]


def choose_splits(speakers: list[str], data_cfg: dict) -> dict[str, list[str]]:
    val, test = data_cfg.get("val_speakers"), data_cfg.get("test_speakers")
    if val is None or test is None:
        if len(speakers) < 3:
            raise SystemExit("Need at least 3 speakers to hold some out for validation and test.")
        shuffled = sorted(speakers)
        random.Random(data_cfg.get("seed", 0)).shuffle(shuffled)
        n_val = 2 if len(speakers) >= 6 else 1
        test = shuffled[:1] if test is None else test
        remaining = [s for s in shuffled if s not in test]
        val = remaining[:n_val] if val is None else val
    unknown = (set(val) | set(test)) - set(speakers)
    if unknown:
        raise SystemExit(f"Config names speakers that are not in the data: {sorted(unknown)}")
    if set(val) & set(test):
        raise SystemExit("A speaker cannot be in both val and test.")
    train = [s for s in speakers if s not in val and s not in test]
    return {"train": train, "val": list(val), "test": list(test)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", default=str(DATA_DIR), help="folder containing standard/")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    sample_rate = cfg["audio"]["sample_rate"]
    data_dir = Path(args.data)
    raw_dir, processed_dir, splits_dir = (data_dir / d for d in ("standard", "processed", "splits"))

    files = sorted(
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in AUDIO_EXTENSIONS and ".git" not in p.parts
    )
    if not files:
        raise SystemExit(f"No audio files found under {raw_dir}")

    rows, skipped = [], []
    for i, path in enumerate(files):
        rel = path.relative_to(raw_dir)
        try:
            parsed = parse_path(rel)
            if parsed is None:
                raise ValueError("path not recognised by parse_path")
            speaker, syllable, tone = parsed
            initial, final = split_syllable(syllable)
            wave = trim_silence(load_audio(path, sample_rate), sample_rate)
            if len(wave) < 0.05 * sample_rate:
                raise ValueError("almost no audio after trimming")
        except Exception as err:  # one bad file should not stop the run
            skipped.append((str(rel), str(err)))
            continue

        out_rel = Path(speaker) / f"{syllable}{tone or ''}_{i:06d}.wav"
        out_path = processed_dir / out_rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), wave, sample_rate, subtype="PCM_16")
        rows.append({
            "path": out_rel.as_posix(), "speaker": speaker, "syllable": syllable,
            "initial": initial, "final": final, "tone": tone or "",
            "seconds": round(len(wave) / sample_rate, 3),
        })
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(files)} files")

    if not rows:
        print("First few failures:")
        for rel, err in skipped[:10]:
            print(f"  {rel}: {err}")
        raise SystemExit("No files could be used. Adjust parse_path() in source_format.py.")

    def write_csv(path: Path, subset: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)

    write_csv(processed_dir / "manifest.csv", rows)

    # Clip length: long enough for 99% of clips plus a little room, rounded up to 0.1 s.
    seconds = np.array([r["seconds"] for r in rows])
    p50, p99 = np.percentile(seconds, [50, 99])
    clip_seconds = math.ceil((p99 + 0.1) * 10) / 10
    stats = {
        "clips": len(rows), "median_seconds": round(float(p50), 3),
        "p99_seconds": round(float(p99), 3), "max_seconds": float(seconds.max()),
        "clip_seconds": clip_seconds,
    }
    (processed_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    speakers = sorted({r["speaker"] for r in rows})
    splits = choose_splits(speakers, cfg["data"])
    for name, members in splits.items():
        write_csv(splits_dir / f"{name}.csv", [r for r in rows if r["speaker"] in members])

    # Report
    print(f"\nUsed {len(rows)} of {len(files)} files; skipped {len(skipped)}.")
    for rel, err in skipped[:10]:
        print(f"  skipped {rel}: {err}")
    if len(skipped) > 10:
        print(f"  ... and {len(skipped) - 10} more")
    per_speaker = Counter(r["speaker"] for r in rows)
    print(f"\n{len(speakers)} speakers:")
    for name, members in splits.items():
        print(f"  {name:5s} " + ", ".join(f"{s} ({per_speaker[s]})" for s in members))
    print(f"\nTrimmed clip length: median {p50:.2f}s, 99% under {p99:.2f}s, max {seconds.max():.2f}s")
    print(f"Model input length: {clip_seconds}s")

    train_rows = [r for r in rows if r["speaker"] in splits["train"]]
    for kind, labels in (("initial", INITIALS), ("final", FINALS)):
        counts = Counter(r[kind] for r in train_rows)
        missing = [label for label in labels if counts[label] == 0]
        fewest = min(labels, key=lambda label: counts[label])
        print(f"{kind}s in training data: fewest examples is {fewest!r} ({counts[fewest]})"
              + (f"; NONE for {missing}" if missing else ""))


if __name__ == "__main__":
    main()
