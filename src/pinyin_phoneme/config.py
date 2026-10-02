"""Where everything lives. Import folders from here instead of typing paths.

    from pinyin_phoneme.config import STANDARD_DIR, RUNS_DIR

All paths are absolute and worked out from this file's own location, so they
are correct no matter which directory a script is run from.
"""

from pathlib import Path

# This file is <repo>/src/pinyin_phoneme/config.py, so the repo is two folders up.
REPO_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = REPO_ROOT / "configs"          # training settings (yaml)
DEFAULT_CONFIG = CONFIGS_DIR / "baseline.yaml"

DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                   # exactly as downloaded, never edited
STANDARD_DIR = DATA_DIR / "standard"         # one folder per speaker, files named ma1, lv3...
PROCESSED_DIR = DATA_DIR / "processed"       # cleaned 16 kHz clips + manifest.csv
SPLITS_DIR = DATA_DIR / "splits"             # train.csv, val.csv, test.csv

RUNS_DIR = REPO_ROOT / "runs"                # one folder per training run
EXPORTS_DIR = REPO_ROOT / "exports"          # the .onnx file and labels.json
